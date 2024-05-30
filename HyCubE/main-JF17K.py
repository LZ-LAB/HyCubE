# -- coding: utf-8 --

from data_process import Data
from modelHyCubE import HyCubE
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import torch
import argparse
import time
import optuna

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

class Experiment:
    def __init__(self, num_iterations, batch_size, lr, dr, input_drop, dropout, dropout_3d, padding, dembd, dembd1, max_ary):
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.lr, self.dr = lr, dr
        self.input_drop, self.dropout, self.dropout_3d = input_drop, dropout, dropout_3d
        self.padding = padding
        self.dembd, self.dembd1 = dembd, dembd1
        self.max_ary = max_ary
        self.device = device

    def get_batch(self, er_vocab, er_vocab_pairs, idx, miss_ent_domain):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = torch.zeros((len(batch), len(d.ent2id)), device=device)
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        batch = torch.tensor(batch, dtype=torch.long).to(device)
        r_idx = batch[:, 0]
        e_idx = batch[:, [i for i in range(1, batch.shape[1]) if i != miss_ent_domain]]
        return batch, targets, r_idx, e_idx

    def get_test_batch(self, test_data_idxs, idx, miss_ent_domain):
        batch = torch.tensor(test_data_idxs[idx:idx+self.batch_size], dtype=torch.long).to(device)
        r_idx = batch[:, 0]
        e_idx = batch[:, [i for i in range(1, batch.shape[1]) if i != miss_ent_domain]]
        return batch, r_idx, e_idx

    def evaluate(self, model, test_data_idxs, ary_test):
        hits, ranks = [], []
        group_hits, group_ranks = [[] for _ in ary_test], [[] for _ in ary_test]
        for _ in [1, 3, 10]:
            hits.append([])
            for h in group_hits:
                h.append([])

        ind = 0
        for ary in ary_test:

            if len(test_data_idxs[ary-2]) > 0:
                for miss_ent_domain in range(1, ary+1):
                    er_vocab = d.all_er_vocab_list[ary-2][miss_ent_domain-1]
                    for i in range(0, len(test_data_idxs[ary-2]), self.batch_size):
                        data_batch, r_idx, e_idx = self.get_test_batch(test_data_idxs[ary-2], i, miss_ent_domain)
                        pred = model.forward(r_idx, e_idx, miss_ent_domain)

                        for j in range(data_batch.shape[0]):
                            er_vocab_key = []
                            for k0 in range(data_batch.shape[1]):
                                er_vocab_key.append(data_batch[j][k0].item())
                            er_vocab_key[miss_ent_domain] = miss_ent_domain * 111111

                            filt = er_vocab[tuple(er_vocab_key)]
                            target_value = pred[j, data_batch[j][miss_ent_domain]].item()
                            pred[j, filt] = -1e10
                            pred[j, data_batch[j][miss_ent_domain]] = target_value

                        sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
                        sort_idxs = sort_idxs.cpu().numpy()

                        for j in range(pred.shape[0]):
                            rank = np.where(sort_idxs[j] == data_batch[j][miss_ent_domain].item())[0][0]
                            ranks.append(rank + 1)
                            group_ranks[ind].append(rank + 1)
                            for id, hits_level in enumerate([1, 3, 10]):
                                if rank + 1 <= hits_level:
                                    hits[id].append(1.0)
                                    group_hits[ind][id].append(1.0)
                                else:
                                    hits[id].append(0.0)
                                    group_hits[ind][id].append(0.0)

            ind += 1

        t_MRR = np.mean(1. / np.array(ranks))
        t_hit10, t_hit3, t_hit1 = np.mean(hits[2]), np.mean(hits[1]), np.mean(hits[0])
        group_MRR = [np.mean(1. / np.array(x)) for x in group_ranks]
        group_HitsRatio = [[] for _ in ary_test]
        for i in range(0, len(group_HitsRatio)):
            for id in range(0, len([1, 3, 10])):
                group_HitsRatio[i].append(np.mean(group_hits[i][id]))
        return t_MRR, t_hit10, t_hit3, t_hit1, group_MRR, group_HitsRatio


    def train_and_eval(self):

        model = HyCubE(len(d.ent2id), len(d.rel2id), self.input_drop, self.dropout, self.dropout_3d, self.padding, self.dembd, self.dembd1, self.max_ary, self.device)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        if self.dr:
            scheduler = ExponentialLR(opt, self.dr)

        print('Training Starts...')
        test_mrr, test_hits = [], []
        best_valid_iter = 0
        best_valid_metric = {'valid_mrr': -1, 'test_mrr': -1, 'test_hit1': -1, 'test_hit3': -1, 'test_hit10': -1, 'group_test_mrr':[], 'group_test_hits':[]}

        ary_er_vocab_list = []
        ary_er_vocab_pair_list = [[] for _ in range(2, self.max_ary+1)]
        for ary in range(2, self.max_ary+1):
            ary_er_vocab_list.append(d.train_er_vocab_list[ary-2])
            for miss_ent_domain in range(1, ary+1):
                ary_er_vocab_pair_list[ary-2].append(list(d.train_er_vocab_list[ary-2][miss_ent_domain-1].keys()))

        mrr_lst = []
        hit1_lst = []
        hit3_lst = []
        hit10_lst = []
        loss_figure = []

        for it in range(1, self.num_iterations + 1):
            model.train()
            losses = []
            print('\nEpoch %d starts training...' % it)
            for ary in args.ary_list:
                for er_vocab_pairs in ary_er_vocab_pair_list[ary-2]:
                    np.random.shuffle(er_vocab_pairs)
                for miss_ent_domain in range(1, ary+1):
                    er_vocab = ary_er_vocab_list[ary-2][miss_ent_domain-1]
                    er_vocab_pairs = ary_er_vocab_pair_list[ary-2][miss_ent_domain-1]
                    for j in range(0, len(er_vocab_pairs), self.batch_size):
                        data_batch, label, rel_idx, ent_idx = self.get_batch(er_vocab, er_vocab_pairs, j, miss_ent_domain)
                        # ents_idx = data_batch[:, 1:]
                        pred = model.forward(rel_idx, ent_idx, miss_ent_domain)
                        pred = pred.to(device)
                        loss = model.loss(pred, label)


                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        losses.append(loss.item())

            if self.dr:
                scheduler.step()
            print('Epoch %d train, Loss=%f' % (it, np.mean(losses)))


            if it % param['eval_step'] == 0:
                model.eval()
                with torch.no_grad():
                    print('\n ~~~~~~~~~~~~~ Valid ~~~~~~~~~~~~~~~~')
                    v_mrr, v_hit10, v_hit3, v_hit1, _, _ = self.evaluate(model, d.valid_facts, args.ary_list)
                    mrr_lst.append(v_mrr)
                    hit1_lst.append(v_hit1)
                    hit3_lst.append(v_hit3)
                    hit10_lst.append(v_hit10)
                    loss_figure.append(np.mean(losses))
                    print('~~~~~~~~~~~~~ Test ~~~~~~~~~~~~~~~~')

                    t_mrr, t_hit10, t_hit3, t_hit1, group_mrr, group_hits = self.evaluate(model, d.test_facts, args.ary_list)

                    if v_mrr >= best_valid_metric['valid_mrr']:
                        best_valid_iter = it
                        best_valid_metric['valid_mrr'] = v_mrr
                        best_valid_metric['test_mrr'] = t_mrr
                        best_valid_metric['test_hit10'], best_valid_metric['test_hit3'], best_valid_metric['test_hit1'] = t_hit10, t_hit3, t_hit1
                        best_valid_metric['group_test_hits'] = group_hits
                        best_valid_metric['group_test_mrr'] = group_mrr
                        print('Epoch=%d, Valid MRR increases.' % it)
                    else:
                        print('Valid MRR didnt increase for %d epochs, Best_MRR=%f' % (it-best_valid_iter, best_valid_metric['test_mrr']))

                    if it - best_valid_iter >= param['valid_patience'] or it == self.num_iterations:
                        print('++++++++++++ Early Stopping +++++++++++++')
                        for i, ary in enumerate(args.ary_list):
                            print('Testing Arity:%d, MRR=%f, Hits@10=%f, Hits@3=%f, Hits@1=%f' % (
                            ary, best_valid_metric['group_test_mrr'][i], best_valid_metric['group_test_hits'][i][2],
                            best_valid_metric['group_test_hits'][i][1], best_valid_metric['group_test_hits'][i][0]))

                        print('Best epoch %d' % best_valid_iter)
                        print('Hits @10: {0}'.format(best_valid_metric['test_hit10']))
                        print('Hits @3: {0}'.format(best_valid_metric['test_hit3']))
                        print('Hits @1: {0}'.format(best_valid_metric['test_hit1']))
                        print('Mean reciprocal rank: {0}'.format(best_valid_metric['test_mrr']))

                        # print(mrr_lst)
                        with open("mrr_10iter_Wi.txt", "w") as f:
                            for mrr in mrr_lst:
                                f.write(str(mrr) + ",")
                            f.close()

                        with open("hit1_10iter_Wi.txt", "w") as f:
                            for hit1 in hit1_lst:
                                f.write(str(hit1) + ",")
                            f.close()
                        with open("hit3_10iter_Wi.txt", "w") as f:
                            for hit3 in hit3_lst:
                                f.write(str(hit3) + ",")
                            f.close()
                        with open("hit10_10iter_Wi.txt", "w") as f:
                            for hit10 in hit10_lst:
                                f.write(str(hit10) + ",")
                            f.close()
                        with open("losses_10iter_Wi.txt", "w") as f:
                            for loss in loss_figure:
                                f.write(str(loss) + ",")
                            f.close()


                        return best_valid_metric['test_mrr']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="JF17K", nargs="?", help="FB-AUTO/JF17K/WikiPeople/WN18RR/FB15K-237.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=384, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.00095, nargs="?", help="Learning rate.")
    parser.add_argument("--dr", type=float, default=0.993, nargs="?", help="Decay rate.")
    
    parser.add_argument("--input_drop", type=float, default=0.3, nargs="?", help="input_drop.")
    parser.add_argument("--dropout", type=float, default=0.9, nargs="?", help="dropout.")
    parser.add_argument("--dropout_3d", type=float, default=0.5, nargs="?", help="dropout_3d.")
    
    parser.add_argument("--padding", type=int, default=1, nargs="?", help="Convolutional padding (conv_kernel size).")
    
    parser.add_argument("--dembed", type=int, default=400, nargs="?")
    parser.add_argument("--dembed1", type=int, default=20, nargs="?")

    parser.add_argument("--eval_step", type=int, default=10, nargs="?", help="Evaluation step.")
    parser.add_argument("--valid_patience", type=int, default=50, nargs="?", help="Valid patience.")
    parser.add_argument("-ary", "--ary_list", type=int, action='append', help="List of arity for train and test")
    args = parser.parse_args()


    ### 2-ary
    # args.ary_list = [2]
    
    ### n-ary
    ## JF17K
    args.ary_list = [2, 3, 4, 5, 6]
    ## WikiPeople
    # args.ary_list = [2, 3, 4, 5, 6, 7, 8, 9]
    ## FB-AUTO
    # args.ary_list = [2, 4, 5]

    param = {}
    param['dataset'] = args.dataset
    param['num_iterations'], param['eval_step'], param['valid_patience'] = args.num_iterations, args.eval_step, args.valid_patience
    param['batch_size'] = args.batch_size
    param['lr'], param['dr'] = args.lr, args.dr
    
    param['input_drop'], param['dropout'], param['dropout_3d'] = args.input_drop, args.dropout, args.dropout_3d
    
    param['padding'] = args.padding
    
    param['dembed'], param['dembed1'] = args.dembed, args.dembed1


    # Reproduce Results
    torch.backends.cudnn.deterministic = True
    seed = 2024
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    data_dir = "./data/%s/" % param['dataset']
    print('\nLoading data...')
    d = Data(data_dir=data_dir)

    Exp = Experiment(num_iterations=param['num_iterations'], batch_size=param['batch_size'], lr=param['lr'], dr=param['dr'],
                     input_drop=param['input_drop'], dropout=param['dropout'], dropout_3d=param['dropout_3d'], padding=param['padding'], 
                     dembd=param['dembed'], dembd1=param['dembed1'], max_ary=d.max_ary)
    Exp.train_and_eval()
