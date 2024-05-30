# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_hit1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def forward(self, pred1, tar1):
        pred1 = F.softmax(pred1, dim=1)
        loss = -torch.log(pred1[tar1 == 1]).sum()
        return loss

class HyCRE(BaseClass):

    def __init__(self, n_ent, n_rel, input_drop, dropout, dropout_3d, padding, emb_dim, emb_dim1, max_arity, device):
        super(HyCRE, self).__init__()
        self.loss = MyLoss()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.device = device
        self.emb_dim = emb_dim
        self.emb_dim1 = emb_dim1
        self.emb_dim2 = emb_dim // emb_dim1
        self.max_arity = max_arity
        
        self.input_drop = nn.Dropout(input_drop) # input_drop 0.2
        self.dropout = nn.Dropout(dropout) # hidden_drop 0.2
        self.dropout_3d = nn.Dropout(dropout_3d) # feature_map_drop 0.3

        
        self.padding = padding
        self.k_size = 2*self.padding + 1
            
        self.ent_embeddings = nn.Parameter(torch.Tensor(self.n_ent, self.emb_dim))
        self.rel_embeddings = nn.Parameter(torch.Tensor(self.n_rel, self.emb_dim))
        self.pos_embeddings = nn.Embedding(self.max_arity, self.emb_dim)
        
        self.conv_layer_2a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 2), padding=((0,0,0)))
        self.conv_layer_3a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 4), padding=((0,0,0)))
        self.conv_layer_4a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 6), padding=((0,0,0)))
        self.conv_layer_5a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 8), padding=((0,0,0)))
        self.conv_layer_6a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 10), padding=((0,0,0)))
        self.conv_layer_7a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 12), padding=((0,0,0)))
        self.conv_layer_8a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 14), padding=((0,0,0)))
        self.conv_layer_9a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 16), padding=((0,0,0)))




        
        self.pool = torch.nn.MaxPool3d((4, 1, 1))
        
        self.conv_size = (self.emb_dim1 * self.emb_dim2) * 8 // 4
        # self.conv_size = (self.emb_dim1 * self.emb_dim2) * 8
        # self.conv_size = ((self.emb_dim1 + 2*self.padding) * (self.emb_dim2 + 2*self.padding)) * 8 // 4
        
        self.fc_layer = nn.Linear(in_features=self.conv_size, out_features=self.emb_dim)

        self.W2 = nn.Linear(2*self.emb_dim1*self.emb_dim2, 2*self.emb_dim1*self.emb_dim2)
        self.W3 = nn.Linear(4*self.emb_dim1*self.emb_dim2, 2*self.emb_dim1*self.emb_dim2)
        self.W4 = nn.Linear(6*self.emb_dim1*self.emb_dim2, 2*self.emb_dim1*self.emb_dim2)
        self.W5 = nn.Linear(8*self.emb_dim1*self.emb_dim2, 2*self.emb_dim1*self.emb_dim2)
        self.W6 = nn.Linear(10*self.emb_dim1*self.emb_dim2, 2*self.emb_dim1*self.emb_dim2)
        self.W7 = nn.Linear(12*self.emb_dim1*self.emb_dim2, 2*self.emb_dim1*self.emb_dim2)
        self.W8 = nn.Linear(14*self.emb_dim1*self.emb_dim2, 2*self.emb_dim1*self.emb_dim2)
        self.W9 = nn.Linear(16*self.emb_dim1*self.emb_dim2, 2*self.emb_dim1*self.emb_dim2)

        self.bn1 = nn.BatchNorm3d(num_features=1)
        # self.bn2 = nn.BatchNorm3d(num_features=4)
        # self.bn3 = nn.BatchNorm2d(num_features=32)
        # self.bn4 = nn.BatchNorm1d(num_features=self.conv_size)
        self.register_parameter('b', nn.Parameter(torch.zeros(n_ent)))



        # Initialization
        nn.init.xavier_uniform_(self.ent_embeddings.data)
        nn.init.xavier_uniform_(self.rel_embeddings.data)
        nn.init.xavier_uniform_(self.pos_embeddings.weight.data)        
        nn.init.xavier_uniform_(self.conv_layer_2a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_3a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_4a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_5a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_6a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_7a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_8a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_9a.weight.data)
        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        nn.init.xavier_uniform_(self.W2.weight.data)
        nn.init.xavier_uniform_(self.W3.weight.data)
        nn.init.xavier_uniform_(self.W4.weight.data)
        nn.init.xavier_uniform_(self.W5.weight.data)
        nn.init.xavier_uniform_(self.W6.weight.data)
        nn.init.xavier_uniform_(self.W7.weight.data)
        nn.init.xavier_uniform_(self.W8.weight.data)
        nn.init.xavier_uniform_(self.W9.weight.data)        
        
        

    def circular_padding(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)
        left_pad = temp[..., -padding:]
        right_pad	= temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded     
    


    def conv3d_circular_alternate(self, concat_input):
        r = concat_input[:, 0, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
        rc = self.circular_padding(r, self.padding)


        if concat_input.shape[1] == 2:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            ec1 = self.circular_padding(e1, self.padding)
            
            cube = torch.cat((rc, ec1), dim=1)
            
            res = torch.cat((r, e1), dim=1)
            res = res.view(-1, 1, 2*self.emb_dim1*self.emb_dim2)
            res = self.W2(res)            
            
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_2a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 3:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            ec1 = self.circular_padding(e1, self.padding)
            ec2 = self.circular_padding(e2, self.padding)
                                   
            cube = torch.cat((rc, ec1, rc, ec2), dim=1)
            
            res = torch.cat((r, e1, r, e2), dim=1)
            res = res.view(-1, 1, 4*self.emb_dim1*self.emb_dim2)
            res = self.W3(res)            
            
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_3a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 4:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            ec1 = self.circular_padding(e1, self.padding)
            ec2 = self.circular_padding(e2, self.padding)
            ec3 = self.circular_padding(e3, self.padding)
            
            cube = torch.cat((rc, ec1, rc, ec2, rc, ec3), dim=1)
            
            res = torch.cat((r, e1, r, e2, r, e3), dim=1)
            res = res.view(-1, 1, 6*self.emb_dim1*self.emb_dim2)
            res = self.W4(res)            
            
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_4a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 5:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            ec1 = self.circular_padding(e1, self.padding)
            ec2 = self.circular_padding(e2, self.padding)
            ec3 = self.circular_padding(e3, self.padding)
            ec4 = self.circular_padding(e4, self.padding)
            
            cube = torch.cat((rc, ec1, rc, ec2, rc, ec3, rc, ec4), dim=1)
            
            res = torch.cat((r, e1, r, e2, r, e3, r, e4), dim=1)
            res = res.view(-1, 1, 8*self.emb_dim1*self.emb_dim2)
            res = self.W5(res)                        
            
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_5a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 6:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            ec1 = self.circular_padding(e1, self.padding)
            ec2 = self.circular_padding(e2, self.padding)
            ec3 = self.circular_padding(e3, self.padding)
            ec4 = self.circular_padding(e4, self.padding)
            ec5 = self.circular_padding(e5, self.padding)
            
            cube = torch.cat((rc, ec1, rc, ec2, rc, ec3, rc, ec4, rc, ec5), dim=1)
            
            res = torch.cat((r, e1, r, e2, r, e3, r, e4, r, e5), dim=1)
            res = res.view(-1, 1, 10*self.emb_dim1*self.emb_dim2)
            res = self.W6(res)            
            
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_6a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 7:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            ec1 = self.circular_padding(e1, self.padding)
            ec2 = self.circular_padding(e2, self.padding)
            ec3 = self.circular_padding(e3, self.padding)
            ec4 = self.circular_padding(e4, self.padding)
            ec5 = self.circular_padding(e5, self.padding)
            ec6 = self.circular_padding(e6, self.padding)
            
            cube = torch.cat((rc, ec1, rc, ec2, rc, ec3, rc, ec4, rc, ec5, rc, ec6), dim=1)
            
            res = torch.cat((r, e1, r, e2, r, e3, r, e4, r, e5, r, e6), dim=1)
            res = res.view(-1, 1, 12*self.emb_dim1*self.emb_dim2)
            res = self.W7(res)            
            
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_7a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 8:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e7 = concat_input[:, 7, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            ec1 = self.circular_padding(e1, self.padding)
            ec2 = self.circular_padding(e2, self.padding)
            ec3 = self.circular_padding(e3, self.padding)
            ec4 = self.circular_padding(e4, self.padding)
            ec5 = self.circular_padding(e5, self.padding)
            ec6 = self.circular_padding(e6, self.padding)
            ec7 = self.circular_padding(e7, self.padding)
            
            cube = torch.cat((rc, ec1, rc, ec2, rc, ec3, rc, ec4, rc, ec5, rc, ec6, rc, ec7), dim=1)
            
            res = torch.cat((r, e1, r, e2, r, e3, r, e4, r, e5, r, e6, r, e7), dim=1)
            res = res.view(-1, 1, 14*self.emb_dim1*self.emb_dim2)
            res = self.W8(res)            
            
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_8a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 9:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e7 = concat_input[:, 7, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e8 = concat_input[:, 8, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            ec1 = self.circular_padding(e1, self.padding)
            ec2 = self.circular_padding(e2, self.padding)
            ec3 = self.circular_padding(e3, self.padding)
            ec4 = self.circular_padding(e4, self.padding)
            ec5 = self.circular_padding(e5, self.padding)
            ec6 = self.circular_padding(e6, self.padding)
            ec7 = self.circular_padding(e7, self.padding)
            ec8 = self.circular_padding(e8, self.padding)
            
            cube = torch.cat((rc, ec1, rc, ec2, rc, ec3, rc, ec4, rc, ec5, rc, ec6, rc, ec7, rc, ec8), dim=1)
            
            res = torch.cat((r, e1, r, e2, r, e3, r, e4, r, e5, r, e6, r, e7, r, e8), dim=1)
            res = res.view(-1, 1, 16*self.emb_dim1*self.emb_dim2)
            res = self.W9(res)            
            
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_9a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        x = x.view(-1, self.conv_size)
        
        res = res.view(-1, self.conv_size)
        xr = x + res
        
        xr = self.dropout_3d(xr)
        return xr



    def forward(self, rel_idx, ent_idx, miss_ent_domain):

        r = self.rel_embeddings[rel_idx].unsqueeze(1)
        ents = self.ent_embeddings[ent_idx]
        
        concat_input = torch.cat((r, ents), dim=1)   
        concat_input = self.input_drop(concat_input) # input_drop
        
        v1 = self.conv3d_circular_alternate(concat_input)
        # v2 = self.conv3d_standard(concat_input)
        
        x= v1
        x = self.dropout(x) #hidden_drop
        x = self.fc_layer(x)

        miss_ent_domain = torch.LongTensor([miss_ent_domain-1]).to(self.device)
        mis_pos = self.pos_embeddings(miss_ent_domain)
        tar_emb = self.ent_embeddings + mis_pos
        scores = torch.mm(x, tar_emb.transpose(0, 1))
        scores += self.b.expand_as(scores)

        return scores
