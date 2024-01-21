# The HJE Model
This paper has been submitted to the SIGKDD 2024.



## Requirements
The version of Python and major packages needed to run the code:
   
    -- python 3.9.16
    -- torch 1.12.0
    -- numpy 1.26.0
    -- tqdm 4.65.0

## How to Run

### 1. Mixed Arity Knowledge Hypergraph
```
python main-JF17K.py                 ## JF17K dataset
python main-WikiPeople.py            ## WikiPeople dataset
python main-FB-AUTO.py               ## FB-AUTO dataset
```

### 2. Fixed Arity Knowledge Hypergraph
```
## JF17K-3 dataset
python main-fixed3.py --dataset JF17K-3 --batch_size 256 --lr 0.0005 --dr 0.995 --dropout 0.1 --dropout_3d 0.2 --padding 2

## JF17K-4 dataset
python main-fixed4.py --dataset JF17K-4 --batch_size 256 --lr 0.0005 --dr 0.995 --dropout 0.6 --dropout_3d 0.5 --padding 3

## WikiPeople-3 dataset
python main-fixed3.py --dataset WikiPeople-3 --batch_size 128 --lr 0.00025 --dr 0.995 --dropout 0.2 --dropout_3d 0.5 --padding 2

## WikiPeople-4 dataset
python main-fixed4.py --dataset WikiPeople-4 --batch_size 128 --lr 0.00025 --dr 0.995 --dropout 0.4 --dropout_3d 0.5 --padding 3
```

### 3. Binary Knowledge Graph
```
## FB15K-237 dataset
python main-fixed2.py --dataset FB15K-237 --batch_size 256 --lr 0.0005 --dr 0.995 --dropout 0.6 --dropout_3d 0.5 --padding 3

## JF17K-2 dataset
python main-fixed2.py --dataset JF17K-2 --batch_size 256 --lr 0.0005 --dr 0.995 --dropout 0.6 --dropout_3d 0.5 --padding 1

## WikiPeople-2 dataset
python main-fixed2.py --dataset WikiPeople-2 --batch_size 128 --lr 0.00025 --dr 0.995 --dropout 0.2 --dropout_3d 0.5 --padding 4

## FB-AUTO-2 dataset
python main-fixed2.py --dataset FB-AUTO-2 --batch_size 128 --lr 0.0005 --dr 0.99 --dropout 0.7 --dropout_3d 0.4 --padding 2
```

## Acknowledgments
We are very grateful for all open-source baseline models:

1. GETD: https://github.com/liuyuaa/GETD
2. HypE/HSimplE: https://github.com/ElementAI/HypE
3. RAM: https://github.com/liuyuaa/RAM
4. HyperMLN: https://github.com/zirui-chen/HyperMLN
5. tNaLP+: https://github.com/gsp2014/NaLP
6. PosKHG: https://github.com/zirui-chen/PosKHG
7. HyConvE: https://github.com/CarllllWang/HyConvE/tree/master
8. RD-MPNN: https://github.com/ooCher/RD-MPNN/tree/main/RD_MPNN
9. ReAlE: https://github.com/baharefatemi/ReAlE