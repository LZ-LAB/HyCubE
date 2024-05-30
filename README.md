# The HyCubE Model
This paper has been submitted to the IEEE TKDE journal.



## Requirements
The version of Python and major packages needed to run the code:
   
    -- python 3.9.16
    -- torch 1.12.0
    -- numpy 1.26.0
    -- tqdm 4.65.0



## How to Run

### HyCubE

#### 1. Mixed Arity Knowledge Hypergraph
```
## JF17K dataset
python main-JF17K.py --dataset JF17K --batch_size 384 --lr 0.00095 --dr 0.993 --input_drop 0.3 --dropout 0.9 --dropout_3d 0.5 --padding 1

## WikiPeople dataset
python main-WikiPeople.py --dataset WikiPeople --batch_size 384 --lr 0.00035 --dr 0.993 --input_drop 0.3 --dropout 0.5 --dropout_3d 0.6 --padding 5

## FB-AUTO dataset
python main-FB-AUTO.py --dataset FB-AUTO --batch_size 64 --lr 0.00099 --dr 0.968 --input_drop 0.5 --dropout 0.5 --dropout_3d 0.5 --padding 3
```

#### 2. Fixed Arity Knowledge Hypergraph
```
## JF17K-3 dataset
python main-fixed3.py --dataset JF17K-3 --batch_size 384 --lr 0.00049 --dr 0.995 --input_drop 0.9 --dropout 0 --dropout_3d 0.9 --padding 5

## JF17K-4 dataset
python main-fixed4.py --dataset JF17K-4 --batch_size 128 --lr 0.00062 --dr 0.978 --input_drop 0.9 --dropout 0.9 --dropout_3d 0 --padding 1

## WikiPeople-3 dataset
python main-fixed3.py --dataset WikiPeople-3 --batch_size 256 --lr 0.00078 --dr 0.967 --input_drop 0.2 --dropout 0.2 --dropout_3d 0.9 --padding 5

## WikiPeople-4 dataset
python main-fixed4.py --dataset WikiPeople-4 --batch_size 64 --lr 0.00094 --dr 0.909 --input_drop 0.2 --dropout 0.7 --dropout_3d 0.7 --padding 5
```



### HyCubE+

#### 1. Mixed Arity Knowledge Hypergraph
```
## JF17K dataset
python main-JF17K.py --dataset JF17K --batch_size 128 --lr 0.00077 --dr 0.985 --input_drop 0.7 --dropout 0.7 --dropout_3d 0.2 --padding 5

## WikiPeople dataset
python main-WikiPeople.py --dataset WikiPeople --batch_size 384 --lr 0.00050 --dr 0.989 --input_drop 0.3 --dropout 0 --dropout_3d 0.8 --padding 3

## FB-AUTO dataset
python main-FB-AUTO.py --dataset FB-AUTO --batch_size 384 --lr 0.00075 --dr 0.977 --input_drop 0.6 --dropout 0.2 --dropout_3d 0.7 --padding 3
```

#### 2. Fixed Arity Knowledge Hypergraph
```
## JF17K-3 dataset
python main-fixed3.py --dataset JF17K-3 --batch_size 64 --lr 0.00088 --dr 0.982 --input_drop 0.3 --dropout 0.9 --dropout_3d 0.3 --padding 5

## JF17K-4 dataset
python main-fixed4.py --dataset JF17K-4 --batch_size 128 --lr 0.00049 --dr 0.978 --input_drop 0.6 --dropout 0.1 --dropout_3d 0.6 --padding 3

## WikiPeople-3 dataset
python main-fixed3.py --dataset WikiPeople-3 --batch_size 64 --lr 0.00094 --dr 0.983 --input_drop 0.9 --dropout 0.5 --dropout_3d 0.3 --padding 1

## WikiPeople-4 dataset
python main-fixed4.py --dataset WikiPeople-4 --batch_size 64 --lr 0.00086 --dr 0.975 --input_drop 0.9 --dropout 0.6 --dropout_3d 0 --padding 2
```





## Acknowledgments
We are very grateful for all open-source baseline models:

1. HypE/HSimplE: https://github.com/ElementAI/HypE
2. HyperMLN: https://github.com/zirui-chen/HyperMLN
3. RAM: https://github.com/liuyuaa/RAM
4. GETD: https://github.com/liuyuaa/GETD
5. tNaLP+: https://github.com/gsp2014/NaLP
6. PosKHG: https://github.com/zirui-chen/PosKHG
7. HyConvE: https://github.com/CarllllWang/HyConvE/tree/master
8. RD-MPNN: https://github.com/ooCher/RD-MPNN/tree/main/RD_MPNN
9. ReAlE: https://github.com/baharefatemi/ReAlE
