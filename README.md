# GAT 
PyG impletement of GAT


## Prerequisites
- python 3.8.5
- pytorch 1.12.0+cu116
- torch_geometric 1.12.0+cu116
- torchmetrics


## Usage
For transductive dataset:

`python train.py --train --test --dataset=Cora`
`python train.py --train --test --dataset=CiteSeer`
`python train.py --train --test --dataset=PubMed --weight_decay=0.001`


For inductive dataset(it take a few time):

`python train_inductive.py --train --test`


## Model Architecture and Parameters 
model architecture and parameters are based on the original paper 

## Experiment Result
| Transductive | Accuracy    |
| -----------  | ----------- |
| Cora         | 0.8320      |
| CiteSeer     | 0.7270      |
| PubMed       | 0.7790      |

&nbsp; 

| Inductive    | Avg_Micro_F1|
| -----------  | ----------- |
| PPI          | 0.9079      |