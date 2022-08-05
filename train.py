import os
import random
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import torchmetrics

from early_stopping import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./dataset')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument("--h-dim", type=int, default=8)
parser.add_argument("--n-heads", type=int, default=8)
parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--n-epoch", type=int, default=100000)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--device-id", type=str, default="1")
parser.add_argument("--dropout", type=float, default=0.6)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--train', action="store_true",default=False)
parser.add_argument('--test', action="store_true",default=False)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# load dataset
dataset = Planetoid(root=args.root, name=args.dataset, split="public",transform=T.NormalizeFeatures())
data = dataset[0].to(device)
# 20 nodes per class, 500 val, 1000 test
num_train = data.train_mask.sum().item()
num_val = data.val_mask.sum().item()
num_test = data.test_mask.sum().item()
print("Dataset: {:d} train, {:d} val, {:d} test".format(num_train,num_val,num_test))


class GAT(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=8, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.gat1 = GATConv(features, hidden, heads=heads, dropout=self.dropout, negative_slope=alpha)
        self.gat2 = GATConv(hidden*heads, classes, heads=1, dropout=self.dropout, negative_slope=alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)


# define model
model = GAT(dataset.num_node_features, args.h_dim,
            dataset.num_classes, heads=args.n_heads, dropout=args.dropout, alpha=args.alpha)
model = model.to(device)
print("Model: GAT")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

loss_fn = torch.nn.NLLLoss()
metric_fn = torchmetrics.Accuracy().to(device)

early_stopping = EarlyStopping(patience=args.patience)

def train():
    model.train()

    optimizer.zero_grad()
    y_hat = model(data)[data.train_mask]
    y = data.y[data.train_mask]
    loss = loss_fn(y_hat, y)
    loss.backward()
    optimizer.step()
    # cal acc
    acc = metric_fn(y_hat,y)

    return loss.item(),acc.item()

@torch.no_grad()
def val():
    model.eval()

    y_hat = model(data)[data.val_mask]
    y = data.y[data.val_mask]
    loss = loss_fn(y_hat, y)

    # cal acc
    acc = metric_fn(y_hat,y)

    return loss.item(), acc.item()

@torch.no_grad()
def test():
    model.eval()

    y_hat = model(data)[data.test_mask]
    y = data.y[data.test_mask]
    loss = loss_fn(y_hat, y)
    # cal acc
    acc = metric_fn(y_hat,y)

    print("Test set results:",
          "loss= {:.4f}".format(loss.item()),
          "accuracy= {:.4f}".format(acc.item()))


if args.train:
    writer = SummaryWriter()
    print("Start training")
    t_total = time.time()
    for epoch in range(1, args.n_epoch + 1):
        train_loss, train_acc = train()
        
        val_loss, val_acc = val()
        print(
            f'Epoch {epoch:02d}, Train/Loss: {train_loss:.4f}, Train/Acc: {train_acc:.4f}, Val/Loss: {val_loss:.4f}, Val/Acc: {val_acc:.4f}')
        writer.add_scalars('Loss',{"Train":train_loss,"Val":val_loss},epoch)
        writer.add_scalars('Acc',{"Train":train_acc,"Val":val_acc},epoch)

        early_stopping(val_loss, val_acc, model)
        if early_stopping.early_stop:
            print("early_stopping")
            break

    writer.close()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if args.test:
    model.load_state_dict(torch.load("best_model.pth"))
    test()
