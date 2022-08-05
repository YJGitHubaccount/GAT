import os
import random
import argparse
import time

import numpy as np
from pandas import concat
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import GATConv
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

import torchmetrics

from early_stopping import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./dataset')
parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--n-epoch", type=int, default=100000)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--device-id", type=str, default="1")
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--train', action="store_true", default=False)
parser.add_argument('--test', action="store_true", default=False)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: " + device)

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# load dataset
train_dataset = PPI(root='./dataset', split="train",
                    transform=T.NormalizeFeatures())
val_dataset = PPI(root='./dataset', split="val",
                  transform=T.NormalizeFeatures())
test_dataset = PPI(root='./dataset', split="test",
                   transform=T.NormalizeFeatures())
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
# 20 nodes per class, 500 val, 1000 test
num_train = 20
num_val = 2
num_test = 2
print("Dataset: {:d} train, {:d} val, {:d} test".format(
    num_train, num_val, num_test))


class GAT(torch.nn.Module):
    def __init__(self, features=50):
        super(GAT, self).__init__()
        self.gat1 = GATConv(features, 256, heads=4)
        self.gat2 = GATConv(256*4, 256, heads=4)
        self.gat3 = GATConv(256*4, 121, heads=6, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        # skip connection
        x = self.gat2(x, edge_index) + x
        x = F.elu(x)

        x = self.gat3(x, edge_index)
        return torch.sigmoid(x)


# define model
model = GAT().to(device)
print("Model: GAT")


optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

loss_fn = torch.nn.CrossEntropyLoss()
metric_fn = torchmetrics.F1Score(num_classes=121, average='micro').to(device)

early_stopping = EarlyStopping(patience=args.patience)


def train():
    model.train()

    running_loss = 0.
    running_micro_f1 = 0.
    step = 0.
    for data in train_loader:
        step += 1.

        data = data.to(device)

        optimizer.zero_grad()
        y_hat = model(data)
        y = data.y
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_micro_f1 += metric_fn(y_hat, y.type(torch.int)).item()

    avg_loss = running_loss/step
    avg_micro_f1 = running_micro_f1/step

    return avg_loss, avg_micro_f1


@torch.no_grad()
def val():
    model.eval()

    running_loss = 0.
    running_micro_f1 = 0.
    step = 0.
    for data in val_loader:
        step += 1.

        data = data.to(device)

        y_hat = model(data)
        y = data.y
        loss = loss_fn(y_hat, y)

        running_loss += loss.item()
        running_micro_f1 += metric_fn(y_hat, y.type(torch.int)).item()

    avg_loss = running_loss/step
    avg_micro_f1 = running_micro_f1/step

    return avg_loss, avg_micro_f1


@torch.no_grad()
def test():
    model.eval()

    running_loss = 0.
    running_micro_f1 = 0.
    step = 0.
    for data in test_loader:
        step += 1.

        data = data.to(device)

        y_hat = model(data)
        y = data.y
        loss = loss_fn(y_hat, y)

        running_loss += loss.item()
        running_micro_f1 += metric_fn(y_hat, y.type(torch.int)).item()

    avg_loss = running_loss/step
    avg_micro_f1 = running_micro_f1/step

    print("Test set results:",
          "avg_loss= {:.4f}".format(avg_loss),
          "avg_micro_f1= {:.4f}".format(avg_micro_f1))


if args.train:
    writer = SummaryWriter()
    print("Start training")
    t_total = time.time()
    val_ap_list = []
    ave_val_ap = 0
    end = 0
    for epoch in range(1, args.n_epoch + 1):
        avg_train_loss, avg_train_micro_f1 = train()

        avg_val_loss, avg_val_micro_f1 = val()
        print(
            f'Epoch {epoch:02d}, Train/Avg_Loss: {avg_train_loss:.4f}, Train/Avg_Micro_F1: {avg_train_micro_f1:.4f}, Val/Avg_Loss: {avg_val_loss:.4f}, Val/Avg_Micro_F1: {avg_val_micro_f1:.4f}')
        writer.add_scalars(
            'Avg_Loss', {"Train": avg_train_loss, "Val": avg_val_loss}, epoch)
        writer.add_scalars(
            'Avg_Micro_F1', {"Train": avg_train_micro_f1, "Val": avg_val_micro_f1}, epoch)

        early_stopping(avg_val_loss, avg_val_micro_f1, model)
        if early_stopping.early_stop:
            print("early_stopping")
            break

    writer.close()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if args.test:
    model.load_state_dict(torch.load("best_model.pth"))
    test()
