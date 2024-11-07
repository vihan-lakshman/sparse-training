from topkast_linear import TopKastLinear
import torch
import torch.nn as nn
from torch.utils import data
import tqdm

device = "cpu"
import torch
from torchvision import datasets, transforms

class TopKastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_in = TopKastLinear(
            784, 128, p_forward=0.9, p_backward=0.5, device=device)
        self.activation = nn.ReLU()
        self.hidden1 = TopKastLinear(
            128, 128, p_forward=0.9, p_backward=0.5, device=device)
        self.layer_out = TopKastLinear(
            128, 10, p_forward=0.9, p_backward=0.5, device=device)


    def forward(self, X, sparse=True):
        X = torch.flatten(X, start_dim=1)
        
        y = self.layer_in(X, sparse=sparse)
        y = self.hidden1(self.activation(y), sparse=sparse)
        return self.layer_out(self.activation(y), sparse=sparse)


net = TopKastNet().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

train_data = datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor())

test_data = datasets.MNIST(
        "./data", train=False, download=True, transform=transforms.ToTensor())

train_loader = data.DataLoader(train_data, shuffle=False, batch_size=250)
test_loader = data.DataLoader(test_data, shuffle=False, batch_size=250)

for e in range(10):
    net.train()
    for x, y in tqdm.tqdm(train_loader):
        opt.zero_grad()
        out = net(x)
        l = torch.nn.functional.cross_entropy(out, y)
        l.backward()
        opt.step()

    correct, total = 0, 0
    net.eval()

    for x, y in test_loader:
        out = net(x)
        preds = torch.argmax(out, dim=-1)
        correct += (preds == y).sum()
        total += len(y)

    print(f"epoch={e} p@1={(correct / total):.3f}")


