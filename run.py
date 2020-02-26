import torch
import argparse
import torch.nn as nn
from nn import ResNet18
from tools import AverageMeter
from progressbar import ProgressBar
from tools import seed_everything
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from trainingmonitor import TrainingMonitor
from optimizer import Lookahead

epochs = 30
batch_size = 128
seed = 42

seed_everything(seed)
model = ResNet18()
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cuda:0")
model.to(device)

parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument("--model", type=str, default='ResNet18')
parser.add_argument("--task", type=str, default='image')
parser.add_argument("--optimizer", default='lookahead',type=str,choices=['lookahead','adam'])
args = parser.parse_args()

if args.optimizer == 'lookahead':
    arch = 'ResNet18_Lookahead_adam'
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = Lookahead(optimizer=optimizer,k=5,alpha=0.5)
else:
    arch = 'ResNet18_Adam'
    optimizer = optim.Adam(model.parameters(), lr=0.001)

train_monitor = TrainingMonitor(file_dir='./',arch = arch)

def train(train_loader):
    pbar = ProgressBar(n_total=len(train_loader),desc='Training')
    train_loss = AverageMeter()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        pbar(step = batch_idx,info = {'loss':loss.item()})
        train_loss.update(loss.item(),n =1)
    return {'loss':train_loss.avg}

def test(test_loader):
    pbar = ProgressBar(n_total=len(test_loader),desc='Testing')
    valid_loss = AverageMeter()
    valid_acc = AverageMeter()
    model.eval()
    count = 0
    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            valid_loss.update(loss,n = data.size(0))
            valid_acc.update(correct, n=1)
            count += data.size(0)
            pbar(step=batch_idx)
    return {'valid_loss':valid_loss.avg,
            'valid_acc':valid_acc.sum /count}

data = {
    'train': datasets.CIFAR10(
        root='./data', download=True,
        transform=transforms.Compose([
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))]
        )
    ),
    'valid': datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))]
        )
    )
}

loaders = {
    'train': DataLoader(data['train'], batch_size=128, shuffle=True,
                        num_workers=10, pin_memory=True,
                        drop_last=True),
    'valid': DataLoader(data['valid'], batch_size=128,
                        num_workers=10, pin_memory=True,
                        drop_last=False)
}

for epoch in range(1, epochs + 1):
    train_log = train(loaders['train'])
    if args.optimizer == 'lookahead':
        optimizer._backup_and_load_cache()
        valid_log = test(loaders['valid'])
        optimizer._clear_and_load_backup()
    else:
        valid_log = test(loaders['valid'])
    logs = dict(train_log, **valid_log)
    show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
    print(show_info)
    train_monitor.epoch_step(logs)

