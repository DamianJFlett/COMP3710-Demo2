import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# -------------------------
# Simple ResNet-18 for CIFAR10 (standard tiny adjustments)
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

def resnet18_cifar():
    return ResNetCIFAR(BasicBlock, [2,2,2,2])

# -------------------------
# MixUp helper
# -------------------------
def mixup_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -------------------------
# Training / Evaluation
# -------------------------
def get_dataloaders(batch_size=256, num_workers=4):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),  # torchvision >=0.13
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean,std)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    testloader  = DataLoader(testset,  batch_size=512, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct/total

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    running_loss = 0.0
    t0 = time.time()
    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.mixup_alpha > 0:
            images, targets_a, targets_b, lam = mixup_data(images, targets, args.mixup_alpha)
        else:
            targets_a, targets_b, lam = targets, None, None

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(images)
            if targets_b is not None:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # optional: gradient clipping
        if args.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    epoch_time = time.time() - t0
    avg_loss = running_loss / len(loader)
    return avg_loss, epoch_time

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=None)  # if None, will be set by linear scaling
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--mixup-alpha', type=float, default=0.8)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--max-norm', type=float, default=0.0)
    parser.add_argument('--onecycle', action='store_true', default=True)
    parser.add_argument('--save', type=str, default='checkpoint.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    trainloader, testloader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    model = resnet18_cifar().to(device)

    # optimizer and lr scaling
    if args.lr is None:
        # linear scaling rule: lr_base * (batch/256)
        base_lr = 0.1
        args.lr = base_lr * (args.batch_size / 256)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.onecycle:
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(trainloader))
    else:
        scheduler = None

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        loss, epoch_time = train_one_epoch(model, trainloader, optimizer, scaler, device, epoch, args)
        if scheduler is not None:
            scheduler.step()
        val_acc = evaluate(model, testloader, device)
        print(f"Epoch {epoch}/{args.epochs}  loss={loss:.4f}  val_acc={val_acc*100:.2f}%  epoch_time={epoch_time:.1f}s")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, args.save)
        # quick early stop if target hit (useful for DAWNBench-style runs)
        if best_acc >= 0.93:
            print(f"Target 93% reached at epoch {epoch}")
            break

if __name__ == '__main__':
    main()