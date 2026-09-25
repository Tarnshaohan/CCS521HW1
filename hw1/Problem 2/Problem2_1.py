
# !pip install tensorboardX
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import datasets, transforms

# from tensorboardX import SummaryWriter

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)


## Dataloaders
# train_dataset = datasets.CIFAR10('cifar10_data/', train=True, download=True, transform=transforms.Compose(
#     [transforms.ToTensor()]
# ))
test_dataset = datasets.CIFAR10('cifar10_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


import os

# -------------------------------------------------------------------------
# Environment & DataLoader Setup
# -------------------------------------------------------------------------
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)
if use_cuda:
    torch.cuda.manual_seed_all(42)

test_dataset = datasets.CIFAR10(
    root='cifar10_data/',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=use_cuda
)

# -------------------------------------------------------------------------
# Model Definitions (PreAct ResNet-18)
# -------------------------------------------------------------------------
def tp_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return .5 * (x + delta) * (1 - ind1) * (1 - ind2) + x * ind2

def tp_smoothed_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return (x + delta) ** 2 / (4 * delta) * (1 - ind1) * (1 - ind2) + x * ind2

class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std

class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1, activation='relu'):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.activation = activation
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def act_function(self, preact):
        if self.activation == 'relu':
            return F.relu(preact)
        elif self.activation.startswith('3prelu'):
            return tp_relu(preact, delta=float(self.activation.split('relu')[1]))
        elif self.activation.startswith('3psmooth'):
            return tp_smoothed_relu(preact, delta=float(self.activation.split('smooth')[1]))
        else:
            assert self.activation.startswith('softplus')
            beta = int(self.activation.split('softplus')[1])
            return F.softplus(preact, beta=beta)

    def forward(self, x):
        out = self.act_function(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.act_function(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, cuda=True, half_prec=False,
                 activation='relu', fts_before_bn=False, normal='none'):
        super(PreActResNet, self).__init__()
        self.bn = True
        self.learnable_bn = True
        self.in_planes = 64
        self.activation = activation
        self.fts_before_bn = fts_before_bn

        if normal == 'cifar10':
            self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1)
            self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)

        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()
        if half_prec:
            self.mu = self.mu.half()
            self.std = self.std.half()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn_out = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, n_cls)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, s, self.activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = self.normalize(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if return_features and self.fts_before_bn:
            return out.view(out.size(0), -1)
        out = F.relu(self.bn_out(out))
        if return_features:
            return out.view(out.size(0), -1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def PreActResNet18(n_cls=10, cuda=True, half_prec=False, activation='relu', fts_before_bn=False, normal='none'):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, cuda=cuda,
                        half_prec=half_prec, activation=activation,
                        fts_before_bn=fts_before_bn, normal=normal)

# -------------------------------------------------------------------------
# Attack Implementations
# -------------------------------------------------------------------------
def pgd_linf_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    adv_x = x.clone().detach()

    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)

        ### >>> TODO RESOLVED: Calculate loss and compute adv_x for Linf <<< ###
        loss = ce_loss(output, labels)
        loss.backward()

        grad = adv_x.grad.data
        adv_x = adv_x.detach() + eps_step * torch.sign(grad)

        eta = torch.clamp(adv_x - x, min=-eps, max=eps)
        adv_x = torch.clamp(x + eta, min=0.0, max=1.0)
        ### >>> END TODO RESOLVED <<< ###

    return adv_x.detach()


def pgd_l2_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    delta_avoid_zero = 1e-10

    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        batch_size = x.size()[0]

        ### >>> TODO RESOLVED: Calculate loss and project delta to L2 ball <<< ###
        loss = ce_loss(output, labels)
        loss.backward()

        grad = adv_x.grad.data
        grad_norm = grad.view(batch_size, -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        normalized_grad = grad / (grad_norm + delta_avoid_zero)
        adv_x = adv_x.detach() + eps_step * normalized_grad

        diff = adv_x - x
        diff_norm = diff.view(batch_size, -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        factor = torch.clamp(diff_norm / eps, min=1.0)
        adv_x = x + (diff / factor)

        adv_x = torch.clamp(adv_x, min=0.0, max=1.0)
        ### >>> END TODO RESOLVED <<< ###

    return adv_x.detach()

# -------------------------------------------------------------------------
# Evaluation Functions
# -------------------------------------------------------------------------
def test_model_on_single_attack(model, attack='pgd_linf', eps=0.1, k=20):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        ### >>> TODO RESOLVED: Get adversarial example according to attack type <<< ###
        if attack == 'clean':
            x_adv = x_batch
        elif attack == 'pgd_linf':
            x_adv = pgd_linf_untargeted(model, x_batch, y_batch, k=k, eps=eps, eps_step=eps / 4.0)
        elif attack == 'pgd_l2':
            x_adv = pgd_l2_untargeted(model, x_batch, y_batch, k=k, eps=eps, eps_step=eps / 4.0)
        else:
            raise ValueError(f"Unknown attack: {attack}")
        ### >>> END TODO RESOLVED <<< ###

        ### >>> TODO RESOLVED: Update tot_test and tot_acc <<< ###
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
            tot_acc += (preds == y_batch).sum().item()
            tot_test += y_batch.size(0)
        ### >>> END TODO RESOLVED <<< ###

    print('Robust accuracy %.5lf' % (tot_acc / tot_test), f'on {attack} attack with eps = {eps}')
    return tot_acc / tot_test


def test_model_on_multi_attacks(model, eps_linf=8./255., eps_l2=0.75, k=20):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        ### >>> TODO RESOLVED: Generate both Linf and L2 adversarial samples <<< ###
        x_adv_linf = pgd_linf_untargeted(model, x_batch, y_batch, k=k, eps=eps_linf, eps_step=eps_linf / 4.0)
        x_adv_l2   = pgd_l2_untargeted(model, x_batch, y_batch, k=k, eps=eps_l2, eps_step=eps_l2 / 4.0)
        ### >>> END TODO RESOLVED <<< ###

        ## calculate union accuracy: correct only if both attacks are correct
        with torch.no_grad():
            out = model(x_adv_linf)
            pred_linf = torch.max(out, dim=1)[1]
            out = model(x_adv_l2)
            pred_l2 = torch.max(out, dim=1)[1]

            ### >>> TODO RESOLVED: Multi-norm union robustness accuracy calculation <<< ###
            tot_acc += ((pred_linf == y_batch) & (pred_l2 == y_batch)).sum().item()
            tot_test += y_batch.size(0)
            ### >>> END TODO RESOLVED <<< ###

    print('Robust accuracy %.5lf' % (tot_acc / tot_test), f'on multi attacks')
    return tot_acc / tot_test

# -------------------------------------------------------------------------
# Execution / Evaluation Pipeline
# -------------------------------------------------------------------------
if __name__ == '__main__':
    model = PreActResNet18(10, cuda=use_cuda, activation='softplus1').to(device)

    # 1. Evaluate on Linf attack with different models (eps = 8/255)
    print("\n--- Linf Attacks (eps = 8/255) ---")
    if os.path.exists('models/pretr_Linf.pth'):
        model.load_state_dict(torch.load('models/pretr_Linf.pth', map_location=device))
        test_model_on_single_attack(model, attack='pgd_linf', eps=8./255.)

    if os.path.exists('models/pretr_L2.pth'):
        model.load_state_dict(torch.load('models/pretr_L2.pth', map_location=device))
        test_model_on_single_attack(model, attack='pgd_linf', eps=8./255.)

    if os.path.exists('models/pretr_RAMP.pth'):
        model.load_state_dict(torch.load('models/pretr_RAMP.pth', map_location=device))
        test_model_on_single_attack(model, attack='pgd_linf', eps=8./255.)

    # 2. Evaluate on L2 attack with different models (eps = 0.75)
    print("\n--- L2 Attacks (eps = 0.75) ---")
    if os.path.exists('models/pretr_Linf.pth'):
        model.load_state_dict(torch.load('models/pretr_Linf.pth', map_location=device))
        test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)

    if os.path.exists('models/pretr_L2.pth'):
        model.load_state_dict(torch.load('models/pretr_L2.pth', map_location=device))
        test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)

    if os.path.exists('models/pretr_RAMP.pth'):
        model.load_state_dict(torch.load('models/pretr_RAMP.pth', map_location=device))
        test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)

    # 3. Evaluate on multi-norm attacks (eps_linf = 8/255, eps_l2 = 0.75)
    print("\n--- Multi-Norm Attacks (Union Robustness) ---")
    if os.path.exists('models/pretr_Linf.pth'):
        model.load_state_dict(torch.load('models/pretr_Linf.pth', map_location=device))
        test_model_on_multi_attacks(model, eps_linf=8./255., eps_l2=0.75)

    if os.path.exists('models/pretr_L2.pth'):
        model.load_state_dict(torch.load('models/pretr_L2.pth', map_location=device))
        test_model_on_multi_attacks(model, eps_linf=8./255., eps_l2=0.75)

    if os.path.exists('models/pretr_RAMP.pth'):
        model.load_state_dict(torch.load('models/pretr_RAMP.pth', map_location=device))
        test_model_on_multi_attacks(model, eps_linf=8./255., eps_l2=0.75)













