# a textbook implementation of a targeted Fast Gradient Sign Method (FGSM) adversarial attack in PyTorch
# Problem 1. 2
import torch
import torch.nn as nn


# 1 Model and Input Setup:

torch.manual_seed(13)

# create the model N as described in the question
# It builds a small 3-layer MLP (N):
# input dimension 10, hidden dimension 10, output dimension 3, using ReLU activations and no bias terms.

N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))


x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data

x.requires_grad_() # this is required so we can compute the gradient w.r.t x

#2 Target class set to 1

t = 1 # target class

epsReal = 1.1374  #depending on your data, this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point errors



original_class = N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
# Passing x through N outputs raw logits. argmax(dim=1) extracts the index of the highest logit.

print("Original Class: ", original_class)
assert(original_class == 2) # Under seed 13, the model naturally classifies x as class 2.

# 3 Compute Loss and gradient w.r.t. tangent class 1
L = nn.CrossEntropyLoss()
loss = L(N(x), torch.tensor([t], dtype=torch.long)) # TO LEARN: make sure you understand this line
loss.backward()

# 4 Standard targeted FGSM step 

adv_x = (x - eps * x.grad.sign()).detach()

# 5 Verification:

# checks if the new prediction 

new_class = N(adv_x).argmax(dim=1).item() #  checks if the new prediction matches target 0

print("New Class: ", new_class)

print(torch.norm((x-adv_x),  p=float('inf')).data)

