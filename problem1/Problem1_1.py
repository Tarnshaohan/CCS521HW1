# a textbook implementation of a targeted Fast Gradient Sign Method (FGSM) adversarial attack in PyTorch
# under an $L_\infty$ perturbation budget
import torch
import torch.nn as nn


# 1 Model Setup:
# fix seed so that random initialization always performs the same
# A fixed random seed (13) ensures the randomly initialized weights and input yield deterministic behavior.

torch.manual_seed(13)

# create the model N as described in the question
# It builds a small 3-layer MLP (N):
# input dimension 10, hidden dimension 10, output dimension 3, using ReLU activations and no bias terms.

N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# 2 Clean Prediction:

# random input: It draws a random input vector x of shape (1, 10) and tracks its gradient with x.requires_grad_()

x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data

x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 0 # target class

epsReal = 0.5  #depending on your data, this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point errors

# The network N classfies x as belonging to class 2
original_class = N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
# Passing x through N outputs raw logits. argmax(dim=1) extracts the index of the highest logit.

print("Original Class: ", original_class)
assert(original_class == 2) # Under seed 13, the model naturally classifies x as class 2.

# 3 Computing the Adversarial Gradient:
# The goal is to force the model to misclassify x into target class t = 0.

# compute gradient
# It sets up nn.CrossEntropyLoss() between the model's logits N(x) and target label 0.
# Note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
# Calling loss.backward() calculates $\nabla_x L$, which PyTorch stores directly inside x.grad.
# This vector points in the direction of greatest increase in loss relative to class 0.

L = nn.CrossEntropyLoss()
loss = L(N(x), torch.tensor([t], dtype=torch.long)) # TO LEARN: make sure you understand this line
loss.backward()

# your code here
# adv_x should be computed from x according to the fgsm-style perturbation such that the new class of xBar is the target class t above
# hint: you can compute the gradient of the loss w.r.t to x as x.grad
# adv_x = TODO is not right,
# should be: adv_x = (x - eps * x.grad.sign()).detach()

# 4 Perturbation (Targeted FGSM):
# To make class 0 more likely, the loss must decrease.
# It moves in the direction of the negative gradient sign
adv_x = (x - eps * x.grad.sign()).detach()

# 5 Verification:

# checks if the new prediction matches target 0

new_class = N(adv_x).argmax(dim=1).item() #  checks if the new prediction matches target 0
print("New Class: ", new_class)
assert(new_class == t)
# it is not enough that adv_x is classified as t.
# We also need to make sure it is 'close' to the original x.
print(torch.norm((x-adv_x),  p=float('inf')).data)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal) # verifies that no single dimension was perturbed
# by more than the allowed 0.5 threshold.
