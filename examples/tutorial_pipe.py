import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import fairscale

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 5))
target = torch.randint(0, 2, size=(20, 1)).squeeze()
data = torch.randn(20, 10)
loss_fn = F.nll_loss

model = fairscale.nn.Pipe(model, balance=[2, 1])

# define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.001)


# zero the parameter gradients
optimizer.zero_grad()

device = torch.device("cuda", rank) if DEVICE == "cuda" else torch.device("cpu")

# outputs and target need to be on the same device
# forward step
outputs = model(data.to(device).requires_grad_())
# compute loss
loss = loss_fn(outputs.to(device), target.to(device))

# backward + optimize
loss.backward()
optimizer.step()

print("Finished Training Step")

del model
