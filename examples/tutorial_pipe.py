import torch
import torch.optim as optim

import fairscale
from helpers import getModel, getData, getLossFun

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = getModel()
data, target = getData()
loss_fn = getLossFun()

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
