from helpers import get_data, get_loss_fun, get_model
import torch
import torch.optim as optim

import fairscale

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANK = 0  # example

model = get_model()
data, target = get_data()[0]
loss_fn = get_loss_fun()

model = fairscale.nn.Pipe(model, balance=[2, 1])

# define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.001)


# zero the parameter gradients
optimizer.zero_grad()

device = torch.device("cuda", RANK) if DEVICE == "cuda" else torch.device("cpu")

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
