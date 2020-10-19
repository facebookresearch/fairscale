import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import fairscale

model = nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5)
        )
target = torch.randint(0,2,size=(20,1)).squeeze()
data = torch.randn(20, 10)
loss_fn = F.nll_loss

model = fairscale.nn.Pipe(model, balance=[2, 1])

#define optimizer and loss function 
optimizer = optim.SGD(model.parameters(), lr=0.001)


# zero the parameter gradients
optimizer.zero_grad()

device = model.devices[0]

## outputs and target need to be on the same device
# forward step
outputs = model(data.to(device))
# compute loss
loss = loss_fn(outputs.to(device), target.to(device))

# backward + optimize
loss.backward()
optimizer.step()

print('Finished Training Step')

del model
