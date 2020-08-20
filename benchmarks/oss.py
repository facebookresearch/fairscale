# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.models import resnet50
import torch.nn as nn
import torch
from typing import List, Any
from torchvision.transforms import ToTensor


def train(num_epochs: int = 10, batch_size: int = 64, device: torch.device = torch.device("cuda")):
    # Standard RN50
    model = resnet50(pretrained=False, progress=True)
    print("Benchmarking model: ", model)

    # Data setup, dummy data
    def collate(inputs: List[Any]):
        return {
            "inputs": torch.stack([i[0] for i in inputs]).to(device),
            "label": torch.stack([i[1] for i in inputs]).to(device),
        }

    dataloader = DataLoader(dataset=FakeData(transform=ToTensor(), size=200), batch_size=batch_size, collate_fn=collate)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)

    # Dummy training loop
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for batch in dataloader:

            def closure():
                model.zero_grad()
                outputs = model(batch["inputs"])
                loss = loss_fn(outputs, batch["label"])
                loss.backward()
                print(f"dummy loss {loss.item()}")
                return loss

            optimizer.step(closure)


if __name__ == "__main__":
    # TODO: move all this to pytorch DDP
    train()
