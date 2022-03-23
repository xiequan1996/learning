import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)


class Module1(nn.Module):
    def __init__(self):
        super(Module1, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=5),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss1 = nn.CrossEntropyLoss()
module1 = Module1()
optim = torch.optim.SGD(module1.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = module1(imgs)
        return_loss = loss1(outputs, targets)
        optim.zero_grad()
        return_loss.backward()
        optim.step()
        running_loss += return_loss
    print(running_loss)
