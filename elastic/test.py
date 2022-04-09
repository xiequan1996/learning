# path = "./data/train1/e11-42-e22-80/g12-12-ph.txt"
# f = open(path, 'r')
# lines = f.readlines()
# print(lines[0].strip().split('\t'))
# print(lines[1].strip().split('\t'))
# list = lines[1].strip().split('\t')
# print(list[0].split(' '))
# print(list[0].split())
# print(list[0].split())

# path1 = "0.3M.txt"
# f1 = open(path1, 'r')
# lines1 = f1.readlines()
# list1=lines1[0].strip().split('\t')
# print(list1)
# print(len(list1))
# print(type(list1[0]))
import numpy as np
import torch
from torch import nn

# from inversion import InverNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class InverNet(nn.Module):
    def __init__(self):
        super(InverNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 6),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.model(x)
        return x


net = InverNet()
net.load_state_dict(torch.load("inver_dict.pth"))
net.to(device)
loss_fn = nn.MSELoss()
loss_fn.to(device)

f = open("0.3M_val.txt", 'r')
lines_list = f.readlines()
lines_len = len(lines_list)
x_val = np.zeros((lines_len, 5))
y_val = np.zeros((lines_len, 6))
for i in range(lines_len):
    temp_list = lines_list[i].strip().split('\t')
    x_val[i, :] = temp_list[9:14]
    # x_train[i, 6:9] = temp_list[6:9]
    # x_train[i, 6] = temp_list[9]
    # for j in range(3):
    #     y_val[i, j] = float(temp_list[j]) / 90
    # for k in range(3):
    #     y_val[i, k + 3] = float(temp_list[k + 3]) / 50
    y_val[i, :] = temp_list[0:6]

x_val = torch.tensor(x_val, dtype=torch.float32, device=device)
y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

outputs = net(x_val)
print(outputs)
# for j in range(lines_len):
#     outputs[j, 0:3] = outputs[j, 0:3] * 90
#     outputs[j, 3:6] = outputs[j, 3:6] * 50
loss = loss_fn(outputs, y_val)
print(loss)
