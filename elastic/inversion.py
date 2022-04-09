import numpy as np
import torch
import torch.nn as nn
import time

from tensorboardX import SummaryWriter

train_path = "total.txt"
# val_path = "data/val"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_normal_2d(origin_data):
    d_min = torch.min(origin_data, dim=0)[0]
    for idx, j in enumerate(d_min):
        if j < 0:
            origin_data[idx, :] += torch.abs(d_min[idx])
            d_min = torch.min(origin_data, dim=0)[0]
    d_max = torch.max(origin_data, dim=0)[0]
    dst = d_max - d_min
    if d_min.shape[0] == origin_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(origin_data, d_min).true_divide(dst)
    return norm_data


class InverNet(nn.Module):
    def __init__(self):
        super(InverNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 6),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.model(x)
        return x


inver_net = InverNet()
# inver_net = torch.load("inver_dict.pth")
inver_net.to(device)
loss_fn = nn.MSELoss()
loss_fn.to(device)
opti = torch.optim.SGD(inver_net.parameters(), lr=0.001)
# opti = torch.optim.Adam(inver_net.parameters(), lr=0.01)
epoch = 1000
total_train_step = 0

# 添加tensorboard
writer = SummaryWriter("../logs")

start_time = time.time()
f = open(train_path, 'r')
lines_list = f.readlines()
lines_len = len(lines_list)
x_train = np.zeros((lines_len, 5))
y_train = np.zeros((lines_len, 6))
for i in range(lines_len):
    temp_list = lines_list[i].strip().split('\t')
    x_train[i, :] = temp_list[9:14]
    y_train[i, :] = temp_list[0:6]

x_train = torch.tensor(x_train, dtype=torch.float32, device=device, requires_grad=True)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device, requires_grad=True)
x_train_nor = data_normal_2d(x_train)
for i in range(epoch):
    outputs = inver_net(x_train_nor)
    return_loss = loss_fn(outputs, y_train)

    opti.zero_grad()
    return_loss.backward(retain_graph=True)
    opti.step()

    total_train_step += 1
    if total_train_step % 10 == 0:
        print("训练次数：{},Loss:{}".format(total_train_step, return_loss.item()))
    # writer.add_scalar("loss_{}".format(a % num), return_loss, i)
    # a += 1

# with torch.no_grad():
#     test_mat, test_labels = folder2matrix(val_path, init_label)
#     test_mat = torch.tensor(test_mat, dtype=torch.float32, device=device)
#     test_labels = torch.tensor(test_labels, dtype=torch.float32, device=device)
#     test_mat = test_mat.reshape(-1, 1, 300, 16)
#     test_outputs = mlnet(test_mat)
#     test_loss = loss_fn(test_outputs, test_labels)
#     # writer.add_scalar("test_loss", test_loss, i)
#     print("测试集第{}次上的Loss:{}".format(i + 1, test_loss))

end_time = time.time()
print(end_time - start_time)
torch.save(inver_net, "inver_dict_1.pth")
print("保存模型")
writer.close()
