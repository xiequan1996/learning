import numpy as np
import torch
import torch.nn as nn
import time

from tensorboardX import SummaryWriter

train_path = "train/0.3M.txt"
# val_path = "data/val"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_normal_2d(origin_data):
    d_min = origin_data.min(0)
    for idx, j in enumerate(d_min):
        if j < 0:
            origin_data[idx, :] += np.abs(d_min[idx])
            d_min = origin_data.min(0)
    d_max = origin_data.max(0)
    dst = d_max - d_min
    norm_data = np.divide(np.subtract(origin_data, d_min), dst)
    return norm_data


class InverNet(nn.Module):
    def __init__(self):
        super(InverNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.model(x)
        return x


# inver_net = InverNet()
inver_net = torch.load("inver_dict_1.pth")
inver_net.to(device)
loss_fn = nn.SmoothL1Loss(reduction='sum')
loss_fn.to(device)
opti = torch.optim.SGD(inver_net.parameters(), lr=0.0001)
# opti = torch.optim.Adam(inver_net.parameters(), lr=0.01)
epoch = 100
total_train_step = 0

# 添加tensorboard
writer = SummaryWriter("../logs")

start_time = time.time()
f = open(train_path, 'r')
lines_list = f.readlines()
# lines_len = len(lines_list)
x_train = np.zeros((31, 4))
y_train = np.zeros((31, 1))
for i in range(31):
    temp_list = lines_list[i + 336].strip().split('\t')
    x_train[i, :] = temp_list[10:14]
    y_train[i, :] = temp_list[0:1]
# x_train = data_normal_2d(x_train)
x_train = torch.tensor(x_train, dtype=torch.float32, device=device, requires_grad=True)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device, requires_grad=True)

for i in range(epoch):
    outputs = inver_net(x_train)
    return_loss = loss_fn(outputs, y_train)

    opti.zero_grad()
    return_loss.backward(retain_graph=True)
    opti.step()

    total_train_step += 1
    # if total_train_step % 10 == 0:
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
