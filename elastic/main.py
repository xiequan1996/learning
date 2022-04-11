import numpy as np
import torch
import torch.nn as nn
import time

from tensorboardX import SummaryWriter

train_path = "0.3M.txt"
# val_path = "data/val"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLNet(nn.Module):
    def __init__(self):
        super(MLNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 4),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.model(x)
        return x


mlnet = MLNet()
mlnet.to(device)
loss_fn = nn.MSELoss()
loss_fn.to(device)
opti = torch.optim.SGD(mlnet.parameters(), lr=0.00001)
epoch = 100000
total_train_step = 0

# 添加tensorboard
writer = SummaryWriter("../logs")

start_time = time.time()
f = open(train_path, 'r')
lines_list = f.readlines()
lines_len = len(lines_list)
x_train = np.zeros((lines_len, 6))
y_train = np.zeros((lines_len, 4))
for i in range(lines_len):
    temp_list = lines_list[i].strip().split('\t')
    for j in range(3):
        x_train[i, j] = float(temp_list[j]) / 90
    for k in range(3):
        x_train[i, k + 3] = float(temp_list[k + 3]) / 50
    # x_train[i, 6:9] = temp_list[6:9]
    # x_train[i, 6] = temp_list[9]
    y_train[i, :] = temp_list[10:14]

x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
for i in range(epoch):
    outputs = mlnet(x_train)
    return_loss = loss_fn(outputs, y_train)

    opti.zero_grad()
    return_loss.backward()
    opti.step()

    total_train_step += 1
    if total_train_step % 1000 == 0:
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
torch.save(mlnet, "mlnet.pth")
print("保存模型")
writer.close()
