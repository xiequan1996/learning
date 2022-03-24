import copy
import torch
import torch.nn as nn
import time

from tensorboardX import SummaryWriter

from file_deal import *

init_label = {"e11": 75, "e22": 70, "e33": 10, "g12": 7.5, "g13": 40, "g23": 7}
train_path = "data/train1"
val_path = "data/val"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLNet(nn.Module):
    def __init__(self):
        super(MLNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 2, 2),
            nn.ReLU(),
            nn.Conv2d(8, 64, 2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 75 * 4, 1200),
            nn.ReLU(),
            nn.Linear(1200, 60),
            nn.ReLU(),
            nn.Linear(60, 6)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# mlnet = MLNet()
mlnet = torch.load("mlnet.pth")
mlnet.to(device)
loss_fn = nn.MSELoss()
loss_fn.to(device)
opti = torch.optim.SGD(mlnet.parameters(), lr=0.0001)
epoch = 1000
total_train_step = 0

# 添加tensorboard
writer = SummaryWriter("../logs")

folders = os.listdir(train_path)
num = len(folders)
a = 0
start_time = time.time()
for i in range(epoch):
    for folder in folders:
        folder_label = copy.deepcopy(init_label)
        foldername_list = folder.split('-')
        folder_label[foldername_list[0]] = int(foldername_list[1])
        folder_label[foldername_list[2]] = int(foldername_list[3])
        folder_mat, folder_labels = folder2matrix(train_path + '/' + folder, folder_label)
        mat_tensor = torch.tensor(folder_mat, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(folder_labels, dtype=torch.float32, device=device)
        mat_tensor = mat_tensor.reshape(-1, 1, 300, 16)

        outputs = mlnet(mat_tensor)
        return_loss = loss_fn(outputs, labels_tensor)

        opti.zero_grad()
        return_loss.backward()
        opti.step()

        # total_train_step += 1
        # if total_train_step % 100 == 0:
        #     print("训练次数：{},Loss:{}".format(total_train_step, return_loss.item()))
        # writer.add_scalar("loss_{}".format(a % num), return_loss, i)
        # a += 1

    with torch.no_grad():
        test_mat, test_labels = folder2matrix(val_path, init_label)
        test_mat = torch.tensor(test_mat, dtype=torch.float32, device=device)
        test_labels = torch.tensor(test_labels, dtype=torch.float32, device=device)
        test_mat = test_mat.reshape(-1, 1, 300, 16)
        test_outputs = mlnet(test_mat)
        test_loss = loss_fn(test_outputs, test_labels)
        # writer.add_scalar("test_loss", test_loss, i)
        print("测试集第{}次上的Loss:{}".format(i + 1, test_loss))

end_time = time.time()
print(end_time - start_time)
torch.save(mlnet, "mlnet.pth")
print("保存模型")
writer.close()
