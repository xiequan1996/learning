import copy
import torch
import torch.nn as nn
import time
from file_deal import *

init_label = {"e11": 75, "e22": 70, "e33": 10, "g12": 7.5, "g13": 40, "g23": 7}
train_path = "data/train"
val_path = "data/val"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLNet2(nn.Module):
    def __init__(self):
        super(MLNet2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(8, 64, 2, 2),
            nn.Flatten(),
            nn.Linear(64 * 150, 120),
            nn.ReLU(),
            nn.Linear(120, 6)
        )

    def forward(self, x):
        x = self.model(x)
        return x


mlnet2 = MLNet2()
# mlnet = torch.load("mlnet_2.pth")
mlnet2.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
opti = torch.optim.SGD(mlnet2.parameters(), lr=0.1)
epoch = 10
total_train_step = 0

if __name__ == '__main__':
    folders = os.listdir(train_path)
    start_time = time.time()
    for i in range(epoch):
        for folder in folders:
            folder_label = copy.deepcopy(init_label)
            foldername_list = folder.split('-')
            folder_label[foldername_list[0]] = int(foldername_list[1])
            folder_label[foldername_list[2]] = int(foldername_list[3])
            folder_mat, folder_labels = folder2matrix_2(train_path + '/' + folder, init_label)
            mat_tensor = torch.tensor(folder_mat, dtype=torch.float32, device=device)
            labels_tensor = torch.tensor(folder_labels, dtype=torch.float32, device=device)

            outputs = mlnet2(mat_tensor)
            return_loss = loss_fn(outputs, labels_tensor)

            opti.zero_grad()
            return_loss.backward()
            opti.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数：{},Loss:{}".format(total_train_step, return_loss.item()))

    # total_test_loss=0
    # with torch.no_grad():
    #     test_mat, test_labels = folder2matrix(val_path, init_label)
    #     test_mat = torch.tensor(test_mat, dtype=torch.float32)
    #     test_labels = torch.tensor(test_labels, dtype=torch.float32)
    #     test_mat = test_mat.reshape(-1, 1, 300, 8)
    #     test_outputs = mlnet(test_mat)
    #     print(test_labels.shape)
    #     print(test_outputs.shape)
    #     test_loss = loss_fn(test_outputs, test_labels)
    #     print(test_loss.item())
end_time = time.time()
print(end_time - start_time)
torch.save(mlnet2, "mlnet_2.pth")
print("保存模型")
