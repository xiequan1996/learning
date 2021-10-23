'''
Descripttion: RMB二分
version: 
Author: xiequan
Date: 2021-10-16 14:11:28
LastEditors: Please set LastEditors
LastEditTime: 2021-10-16 18:03:19
'''
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from moduel import lenet


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed()  # 设置随机种子

# 参数设置
rmb_label = {"1": 0, "100": 1}
train_dir = '../Dataset/rmb_split/train'
valid_dir = '../Dataset/rmb_split/valid'
norm_mean = [0.485, 0.456, 0.406]  # 通道均值
norm_std = [0.229, 0.224, 0.225]  # 通道标准差
# 设置训练集的数据增强和转化
train_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),  # 随机裁剪，上下左右填充，默认常量
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
)

# 设置验证集的数据增强和转化，不需要 RandomCrop
valid_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
)
BATCH_SIZE = 64  # 一个迭代批的大小
MAX_EPOCH = 10
LR = 0.01
log_interval = 10
val_interval = 1

# 继承Dataset,重写__getitem()__方法和__len__()方法
class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    # 将data_dir里的所有img的地址和标签以元组形式保存到data_info
    def get_img_info(data_dir):
        data_info = list()
        # data_dir 是训练集、验证集或者测试集的路径
        for root, dirs, _ in os.walk(data_dir):  # 遍历文件夹名
            # 遍历类别
            # dirs ['1', '100']
            for sub_dir in dirs:
                # 文件列表
                img_names = os.listdir(
                    os.path.join(root, sub_dir)
                )  # 返回sub_dir下的所有文件及文件夹
                # 取出 jpg 结尾的文件
                # filter()函数 过滤序列，返回新的列表
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    # 图片的绝对路径
                    path_img = os.path.join(root, sub_dir, img_name)
                    # 标签，这里需要映射为 0、1 两个类别
                    label = rmb_label[sub_dir]
                    # 保存在 data_info 变量中
                    data_info.append((path_img, int(label)))
        return data_info

    def __getitem__(self, index):
        # 通过 index 读取样本
        path_img, label = self.data_info[index]
        # 注意这里需要 convert('RGB')
        # convert() PIL库的图片模式转换
        img = Image.open(path_img).convert('RGB')  # 0~255
        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等
        # 返回是样本和标签
        return img, label

    # 返回所有样本的数量
    def __len__(self):
        return len(self.data_info)


# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)  # 训练集
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)  # 验证集

# 构建DataLoder
# 其中训练集设置 shuffle=True，表示每个 Epoch 都打乱样本
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# 采用经典的Lenet图片分类网络
net = lenet(classes=2)
net.initialize_weights()

# 设置损失函数，这里使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 设置优化器，这里采用SGD优化器
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1
)  # 设置学习率下降策略

# 训练
train_curve = list()
valid_curve = list()

# 迭代训练模型
for epoch in range(MAX_EPOCH):

    loss_mean = 0.0
    correct = 0.0
    total = 0.0

    net.train()
    # 遍历 train_loader 取数据
    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        outputs = net(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i + 1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print(
                "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch,
                    MAX_EPOCH,
                    i + 1,
                    len(train_loader),
                    loss_mean,
                    correct / total,
                )
            )
            loss_mean = 0.0
    scheduler.step()  # 更新学习率
    # 每个 epoch 计算验证集得准确率和loss
    ...
    ...
