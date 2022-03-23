import torch
import numpy as np

print(torch.version)  # PyTorch version
print(torch.version.cuda)  # Corresponding CUDA version
print(torch.backends.cudnn.version())  # Corresponding cuDNN version
print(torch.cuda.get_device_name(0))  # GPU type

# 数据类型有9种 float16(half)/32(float)/64(double),uint8,int8/16(short)/32(int)/64(long),bool
# data: 被包装的 Tensor。
# grad: data 的梯度。
# grad_fn: 创建 Tensor 所使用的 Function，是自动求导的关键，因为根据所记录的函数才能计算出导数。
# requires_grad: 指示是否需要梯度，并不是所有的张量都需要计算梯度。
# is_leaf: 指示是否叶子节点(张量)
# dtype: 张量的数据类型，如 torch.FloatTensor，torch.cuda.FloatTensor。
# shape: 张量的形状。如 (64, 3, 224, 224)
# device: 张量所在设备 (CPU/GPU)，GPU 是加速计算的关键
arr1 = np.ones((3, 3))
print("ndarray的数据类型：", arr1.dtype)
# 创建存放在 GPU 的数据
# t = torch.tensor(arr, device='cuda')
# 从numpy创建tensor,共享内存
# t=torch.from_numpy(arr1)
t = torch.tensor(arr1, dtype=torch.int8)
print(t)

# 任何使张量会发生变化的操作都有一个前缀 ‘_’。例如：x.copy_(y), x.t_(), 将会改变 x
x = torch.rand(5, 3)  # [0,1]中随机抽样
# randn  是在正态分布(0,1)中随机抽样
y = torch.tensor([3.6, 10, 6.4])
result = torch.empty(5, 3)
torch.add(x, y, out=result)  # 加法
print(x)
print(x.size(), x.view(-1, 5).size())  # view 改变tensor形状
print(x[:, 1])  # 索引
print(result)

# 张量操作
t = torch.ones((2, 3))
t_0 = torch.cat([t, t], dim=0)  # 按照维度进行拼接
t_stack = torch.stack([t, t, t], dim=0)  # 在新建维度上进行拼接
print(t_0.shape)
print(t_stack.shape)

a = torch.ones((2, 7))
list_of_tensors = torch.chunk(a, dim=1, chunks=3)  # 按照维度进行平均切分，3份
for idx, t in enumerate(list_of_tensors):  # 枚举
    print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))
list_of_tensors = torch.split(a, [2, 3, 2], dim=1)  # 按照维度进行平均切分，可指定分量长度
for idx, t in enumerate(list_of_tensors):
    print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))

t = torch.randint(0, 9, size=(3, 3))
mask = t.le(5)  # ge >= ;gt > ;le <= ;lt <
t_select = torch.masked_select(t, mask)  # 按照mask中的true进行索引拼接得到一维张量
print("t:\n{}\nmask:\n{}\nt_select:\n{} ".format(t, mask, t_select))

# reshape 共享内存

# torch.add(input, other, *, alpha=1, out=None)
# 功能：逐元素计算 input + alpha * other。因为在深度学习中经常用到先乘后加的操作。

# torch.addcdiv()
# torch.addcdiv(input, tensor1, tensor2, *, value=1, out=None)
# 计算公式为：out= input+value*tensor1/tensor2

# torch.addcmul(input, tensor1, tensor2, *, value=1, out=None)
# 计算公式为：out=input+ value*tensor1*tensor2
