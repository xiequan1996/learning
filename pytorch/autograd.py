import torch
from torch import mean

x = torch.ones(2, 2, requires_grad=True)  # 设置 requires_grad=True 来跟踪tensor的计算
print(x)
y = x + 2
print(y)
z = y * y * 3
out = mean(z)  # 求平均值
print(z, out)
out.backward()  # 第一次执行梯度求导，动态图机制，默认每次反向传播后都会释放计算图
print(x.grad)

w = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([2.0], requires_grad=True)
a = torch.add(w, x)
b = torch.add(w, 1)
y0 = torch.mul(a, b)  # y0 = (x+w) * (w+1)
y1 = torch.add(a, b)  # y1 = (x+w) + (w+1)    dy1/dw = 2
# 把两个 loss 拼接都到一起
loss = torch.cat([y0, y1], dim=0)  # [y0, y1]
# 设置两个 loss 的权重: y0 的权重是 1，y1 的权重是 2
grad_tensors = torch.tensor([1.0, 2.0])
loss.backward(
    gradient=grad_tensors
)  # gradient 传入 torch.autograd.backward()中的grad_tensors
# 最终的 w 的导数由两部分组成。∂y0/∂w * 1 + ∂y1/∂w * 2
print(w.grad)

# torch.autograd.grad()的返回结果是一个 tuple，需要取出第 0 个元素才是真正的梯度
x = torch.tensor([3.0], requires_grad=True)
y = torch.pow(x, 2)
# 如果需要求 2 阶导，需要设置 create_graph=True，让一阶导数 grad_1 也拥有计算图
grad_1 = torch.autograd.grad(y, x, create_graph=True)
print(grad_1)
# 这里求 2 阶导
grad_2 = torch.autograd.grad(grad_1[0], x)
print(grad_2)

# 叶子节点不可执行 inplace 操作。
# 以加法来说，inplace 操作有a += x，a.add_(x)，改变后的值和原来的值内存地址是同一个。
# 非 inplace 操作有a = a + x，a.add(x)，改变后的值和原来的值内存地址不是同一个
