import torch

# torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None)
# input:输入tensor
# p (int, float, inf, -inf, 'fro', 'nuc', optional):范数计算中的幂指数值。默认为'fro'
# dim (int，2-tuple，2-list， optional):指定计算的维度。
# 如果是一个整数值，向量范数将被计算；如果是一个大小为2的元组，矩阵范数将被计算；
# 如果为None，当输入tensor只有两维时矩阵计算矩阵范数；
# 当输入只有一维时则计算向量范数。如果输入tensor超过2维，向量范数将被应用在最后一维
# keepdim（bool，optional):指明输出tensor的维度dim是否保留。如果dim=None或out=None,则忽略该参数。默认值为False，不保留
# out（Tensor, optional）:tensor的输出。如果dim=None或out=None,则忽略该参数。
# dtype（torch.dtype，optional）:指定返回tensor的期望数据类型。如果指定了该参数，在执行该操作时输入tensor将被转换成 :attr:’dtype’
a = torch.arange(9, dtype=torch.float) - 4
a = a.reshape(3, 3)
print(a)
print(a.norm())
print(a.norm(float('inf')))

c = torch.arange(8, dtype=torch.float).reshape(2, 2, 2)
print(c)
print(torch.norm(c, dim=0))
print(torch.norm(c, dim=1))
print(torch.norm(c, p=1, dim=1))

# 向量范数
# 1-范数 向量元素绝对值之和
# 2-范数 向量元素平方和再开方
# 正无穷-范数 向量元素绝对值中的最大值
# 负无穷-范数 向量元素绝对值中的最小值
# 0-范数 向量元素的非零个数

# 矩阵范数
# 1-范数 所有矩阵列向量绝对值之和的最大值
# 2-范数 AtA的最大特征值再开方  也是||Ax||/||x||的最大值
# 无穷-范数 所有矩阵行向量绝对值之和的最大值
# F-范数 矩阵元素绝对值的平方和再开平方

# 矩阵范数默认计算F范数
