'''
Descripttion: 一个由多维数组对象和用于处理数组的例程集合组成的库
version: 
Author: xiequan
Date: 2021-07-30 17:24:23
LastEditors: Please set LastEditors
LastEditTime: 2021-09-22 21:45:19
'''
import numpy as np

# 数据类型
a = np.array([[1, 2], [3, 4]])
b = np.array([1, 2, 3, 4, 5, 6], ndmin=3)  # ndmin 数组的最小维度  默认为0，其实也1维
c = np.array([1, 2, 3], dtype=complex)  # dtype 数组的数据类型
dt = np.dtype([('age', np.int8)])
d = np.array([(10,), (20,), (30,)], ndmin=2, dtype=dt)  # 结构体化数据类型
student = np.dtype(
    [('name', 'S20'), ('marks', 'f4'), ('age', 'i1')]
)  #  < 小端在最小地址；> 小端在最大地址
print('~' * 5 + '数据类型' + '~' * 5)
print(a)
print(b)
print(c)
print(d['age'])
print(student)

# 数组属性
e = np.array([[1, 2, 3], [4, 5, 6]])
e.shape = (3, 2)
f = np.arange(24)  # 一维数组
f1 = f.reshape(2, 4, 3)  # 高，宽，长
g = np.array([1, 2, 3, 4, 5], dtype=np.int8)  # int8为一个字节
print('~' * 5 + '数据属性' + '~' * 5)
print(e)
print(f1[1, 2, 1])
print(g.itemsize)  # 每个元素的字节单位长度

# 数组创建
h = np.empty([3, 2], dtype=int)  # 第一个参数是形状;第二个参数是数组类型,默认是float
h1 = np.zeros((2, 2), dtype=[('g', 'i4'), ('y', 'i4')])
h1['g'][0, 0] = 10
h2 = np.ones([2, 2])
print('~' * 5 + '数据创建' + '~' * 5)
print(h)
print(h1)
print(h2)

# 从现有数据的数组创建
ar1 = [(1, 2, 3), (4, 5)]
i = np.asarray(ar1, dtype=object)
str1 = b'Hello World'
i1 = np.frombuffer(str1, dtype='S1')
it = iter([1, 2, 3, 4, 5])
i2 = np.fromiter(it, dtype=float)
print('~' * 5 + '从现有数据的数组创建' + '~' * 5)
print(i)
print(i1)
print(i2)

# 从数值范围创建数组
j = np.arange(10, 20, 2)
j1 = np.linspace(10, 20, 5, endpoint=False, retstep=True)  # 起点，终点，数量，包含终点，返回数组和步长
j2 = np.logspace(1, 10, num=10, base=2)  # linspace的对数进阶版
print('~' * 5 + '从数值范围创建数组' + '~' * 5)
print(j)
print(j1)
print(j2)

# 切片和索引
k = np.arange(10)
k1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
k2 = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])  # np.nan 是float型的空值，差不多是none
sl = slice(2, 7, 2)
rows = np.array([[0, 0], [2, 3]])
cols = np.array([[0, 1], [0, 2]])
print('~' * 5 + '切片和索引' + '~' * 5)
print(k[sl])
print(k[2:7:2])
print(k1[..., 1:])  # 各行的第二列及其剩余元素
print(k1[rows, cols])
print(k1[k1 > 5])
print(k2[~np.isnan(k2)])  # 使用取补运算符~来过滤nan

# 广播
# 两个数组的后缘维度的轴长度相符，或其中的一方的长度为1，则认为它们是广播兼容的
# 就是相同（或差一)维度，0和1可以在范围内任取
arr2 = np.arange(90)
arr2 = arr2.reshape(3, 5, 6)
arr3 = np.array([[0, 10, 100, 1000, 1000, 10000]])
print('~' * 5 + '广播' + '~' * 5)
print(arr2)
print(arr2 + arr3)  # （1,6)扩展成（5,6）,再扩展成(3，5，6)

# 数组上的迭代
# 迭代顺序只看内存顺序,reshape和转置T都不改变内存顺序
arr4 = np.array([1, 2, 3, 4])
arr5 = np.arange(0, 60, 5)
arr5 = arr5.reshape(3, 4)
arr5t = arr5.T
arr5c = arr5t.copy(order='C')  # 深拷贝,重新分配了另一个内存;C是行排列
arr5f = arr5t.copy(order='F')  # F是列排列
print('~' * 5 + '数组上的迭代' + '~' * 5)
print(arr5)
print(arr5t)
list_arr = []
for x in np.nditer(arr5t):
    list_arr.append(x)
print(list_arr)
list_arr.clear()
print(arr5c)
for x in np.nditer(arr5c):
    list_arr.append(x)
print(list_arr)
list_arr.clear()
print(arr5f)
for x in np.nditer(arr5f):
    list_arr.append(x)
print(list_arr)
list_arr.clear()
for x, y in np.nditer([arr5, arr4]):  # 广播迭代
    list_arr.append('%d:%d' % (x, y))
print(list_arr)

# 数组的操作
arr6 = np.arange(18).reshape(2, 3, 3)
print('~' * 5 + '数组的操作' + '~' * 5)
print(arr6.flatten())  # 返回一维数组，相当于拷贝
print(arr6.ravel())  # 返回一维数组，相当于视图
print(arr6.transpose((1, 0, 2)))  # 将0轴和1轴交换
print(np.rollaxis(arr6, 2, 1))  # 向后滚动轴
print(arr6.swapaxes(2, 1))  # 交换两个轴
print(np.expand_dims(arr6, axis=1))  # 插入新的轴来扩展数组
