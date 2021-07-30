'''
Descripttion: 一个由多维数组对象和用于处理数组的例程集合组成的库
version: 
Author: xiequan
Date: 2021-07-30 17:24:23
LastEditors: Please set LastEditors
LastEditTime: 2021-07-30 20:06:30
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
print(a)
print(b)
print(c)
print(d['age'])
print(student)

# 数组属性
e = np.array([[1, 2, 3], [4, 5, 6]])
e.shape = (3, 2)
f = np.arange(24)
g = f.reshape(2, 4, 3)
print(e)
print(g)
print(g[1, 2, 1])

