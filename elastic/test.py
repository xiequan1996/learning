import numpy as np
import torch

# def file2matrix(path, name):
#     err = True
#     file_temp = open(path + '/' + name, "r")
#     lines = file_temp.readlines()
#     line0 = lines[0]
#     line0_list = line0.strip().split('\t')
#     print(line0_list)
#     s0_index=line0_list.index('Sym0')
#     a0_index=line0_list.index('Asym0')
#     s1_index=line0_list.index('Sym1')
#     a1_index=line0_list.index('Asym1')
#     mat_temp = np.zeros((300, 8))
#     for j in range(300):
#         line = lines[j + 3]
#         line_list = line.split('\t')
#         try:
#             mat_temp[j, 0:2] = line_list[s0_index:s0_index+2]
#             mat_temp[j, 2:4] = line_list[a0_index:a0_index+2]
#             mat_temp[j, 4:6] = line_list[s1_index:s1_index+2]
#             mat_temp[j, 6:8] = line_list[a1_index:a1_index+2]
#         except ValueError:
#             err = False
#             break
#     return mat_temp, err
#
#
# path1 = "data/train/e11-30-e22-85"
# name1 = "e33-60-gr.txt"
# mat, err1 = file2matrix(path1, name1)
# print(mat)
import torch

list1=[] #10-90
for i in range(41):
    list1.append(10+2*i)
tensor1=torch.tensor(list1,dtype=torch.float32)
print(tensor1.mean()) #50
print(tensor1.std())  #23.9583

list2=[] #0-50
for i in range(26):
    list2.append(2*i)
tensor2=torch.tensor(list2,dtype=torch.float32)
print(tensor2.mean()) #25
print(tensor2.std())  #15.2971

print(torch.cuda.is_available())
