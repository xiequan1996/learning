import os
import numpy as np


def file2matrix(path, name, label):
    no_err = True
    mat_temp = np.zeros((300, 8))
    file_temp = open(path + '/' + name, "r")
    lines = file_temp.readlines()
    line0 = lines[0]
    line0_list = line0.strip().split('\t')
    try:
        s0_index = line0_list.index('Sym0')
        a0_index = line0_list.index('Asym0')
        s1_index = line0_list.index('Sym1')
        a1_index = line0_list.index('Asym1')
    except ValueError:
        no_err = False
        return mat_temp, label, no_err
    for j in range(300):
        line = lines[j + 3]
        line_list = line.split('\t')
        try:
            mat_temp[j, 0:2] = line_list[s0_index:s0_index + 2]
            mat_temp[j, 2:4] = line_list[a0_index:a0_index + 2]
            mat_temp[j, 4:6] = line_list[s1_index:s1_index + 2]
            mat_temp[j, 6:8] = line_list[a1_index:a1_index + 2]
        except ValueError:
            no_err = False
            break
    name_list = name.split('-')
    label[name_list[0]] = int(name_list[1])
    return mat_temp, label, no_err


def folder2matrix(path, label):
    files = os.listdir(path)
    files_num = len(files)
    mat0 = int(files_num / 2)
    mat = np.zeros((mat0, 300, 16))
    file_index = 0
    labels = np.zeros((mat0, 6))
    for i in range(mat0):
        file1 = files[2 * i]
        file2 = files[2 * i + 1]
        mat1, label, no_err1 = file2matrix(path, file1, label)
        mat2, label, no_err2 = file2matrix(path, file2, label)
        if no_err1 & no_err2:
            mat[file_index, :, 0:8] = mat1
            mat[file_index, :, 8:16] = mat2
            labels[file_index] = list(label.values())
            file_index += 1
        else:
            mat = np.delete(mat, -1, axis=0)
            labels = np.delete(labels, -1, axis=0)
    return mat, labels
