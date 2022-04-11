import copy
import os

import numpy as np

path = "./data/val1"
freq = 0.3
init_label = {"e11": 75, "e22": 70, "e33": 10, "g12": 7.5, "g13": 40, "g23": 7, "v12": 0.012, "v13": 0.777, "v23": 0.45,
              "freq": freq}


def file_freq(folder1, index, freq1, label):
    list_file1 = os.listdir(folder1)
    file_gr = list_file1[index]
    file_ph = list_file1[index + 1]
    name = file_gr.split("-")
    label[name[0]] = int(name[1])
    list_gr, correct_gr = file2v(folder1, file_gr, freq1)
    list_ph, correct_ph = file2v(folder1, file_ph, freq1)
    label["s0_gr"] = list_gr[0]
    label["a0_gr"] = list_gr[1]
    label["s0_ph"] = list_ph[0]
    label["a0_ph"] = list_ph[1]
    return label, correct_gr & correct_ph


def file2v(folder2, file, freq2):
    f = open(folder2 + "/" + file, 'r')
    lines = f.readlines()
    line0_list = lines[0].strip().split('\t')
    try:
        s0_index = line0_list.index('Sym0')
        a0_index = line0_list.index('Asym0')
    except ValueError:
        print(folder2 + "/" + file + "    Not found S0/A0")
        return [0, 0], False
    freq_less_s0 = True
    freq_less_a0 = True
    line1_list = lines[1].strip().split('\t')
    s0_num_list = line1_list[s0_index].split()
    a0_num_list = line1_list[a0_index].split()
    line_index_s0 = 3
    line_index_a0 = 3
    s0_num = int(s0_num_list[0]) + 2
    a0_num = int(a0_num_list[0]) + 2
    while freq_less_s0 or freq_less_a0:
        if freq_less_s0:
            if line_index_s0 <= s0_num:
                line_list = lines[line_index_s0].split('\t')
                if float(line_list[s0_index]) < freq2:
                    line_index_s0 += 1
                else:
                    freq_less_s0 = False
            else:
                print(folder2 + "/" + file + "    S0 less than {}M".format(freq))
                return [0, 0], False
        if freq_less_a0:
            if line_index_a0 <= a0_num:
                line_list = lines[line_index_a0].split('\t')
                if float(line_list[a0_index]) < freq2:
                    line_index_a0 += 1
                else:
                    freq_less_a0 = False
            else:
                print(folder2 + "/" + file + "    A0 less than {}M".format(freq))
                return [0, 0], False
    if (line_index_s0 == 3) or (line_index_a0 == 3):
        print(folder2 + "/" + file + "    more than {}M".format(freq))
        return [0, 0], False
    v_s0 = inter(lines, line_index_s0, s0_index, freq2)
    v_a0 = inter(lines, line_index_a0, a0_index, freq2)
    return [v_s0, v_a0], True


def inter(list_lines, row, column, x):
    list1 = list_lines[row - 1].split('\t')
    list2 = list_lines[row].split('\t')
    return np.interp(x, [float(list1[column]), float(list2[column])],
                     [float(list1[column + 1]), float(list2[column + 1])])


fstream = open("{}M.txt".format(freq), 'a')
list_folder = os.listdir(path)
for folder in list_folder:
    folder_label = copy.deepcopy(init_label)
    foldername_list = folder.split('-')
    folder_label[foldername_list[0]] = int(foldername_list[1])
    folder_label[foldername_list[2]] = int(foldername_list[3])
    list_file = os.listdir(path + "/" + folder)
    num_file = int(len(list_file) / 2)
    for i in range(num_file):
        label_res, correct = file_freq(path + "/" + folder, 2 * i, freq, folder_label)
        if correct:
            for value in label_res.values():
                fstream.write(str(value))
                fstream.write('\t')
            fstream.write('\n')
# fstream = open("{}M_val.txt".format(freq), 'a')
# list_file = os.listdir(path)
# num_file = int(len(list_file) / 2)
# for i in range(num_file):
#     label_res, correct = file_freq(path, 2 * i, freq, init_label)
#     if correct:
#         for value in label_res.values():
#             fstream.write(str(value))
#             fstream.write('\t')
#         fstream.write('\n')
# fstream.close()
