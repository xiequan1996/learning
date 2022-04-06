path = "./data/train1/e11-42-e22-80/g12-12-ph.txt"
f = open(path, 'r')
lines = f.readlines()
print(lines[0].strip().split('\t'))
print(lines[1].strip().split('\t'))
list1 = lines[1].strip().split('\t')
print(list1[0].split(' '))
print(list1[0].split())
print(list1[0].split())