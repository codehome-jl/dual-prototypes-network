
import numpy as np

filename = './data/tar_ucf.csv'
# delimiter参数的作用是指定分隔符,dtype参数的作用是指定数据类型
data = np.genfromtxt(filename, delimiter=' ', dtype=str)

data = data.T

source = []
for i in data:
    for j in i:
        source.append(float(j))
source = np.array(source)
print(source)
temp = []
obj = []
num = -1
for row in range(5):
    for col in range(25):
        num = num+1
        temp.append(row)
        temp.append(col)
        temp.append(source[num])
        obj.append(temp)
        temp = []
print(obj)
