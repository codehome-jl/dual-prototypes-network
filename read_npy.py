import numpy as np



data = np.load("pro.npy",allow_pickle=True)

data = np.array(data)

source  = []

for i in range(25):
        source.append(np.array(data[i][1]))
source = np.array(source)
source = source.transpose()
source = source.flatten()
# source = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,
#           0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,
#           0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,
#           0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,
#           0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
#           ]
# source = np.array(source)
#
# source = source.flatten()
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
