import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np


def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=20)

    x_ts = ts.fit_transform(feat)


    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final


# 设置散点形状
maker = ['o', 'o', 'o', 'o', 'o']
# 设置散点颜色
colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue']
# 图例名称
Label_Com = ['a', 'b', 'c', 'd','e']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }


def plotlabels(S_lowDWeights, Trure_labels, name):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})


    for index in range(5):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=32, fontweight='normal', pad=20)


# feat = torch.rand(25, 8192)  # 128个特征，每个特征的维度为1024
# label_test1 = [index%5 for index in range(25)]
#
#
# label_test = np.array(label_test1)
#
#
# fig = plt.figure(figsize=(10, 10))
#
# plotlabels(visual(feat), label_test, '(a)')
# plt.savefig("./")
# plt.show()