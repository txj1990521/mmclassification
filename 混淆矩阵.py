import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

classes = ['体上', '体下', '十二指肠', '胃体大弯', '胃体小弯', '胃底', '胃窦', '胃角', '贲门', '食管']
# 输入特征矩阵
# confusion_matrix = np.array(
#     [[451, 1, 12, 6, 1, 3, 5, 2],
#      [18, 451, 25, 19, 24, 14, 7, 2],
#      [41, 27, 487, 2, 15, 2, 24, 3],
#      [14, 20, 4, 395, 7, 16, 15, 5],
#      [1, 8, 30, 25, 421, 16, 14, 14],
#      [13, 18, 1, 15, 13, 455, 18, 19],
#      [19, 7, 12, 17, 4, 21, 352, 15],
#      [15, 23, 31, 15, 3, 9, 15, 458]
#      ], dtype=np.int)  # 输入特征矩阵
confusion_matrix = np.array([[6, 6, 0, 2, 0, 4, 0, 0, 2, 0],
                             [0, 42, 0, 13, 8, 0, 5, 2, 0, 2],
                             [0, 0, 49, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 34, 1, 7, 0, 2, 0, 0],
                             [0, 0, 0, 1, 68, 4, 0, 1, 0, 0],
                             [0, 0, 0, 27, 11, 300, 9, 3, 6, 0],
                             [0, 2, 1, 0, 6, 0, 153, 4, 0, 1],
                             [0, 0, 1, 0, 1, 0, 0, 62, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 6, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 18]], dtype=np.int64)
# 矩阵初始化
proportion = []
length = len(confusion_matrix)
print(length)
for i in confusion_matrix:
    for j in i:
        temp = j / (np.sum(i))
        proportion.append(temp)
# print(np.sum(confusion_matrix[0]))
# print(proportion)
pshow = []
for i in proportion:
    pt = "%.2f%%" % (i * 100)
    pshow.append(pt)
proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
pshow = np.array(pshow).reshape(length, length)
# print(pshow)
# 设置混淆矩阵图片样式
config = {
    # "font.family": 'Times New Roman',  # 设置字体类型
    "font.sans-serif": "SimHei",
    "axes.unicode_minus": False,
}
rcParams.update(config)
plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
# (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
# plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=30, fontsize=8)
plt.yticks(tick_marks, classes, rotation=0, fontsize=8)
# 计算准确率数值及颜色渐变设置
iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
for i, j in iters:
    if (i == j):
        plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=8, color='white',
                 weight=5)  # 显示对应的数字
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=8, color='white')
    else:
        plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=8)  # 显示对应的数字
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=8)

plt.ylabel('True label', fontsize=16)
plt.xlabel('Predict label', fontsize=16)
plt.tight_layout()
plt.show()
# plt.savefig('混淆矩阵.png')
