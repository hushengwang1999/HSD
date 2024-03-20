import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# 设置字体为罗马字体
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
# 示例数据
x = [0, 0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1.0]#时间步长T
# DVScifar10数据
y2 = [94.7, 94.8, 94.8, 95.2, 94.6,94.6,95.0,95.2,94.3,94.3,94.1]#红色的线

# 创建子图网格
fig = plt.figure(figsize=(18,9))

# 第一个子图
plt.plot(x, y2, label=r'$\lambda=0.7$', color='#E76254', marker='o',linewidth=2.5)
#plt.legend(loc='lower right',fontsize=28)
plt.grid(True)
plt.xlabel(r'$\alpha$',fontsize=30)
plt.ylabel("Top-1 Accuracy (%)",fontsize=30)
plt.xticks([0, 0.1, 0.2,0.3,0.4, 0.5, 0.6, 0.7,0.8,0.9,1.0], fontsize=30)
plt.yticks(fontsize=30)
for i, txt in enumerate(y2):
    plt.text(x[i], y2[i], f'{y2[i]}', fontsize=20, ha='right', va='bottom')

plt.tick_params(labelsize=30)
#plt.show()
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('dp.pdf')
pdf.savefig(bbox_inches='tight')
pdf.close()
plt.show()