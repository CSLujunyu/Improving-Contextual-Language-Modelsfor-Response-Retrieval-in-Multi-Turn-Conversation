# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

loss_x = np.array([0,1,2,4,8,16,24,32,40,50])
loss_SP_wo_pretraining = np.array([0.98522, 0.7866, 0.7243, 0.66179, 0.7537, 0.6189, 0.71477, 0.7355, 0.71359, 0.7515])
loss_SP_wo_finetuning =  np.array([0.6920, 0.5613, 0.530, 0.5975, 0.560, 0.563, 0.514, 0.4909, 0.5593, 0.5237])
loss_SP = np.array([0.5404, 0.478, 0.393, 0.3610, 0.2710, 0.3091, 0.2781, 0.2564, 0.2964, 0.27217])
loss_SS = np.array([0.5339, 0.458, 0.383, 0.328, 0.2851, 0.2509, 0.2512, 0.2315, 0.2458, 0.24217])

x = np.array([0,1,2,4,8,16,24,32,40,50])
SP_wo_pretraining = np.array([0.1139, 0.1226, 0.1382, 0.1664, 0.135, 0.152, 0.2345, 0.1408, 0.1831, 0.173])
SP_wo_finetuning =  np.array([0.0507, 0.1938, 0.279, 0.240, 0.367, 0.351, 0.437, 0.503, 0.431, 0.402])
SP = np.array([0.0507, 0.4366, 0.580, 0.7054, 0.74202, 0.76932, 0.77942,  0.78452, 0.7772, 0.778])
SS = np.array([0.0939, 0.478, 0.626, 0.71312, 0.77804, 0.79646, 0.81226, 0.80518, 0.80744, 0.80674])

# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
# plt.figure(figsize=(5, 5), dpi=300)

# 创建图形
fig = plt.figure(figsize=(10, 5), dpi=300)
fig.set_facecolor('white')

# 第一行第一列图形
ax1 = plt.subplot(1, 2, 1)

# plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax1.spines['top'].set_visible(False)  # 去掉上边框
ax1.spines['right'].set_visible(False)  # 去掉右边框
plt.tick_params(direction='out', top=False, right=False)
plt.plot(loss_x, loss_SP_wo_pretraining, marker='o', color="green", label="BERT(w/o pretraining)", linewidth=3.0, markersize=7)
plt.plot(loss_x, loss_SP_wo_finetuning, marker='v', color="darkorchid", label="BERT(w/o finetuning)", linewidth=3.0, markersize=7)
plt.plot(loss_x, loss_SP, marker='d', color="blue", label="BERT", linewidth=3.0, linestyle='--', markersize=7)
plt.plot(loss_x, loss_SS, marker='s', color="red", label="BERT-SS", linewidth=3.0, linestyle='--', markersize=7)

# group_labels = ['Top 0-5%', 'Top 5-10%', 'Top 10-20%', 'Top 20-50%', 'Top 50-70%', ' Top 70-100%']  # x轴刻度的标识
plt.xticks(x, fontsize=12)  # 默认字体大小为10
plt.yticks(fontsize=12)
# plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
plt.xlabel("Iteration($\mathregular{10^3}$)", fontsize=15, fontweight='bold')
plt.ylabel("Loss", fontsize=15, fontweight='bold')
# plt.xlim(0, 10)  # 设置x轴的范围
# plt.ylim(8, 90)

# plt.legend()          #显示各曲线的图例
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), numpoints=1, frameon=True)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=13, fontweight='bold')  # 设置图例字体的大小和粗细

# 第一行第二列图形
ax2 = plt.subplot(1, 2, 2)
ax2.spines['top'].set_visible(False)  # 去掉上边框
ax2.spines['right'].set_visible(False)  # 去掉右边框
plt.tick_params(direction='out', top=False, right=False)
plt.plot(x, SP_wo_pretraining, marker='o', color="green", label="BERT-SP(w/o pretraining)", linewidth=3.0, markersize=7)
plt.plot(x, SP_wo_finetuning, marker='v', color="darkorchid", label="BERT-SP(w/o finetuning)", linewidth=3.0, markersize=7)
plt.plot(x, SP, marker='d', color="blue", label="BERT-SP", linewidth=3.0, linestyle='--', markersize=7)
plt.plot(x, SS, marker='s', color="red", label="BERT-SS", linewidth=3.0, linestyle='--', markersize=7)

# group_labels = ['Top 0-5%', 'Top 5-10%', 'Top 10-20%', 'Top 20-50%', 'Top 50-70%', ' Top 70-100%']  # x轴刻度的标识
plt.xticks(x, fontsize=12)  # 默认字体大小为10
plt.yticks(fontsize=12)
# plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
plt.xlabel("Iteration($\mathregular{10^3}$)", fontsize=15, fontweight='bold')
plt.ylabel("R10@1", fontsize=15, fontweight='bold')
# plt.xlim(0, 10)  # 设置x轴的范围
# plt.ylim(8, 90)


plt.savefig('./pretrain-finetune-analysis.png', dpi=300, bbox_inches="tight")
plt.show()