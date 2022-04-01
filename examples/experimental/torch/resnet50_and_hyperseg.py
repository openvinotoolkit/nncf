import numpy as np
import matplotlib.pyplot as plt

N = 3
ind = np.arange(N)
width = 0.27

fig, ax = plt.subplots(1, 2, figsize=(16,5), gridspec_kw={'width_ratios': [2.5, 2.5]})
twin1 = ax[0].twinx()

colors = ['#003f5c', '#bc5090', '#ffa600']
colors = ['#004c6d', '#638cab', '#b3d3ed']
colors = [(180, 68, 65), (136,178,90), (35, 103, 151)]
colors = [(x/255.0, y/255.0, z/255.0) for (x, y, z) in colors]


yvals = [1, 1, 1]
rects1 = ax[0].bar(ind, yvals, width, color=colors[0])
zvals = [2.16,3,3.08]
rects2 = ax[0].bar(ind+width, zvals, width, color=colors[1])
kvals = [1.47,1.7,1.87]
rects3 = ax[0].bar(ind+width*2, kvals, width, color=colors[2])

ind2 = np.arange(3, step=2)
r50 = [76.13, 75.6]
msize = 15
rects4 = twin1.scatter(ind2, r50, msize, marker='o', color=colors[0])
ba = [75.46, 75]
rects5 = twin1.scatter(ind2+width, ba, msize, marker='o', color=colors[1])
bb = [76.58, 75.8]
rects6 = twin1.scatter(ind2+width*2, bb, msize,  marker='o', color=colors[1])

for x, y in zip(ind2, r50):
    label = "{:.2f}".format(y)
    twin1.annotate(label, (x, y), textcoords="offset points",
                 xytext=(0,4),
                 ha='center')

for x, y in zip(ind2+width, ba):
    label = "{:.2f}".format(y)
    twin1.annotate(label, (x, y), textcoords="offset points",
                 xytext=(0,4),
                 ha='center')

for x, y in zip(ind2+width*2, bb):
    label = "{:.2f}".format(y)
    twin1.annotate(label, (x, y), textcoords="offset points",
                 xytext=(0,4),
                 ha='center')

ax[0].set_ylim(0, 5)
twin1.set_ylim(65, 80)
ax[0].set_ylabel('Improvement (Higher is better)')
twin1.set_ylabel('Accuracy Top 1')
ax[0].set_xticks(ind+width)
ax[0].set_xticklabels( ('Latency (FP32)', 'Model Size (reduction)', 'Latency (INT8)') )
ax[0].legend( (rects1[0], rects2[0], rects3[0]), ('ResNet50 Torchvision', 'BootstrapNAS A', 'BootstrapNAS B') )
ax[0].title.set_text("BootstrapNAS | ResNet-50 | Imagenet")

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax[0].text(rect.get_x()+rect.get_width()/2., 1.01*h, '{:.2f}'.format(round(h, 2)),
                ha='center', va='bottom')

twin1.text(width/2-0.1, 77.5, 'FP32 Top 1')
twin1.text(width*8-0.102, 76.6, 'INT8 Top 1')

ax[0].text(1.3, -0.5, 'Intel® Xeon® Gold 6252 CPU @ 2.1GHz (Cascade Lake)', fontsize=7)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

N = 1
ind = np.arange(N)
width = 0.27

twin1_h = ax[1].twinx()
colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']

hseg = [1]
rects1 = ax[1].bar(ind, hseg, width, color=colors[0])
d6 = [1.38]
rects2 = ax[1].bar(ind+width, d6, width, color=colors[1])
d7 = [1.19]
rects3 = ax[1].bar(ind+width*2, d7, width, color=colors[2])
d8 = [1]
rects4 = ax[1].bar(ind+width*3, d8, width, color=colors[3])


ind2 = np.arange(1)
print(ind2)

hseg = [0.765]
msize = 15
rects5 = twin1_h.scatter(ind2, hseg, msize, marker='v', color=colors[0])
d6 = [0.7807]
rects6 = twin1_h.scatter(ind2+width, d6, msize, marker='o', color=colors[1])
d7 = [0.7936]
rects7 = twin1_h.scatter(ind2+width*2, d7, msize,  marker='^', color=colors[1])
d8 = [0.8099]
rects8 = twin1_h.scatter(ind2+width*3, d8, msize,  marker='.', color=colors[1])

for x, y in zip(ind2, hseg):
    label = "{:.3f}".format(y)
    twin1_h.annotate(label, (x, y), textcoords="offset points",
                 xytext=(0,4),
                 ha='center')
#
for x, y in zip(ind2+width, d6):
    label = "{:.3f}".format(y)
    twin1_h.annotate(label, (x, y), textcoords="offset points",
                 xytext=(0,4),
                 ha='center')

for x, y in zip(ind2+width*2, d7):
    label = "{:.3f}".format(y)
    twin1_h.annotate(label, (x, y), textcoords="offset points",
                 xytext=(0,4),
                 ha='center')

for x, y in zip(ind2+width*3, d8):
    label = "{:.3f}".format(y)
    twin1_h.annotate(label, (x, y), textcoords="offset points",
                 xytext=(0,4),
                 ha='center')

ax[1].set_ylim(0, 2.5)
twin1_h.set_ylim(0.70, 0.82)
ax[1].set_ylabel('Relative Latency Improvement (Higher is better)')
twin1_h.set_ylabel('mIoU')
ax[1].set_xticklabels([])

ax[1].set_xlabel('Latency (FP32)')
ax[1].legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('Hyperseg Lite', 'BootstrapNAS Seg A', 'BootstrapNAS Seg B', 'BootstrapNAS Seg C'), loc='upper left' )
ax[1].title.set_text("BootstrapNAS | Hyperseg")
#
def autolabel_h(rects):
    for rect in rects:
        h = rect.get_height()
        ax[1].text(rect.get_x()+rect.get_width()/2., 1.01*h, '{:.2f}'.format(round(h, 2)),
                ha='center', va='bottom')

ax[1].text(0.40, -0.25, 'Intel® Xeon® Gold 6252 CPU @ 2.1GHz (Cascade Lake)', fontsize=7)
#
autolabel_h(rects1)
autolabel_h(rects2)
autolabel_h(rects3)
autolabel_h(rects4)

plt.show()

