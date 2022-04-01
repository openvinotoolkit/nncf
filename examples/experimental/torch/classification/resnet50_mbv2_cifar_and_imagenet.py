import matplotlib.pyplot as plt
import csv

resnet50_imagenet = '../data/results_imagenet.csv'
resnet50_cifar = '../data/search_resnet50_CIFAR10_3000.csv'
mobilenetV2_cifar = '../data/search_mbv2_DV2_CIFAR10_3000.csv'
show_cross = True
size_pm = 120
size_sub = 9
marker_sub = 'D'
size_bnas = 120
marker_bnas = 'o'
border_size = 2.5
acc1_set = []
m_macs_set = []
num_params_set = []

with open(resnet50_imagenet, 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    for row in reader:
        top1 = float(row[1])
        top5 = float(row[2])
        flops = float(row[3])
        model_params = float(row[4])

        num_params_set.append(model_params/1000000)
        macs = flops*1000
        m_macs_set.append(macs)
        acc1_set.append(top1)

colors = range(len(m_macs_set))

fig, ax = plt.subplots(1, 3, figsize=(15,4), gridspec_kw={'width_ratios': [2.5, 2.5, 2.5]})
colormap = plt.cm.get_cmap('viridis_r')

fig.suptitle('Image Classification Super-Networks Generated From Pre-trained Models')

im = ax[0].scatter(m_macs_set, acc1_set, s=size_sub, c=colors, alpha=0.5, marker=marker_sub, cmap=colormap)
ax[0].scatter([1395], [75.03], marker=marker_bnas, s=size_bnas, color='yellow', label='BootstrapNAS A', edgecolors='black', linewidth=border_size)
ax[0].scatter([2420], [76.58], marker=marker_bnas, s=size_bnas, color='white', label='BootstrapNAS B', edgecolors='black', linewidth=border_size)
ax[0].scatter([1920], [76.22], marker=marker_bnas, s=size_bnas, color='orange', label='BootstrapNAS C', edgecolors='gray', linewidth=border_size)
ax[0].scatter([4089], [76.13], marker="s", s=size_pm, color='blue', label='Torchvision ResNet-50', edgecolors='black')
if show_cross:
    ax[0].plot([1400, 4000], [76.13, 76.13], '--')
ax[0].text(3580, 75.9, "Input")
ax[0].text(3400, 75.7, "Pre-Trained")
ax[0].text(3560, 75.5, "Model")

ax[0].set_title("ResNet-50 | Imagenet")

ax[0].set_xlabel("MACs [M]")
ax[0].set_ylabel("Top1 Accuracy [%]")  #Imagenet - Top1 Accuracy [%]")

ax[0].legend()

acc1_set = []
m_macs_set = []
num_params_set = []

with open(resnet50_cifar, 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    for row in reader:
        top1 = float(row[2])
        top5 = float(row[3])
        flops = float(row[4])
        model_params = float(row[5])

        num_params_set.append(model_params/1000000.0)
        macs = flops/2000000.0
        m_macs_set.append(macs)
        acc1_set.append(top1)

colors = range(len(m_macs_set)) # , 0, -1)

ax[1].scatter(m_macs_set, acc1_set, s=size_sub, c=colors, alpha=0.5, marker=marker_sub, cmap=colormap)
ax[1].scatter([89.17], [92.66], marker=marker_bnas, s=size_bnas, color='yellow', label='BootstrapNAS A-RC', edgecolors='black', linewidth=border_size)
ax[1].scatter([115.90], [93.70], marker=marker_bnas, s=size_bnas, color='white', label='BootstrapNAS B-RC', edgecolors='black', linewidth=border_size)
ax[1].scatter([325.80], [93.65], marker='s', s=size_bnas, color='blue', label='Phan ResNet-50', edgecolors='black')
if show_cross:
    ax[1].plot([90, 320], [93.65, 93.65], '--')
ax[1].text(290, 93.50, "Input")
ax[1].text(275, 93.34, "Pre-Trained")
ax[1].text(288, 93.18, "Model")
ax[1].set_title("ResNet-50 | CIFAR-10")
ax[1].set_xlabel("MACs [M]")
ax[1].set_ylabel("Top1 Accuracy [%]")

ax[1].legend()

acc1_set = []
m_macs_set = []
num_params_set = []

with open(mobilenetV2_cifar, 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    for row in reader:
        top1 = float(row[2])
        top5 = float(row[3])
        flops = float(row[4])
        model_params = float(row[5])

        num_params_set.append(model_params/1000000.0)
        macs = flops/2000000.0
        m_macs_set.append(macs)
        acc1_set.append(top1)

colors = range(len(m_macs_set))

ax[2].scatter(m_macs_set, acc1_set, s=size_sub, c=colors, alpha=0.5, marker=marker_sub, cmap=colormap)
if show_cross:
    ax[2].plot([27, 87.98], [93.91, 93.91], '--')

ax[2].text(78.5, 93.68, "Input")
ax[2].text(74.5, 93.48, "Pre-Trained")
ax[2].text(78.4, 93.28, "Model")
ax[2].scatter([24.71], [92.85], marker=marker_bnas, s=size_bnas, color='yellow', label='BootstrapNAS A-MC', edgecolors='black', linewidth=border_size)
ax[2].scatter([36.75], [93.91], marker=marker_bnas, s=size_bnas, color='white', label='BootstrapNAS B-MC', edgecolors='black', linewidth=border_size)
ax[2].scatter([87.98], [93.91], marker="s", s=size_pm, color='blue', label='Phan MobileNetV2', edgecolors='black')
ax[2].set_title("MobileNetV2 | CIFAR-10")
ax[2].set_xlabel("MACs [M]")
ax[2].set_ylabel("Top1 Accuracy [%]")

ax[2].legend()
plt.show()