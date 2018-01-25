import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.rcParams.update({'font.size': 14})

fig,ax = plt.subplots(1, figsize=(8,6))

# create some x data and some integers for the y axis
#x = np.array([0,5,6,7])
x = np.array([100,28.6,14.3,0])
y = np.array([1135,180,143,127])

x1 = np.array([100,36.8,10.5,0])
y1 = np.array([2049,942,450,153])

x2 = np.array([100,50,25,12.5,0])
y2 = np.array([444,278,220,213,177])

x3 = np.array([100,28.6,14.3,0])
y3 = np.array([608,89,82,82])

plt.ylabel('Equivalent FLOPs')
plt.xlabel('Percentage of unbinarized layers')

# plot the data
ax.plot(x,y,linestyle='-', marker='s',label='Sketch-A-Net')
ax.plot(x3,y3,linestyle='-', marker='s',label='AlexNet')
ax.plot(x1,y1,linestyle='-', marker='s',label='Resnet18')
ax.plot(x2,y2,linestyle='-', marker='s',label='SqueezeNet')

point_labels1 = ['WBin', 'Hybrid1', 'Hybrid2', 'FBin']
point_labels2 = ['WBin', 'Hybrid1', 'Hybrid2', 'FBin']
point_labels3 = ['WBin', 'Hybrid1', 'Hybrid2', 'Hybrid3', 'FBin']

plt.ylim(50,2200)
plt.xlim(-5,112)

legend = ax.legend(loc='upper left')

matplotlib.rcParams.update({'font.size': 9})

cnt = 0
for xy in zip(x1, y1):
    if cnt < 3:
        ax.annotate('%s' % (point_labels2[cnt]), xy=(xy[0]-5,xy[1]+70), textcoords='data')
    cnt += 1
cnt = 0
for xy in zip(x, y):
    ax.annotate('%s' % (point_labels1[cnt]), xy=(xy[0]+1.1,xy[1]-50), textcoords='data')
    cnt += 1
cnt = 0
for xy in zip(x2, y2):
    ax.annotate('%s' % (point_labels3[cnt]), xy=(xy[0]+1.1,xy[1]+70), textcoords='data')
    cnt += 1

plt.grid(linestyle='--')
plt.show()
