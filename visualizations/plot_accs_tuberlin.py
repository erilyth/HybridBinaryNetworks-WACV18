import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.rcParams.update({'font.size': 14})

fig,ax = plt.subplots(1, figsize=(8,6))

# create some x data and some integers for the y axis
#x = np.array([0,5,6,7])
x = np.array([100,28.6,14.3,0])
y = np.array([73.0,73.1,71.0,59.6])

x1 = np.array([100,36.8,10.5,0])
y1 = np.array([73.4,73.8,72.8,68.8])

x2 = np.array([100,50,25,12.5,0])
y2 = np.array([66.7,64.8,61.6,59.3,56.8])

plt.ylabel('Accuracy')
plt.xlabel('Percentage of WeightBinConv layers')

# plot the data
ax.plot(x,y,linestyle='-', marker='s',label='Sketch-A-Net')
ax.plot(x1,y1,linestyle='-', marker='s',label='Resnet18')
ax.plot(x2,y2,linestyle='-', marker='s',label='SqueezeNet')

point_labels1 = ['WBin', 'Hybrid1', 'Hybrid2', 'FBin']
point_labels2 = ['WBin', 'Hybrid1', 'Hybrid2', 'FBin']
point_labels3 = ['WBin', 'Hybrid1', 'Hybrid2', 'Hybrid3', 'FBin']

plt.ylim(45,78)
plt.xlim(-5,112)

legend = ax.legend(loc='lower right')

matplotlib.rcParams.update({'font.size': 9})

cnt = 0
for xy in zip(x, y):
    ax.annotate('%s\n(%s)' % (point_labels1[cnt],xy[1]), xy=(xy[0]+1.3,xy[1]-1.9), textcoords='data')
    cnt += 1

cnt = 0
for xy in zip(x1, y1):
    ax.annotate('%s\n(%s)' % (point_labels2[cnt],xy[1]), xy=(xy[0]+0.2,xy[1]+1), textcoords='data')
    cnt += 1

cnt = 0
for xy in zip(x2, y2):
    ax.annotate('%s\n(%s)' % (point_labels3[cnt],xy[1]), xy=(xy[0]+1.3,xy[1]-1.7), textcoords='data')
    cnt += 1

plt.grid(linestyle='--')
plt.show()
