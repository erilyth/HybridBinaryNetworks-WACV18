import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

import matplotlib

matplotlib.rcParams['legend.numpoints'] = 1

matplotlib.rcParams.update({'font.size': 15})

fig,ax = plt.subplots(1, figsize=(13,6))

# create some x data and some integers for the y axis
#x = np.array([0,5,6,7])
y = np.array([7.192249767, 14.39824368, 13.05819078, 15.47645223, 19.61632079, 41.40185173])
x = np.array(range(2,len(y)+2))

y_nobin = np.array([19.61632079, 41.40185173])
x_nobin = np.array([6,7])
y_bin = np.array([7.192249767, 14.39824368, 13.05819078, 15.47645223])
x_bin = np.array([2,3,4,5])

x_smooth = np.linspace(x.min(), x.max(), 200)
y_smooth = spline(x, y, x_smooth)

y1 = np.array([7.778161546, 8.461481603, 7.505486858, 7.681106623, 8.898121926, 7.159081028, 8.935400266, 8.603094771, 10.46739098, 8.399149204, 9.866220439, 10.00512817, 12.74742177, 10.78683444, 13.66791164, 13.94559512])
x1 = np.array(range(2,len(y1)+2))
print(y1.size)
y1_nobin = []
x1_nobin = np.array([12,13,14,15,16,17])
for idx in x1_nobin:
	y1_nobin.append(y1[idx-2])
y1_bin = []
x1_bin = np.array([2,3,4,5,6,7,8,9,10,11])
for idx in x1_bin:
	y1_bin.append(y1[idx-2])

x1_smooth = np.linspace(x1.min(), x1.max(), 800)
y1_smooth = spline(x1, y1, x1_smooth)

y2 = np.array([14.82375307, 17.58301921, 8.835599047, 11.95660176, 6.984417692, 8.993459398, 6.0594408, 10.69682376, 6.934750442, 9.601049151, 6.272803935, 8.658359955, 6.179147811, 8.391750271, 7.874301322, 6.119591339, 12.17591362, 17.84112251, 11.56444865])
x2 = np.array(range(2,len(y2)+2))

indexes_nonbin = [2,3,4,5,7,9,11,13,15,18,19,20]
indexes_bin = [6,8,10,12,14,16,17]

y2_nobin = []
for idx in indexes_nonbin:
	y2_nobin.append(y2[idx-2])
x2_nobin = np.array(indexes_nonbin)
y2_bin = []
for idx in indexes_bin:
	y2_bin.append(y2[idx-2])
x2_bin = np.array(indexes_bin)

x2_smooth = np.linspace(x2.min(), x2.max(), 800)
y2_smooth = spline(x2, y2, x2_smooth)

plt.ylabel('Metric Score')
plt.xlabel('Layer number')

plt.ylim([0,45])

ax.plot(x,y,linestyle='--',label='Sketch-A-Net',color='lightgreen')
ax.plot(x1,y1,linestyle='--',label='Resnet18',color='mediumturquoise')
ax.plot(x2,y2,linestyle='--',label='SqueezeNet',color='salmon')
ax.plot([0.0],[-1],'*',color='black',markersize=9,label='Weight binarized')
ax.plot([0.0],[-1],'s',color='black',label='Full binarized')
# plot the data
ax.plot(x_nobin,y_nobin,'*',color='darkgreen',markersize=9)
ax.plot(x_bin,y_bin,'s',color='darkgreen')
ax.plot(x1_nobin,y1_nobin,'*',color='darkblue',markersize=9)
ax.plot(x1_bin,y1_bin,'s',color='darkblue')
ax.plot(x2_nobin,y2_nobin,'*',color='darkred',markersize=9)
ax.plot(x2_bin,y2_bin,'s',color='darkred')

ax.grid(linestyle='--')

#plt.ylim(45,75)
#plt.xlim(-5,112)

legend = ax.legend(loc='upper left')

matplotlib.rcParams.update({'font.size': 7})

plt.show()
