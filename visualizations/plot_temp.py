import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.rcParams.update({'font.size': 14})

fig,ax = plt.subplots(1, figsize=(8,6))

# create some x data and some integers for the y axis
#x = np.array([0,5,6,7])
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([30,40,42,48,44,50,58,54,65,70])

# plot the data
ax.plot(x,y,linestyle='--', marker='s',mew=10,linewidth=4)

plt.show()
