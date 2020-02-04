import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

currentViewImage = 0

# Change Image Index on scroll
def on_press(event):
    global currentViewImage
    if(event.button == "up"):
        currentViewImage += 2
    if(event.button == "down"):
        currentViewImage -= 2
    currentViewImage = np.clip(currentViewImage, 0, 10000)
    # print('you pressed', event.button, event.xdata, event.ydata, currentViewImage)

# Update to current image
def animate(i):
    img = mnistTest.iloc[currentViewImage][:].values[1:].reshape(28,28)
    shownDigit1.set_data(img[1:])
    img = reducedmnistTest[currentViewImage].reshape(28,28)
    shownDigit2.set_data(img[1:])
    img = reducedmnistTest2[currentViewImage].reshape(28,28)
    shownDigit3.set_data(img[1:])
    img = reducedmnistTest3[currentViewImage].reshape(28,28)
    shownDigit4.set_data(img[1:])
    img = mnistTest.iloc[currentViewImage + 1][:].values[1:].reshape(28,28)
    shownDigit5.set_data(img[1:])
    img = reducedmnistTest[currentViewImage + 1].reshape(28,28)
    shownDigit6.set_data(img[1:])
    img = reducedmnistTest2[currentViewImage + 1].reshape(28,28)
    shownDigit7.set_data(img[1:])
    img = reducedmnistTest3[currentViewImage + 1].reshape(28,28)
    shownDigit8.set_data(img[1:])
    return

# Import test and training
mnistTraining = pd.read_csv('data/mnist_train.csv', header=None)
mnistTest     = pd.read_csv('data/mnist_test.csv',  header=None)

# Parse data from data frame
lablesTraining = mnistTraining.iloc[:][0]
dataTraining   = mnistTraining.drop(columns=0)
lablesTest     = mnistTest.iloc[:][0]
dataTest       = mnistTest.drop(columns=0)

# Explained variance ratios
# This is from the test data
#  25 : 70.189
#  33 : 51.839
#  50 : 83.160
# 100 : 91.804
# 200 : 96.786
# 300 : 98.714
# 324 : 99.002
# 400 : 99.630

# Show ~99%
pca = PCA(n_components=324)
pca.fit(dataTraining)
# PCAX = pca.transform(data)
PCAX = pca.transform(dataTest)
reducedmnistTest = pca.inverse_transform(PCAX)
explained = np.cumsum(pca.explained_variance_ratio_)
print(math.floor(explained[len(explained)-1]*10000)/100)

# Show ~75%
pca = PCA(n_components=33)
pca.fit(dataTraining)
# PCAX = pca.transform(data)
PCAX = pca.transform(dataTest)
reducedmnistTest2 = pca.inverse_transform(PCAX)
explained = np.cumsum(pca.explained_variance_ratio_)
print(math.floor(explained[len(explained)-1]*10000)/100)

# Show ~50%
pca = PCA(n_components=11)
pca.fit(dataTraining)
# PCAX = pca.transform(data)
PCAX = pca.transform(dataTest)
reducedmnistTest3 = pca.inverse_transform(PCAX)
explained = np.cumsum(pca.explained_variance_ratio_)
print(math.floor(explained[len(explained)-1]*10000)/100)


# Create subplots to show multi images
fig = plt.figure()
ax1 = fig.add_subplot(241)
ax1.axis('off')
ax2 = fig.add_subplot(242)
ax2.axis('off')
ax3 = fig.add_subplot(243)
ax3.axis('off')
ax4 = fig.add_subplot(244)
ax4.axis('off')
ax5 = fig.add_subplot(245)
ax5.axis('off')
ax6 = fig.add_subplot(246)
ax6.axis('off')
ax7 = fig.add_subplot(247)
ax7.axis('off')
ax8 = fig.add_subplot(248)
ax8.axis('off')

# Show initial image
img = reducedmnistTest[0].reshape(28,28)
shownDigit1 = ax1.imshow(img[1:])
shownDigit2 = ax2.imshow(img[1:])
shownDigit3 = ax3.imshow(img[1:])
shownDigit4 = ax4.imshow(img[1:])
shownDigit5 = ax5.imshow(img[1:])
shownDigit6 = ax6.imshow(img[1:])
shownDigit7 = ax7.imshow(img[1:])
shownDigit8 = ax8.imshow(img[1:])

# Add some suff
cid = fig.canvas.mpl_connect('scroll_event', on_press)
anim = animation.FuncAnimation(fig, animate)
plt.show()
