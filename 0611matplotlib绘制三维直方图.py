# -*- coding: utf-8 -*-
# http://matplotlib.org/examples/mplot3d/hist3d_demo.html
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir(r'C:\Users\jk\Desktop\诸葛兼职\060811-200')

# 读取二维的数据，分别将二维的两个数据存储为两个指标
df = pd.read_excel(r'xy.xlsx')
x = df['minPrice'].values
y = df['maxPrice'].values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置每一个维度的坐标的最大、最小值，以及之间柱状图的个数，之间的间隔bins
hist, xedges, yedges = np.histogram2d(x, y, bins=14, range=[[16, 44], [17, 45]])
# print(hist)
# print(xedges)
# print(yedges)
# Construct arrays for the anchor positions of the 16 bars.
# Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
# # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
# # with indexing='ij'.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

# Construct arrays with the dimensions for the 16 bars.
dx = np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#0033FF', alpha=0.7, zsort='average')

ax.set_xlabel('minPrice')
ax.set_ylabel('maxPrice')
ax.set_zlabel('frequency')

plt.show()


# 读取时间处理的数据
df = pd.read_excel(r'wzdata.xls')
x = df['timeOfMinPrice'].values
y = df['timeOfMaxPrice'].values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置每一个维度的坐标的最大、最小值，以及之间柱状图的个数，之间的间隔bins
hist, xedges, yedges = np.histogram2d(x, y, bins=8, range=[[1, 9], [1, 9]])
print(hist)
print(xedges)
print(yedges)
# Construct arrays for the anchor positions of the 16 bars.
# Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
# # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
# # with indexing='ij'.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

# Construct arrays with the dimensions for the 16 bars.
dx = 0.3 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#0033FF', alpha=0.7, zsort='average')

ax.set_xlabel('timeOfMinPrice')
ax.set_ylabel('timeOfMaxPrice')
ax.set_zlabel('frequency')
# 设置x,y轴的标签为时间数据
labels = ['9:30','10:00','10:30','11:00','11:30/13:00','13:30','14:00','14:30','15:00']
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

plt.show()
