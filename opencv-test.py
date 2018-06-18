#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/6/18 23:03'
__author__ = 'ooo'

import os
import cv2
import scipy
import skimage
import numpy as np

dir = 'G://MSCOCO2017//val2017'
name = '000000079651.jpg'
path = os.path.join(dir, name)

print('文件存在吗？', os.path.isfile(path), os.access(path, os.R_OK))

# img = skimage.io.imread(path)
# if img.ndim != 3:
#     image = skimage.color.gray2rgb(img)

img = cv2.imread(path)
cv2.imshow("img", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

edges = cv2.Canny(gray, 100, 200)
rows, clos = np.where(edges)
cv2.imshow("edges", edges)

cv2.waitKey(-1)
