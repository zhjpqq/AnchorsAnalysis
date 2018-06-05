#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/7 21:22'
__author__ = 'ooo'

import sys
import os
import math
import random
import numpy as np
import scipy.misc
import skimage.color
import skimage.io
from PIL import Image
import cv2
import urllib.request
import shutil
import logging

import torch
from torch.utils.data import Dataset

from Configs.config import Config
from DataSets.imdb import IMDB
from Utils import utils


class VOC2007(IMDB, Dataset):

    def load_voc(self, data_dir, year, subset, class_ids=None, class_map=None, auto_download=None, **kwargs):
        pass


