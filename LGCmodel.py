# import tensorflow.contrib.slim as slim
# import scipy.misc
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import shutil
# import utils
import os
import cv2
from PIL import Image
# from data import Bicubic
# import numpy as np
# import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
class LGC(tf.keras.Model):
    def __init__(self):
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        self.conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        self.conv5 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        self.conv6 = tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
        self.conv7 = tf.keras.layers.Conv2D(filters=3, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x5)
        x6 = self.conv(x3+x4+x5)
        y = self.conv(x6)
        output = y+inputs
        return output