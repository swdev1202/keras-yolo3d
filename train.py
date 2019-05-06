from yolo3d_model import *
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
import os
import cv2


generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : 50,
}

def normalize(image):
    return image / 255.

model = create_yolo3d_model()
# optimizer = Adam(lr=0.1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)

optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
model.compile(loss=yolo3d_loss, optimizer=optimizer)
model.summary()

