from yolo3d_model import *
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
import keras.backend as K
import tensorflow as tf
import numpy as np
import os
import cv2

###############################################
## Parse data and create batches
###############################################

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

train_annot_file = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/annotations/train_ann.txt"
train_image_folder = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/train/"
valid_annot_file = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/annotations/val_ann.txt"
valid_image_folder = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/val/"

train_imgs, seen_train_labels = parse_annotation(train_annot_file, train_image_folder)
valid_imgs, seen_valid_labels = parse_annotation(valid_annot_file, valid_image_folder)

train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize)
valid_batch = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)

##################################################
## Callbacks
##################################################
def normalize(image):
    return image / 255.

early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint('weights_coco.h5', 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)

tb_counter  = len([log for log in os.listdir(os.path.expanduser('~/logs/')) if 'lgsvl_' in log]) + 1
tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/') + 'lgsvl_' + '_' + str(tb_counter), 
                          histogram_freq=0, 
                          write_graph=True, 
                          write_images=False)

#####################################################
## Compile model and start training
#####################################################

model = create_yolo3d_model()
optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
model.compile(loss=yolo3d_loss, optimizer=optimizer)
model.summary()

model.fit_generator(generator        = train_batch, 
                    steps_per_epoch  = len(train_batch), 
                    epochs           = 100, 
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = [early_stop, checkpoint, tensorboard], 
                    max_queue_size   = 3)
