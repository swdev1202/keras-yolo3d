from yolo3d_model import *
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
import keras.backend as K
import tensorflow as tf
import numpy as np
import os
import cv2
from preprocessing import parse_annotations, BatchGenerator, MyGenerator

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
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}

train_annot_file = "../dataset_front/annotations/train_ann.txt"
train_image_folder = "../dataset_front/train/"
valid_annot_file = "../dataset_front/annotations/val_ann.txt"
valid_image_folder = "../dataset_front/val/"

train_imgs, seen_train_labels = parse_annotations(train_annot_file, train_image_folder)
valid_imgs, seen_valid_labels = parse_annotations(valid_annot_file, valid_image_folder)

def normalize(image):
    return image / 255.

train_batch = MyGenerator(train_imgs, generator_config, "training_batch", norm=normalize, shuffle=False)
valid_batch = MyGenerator(valid_imgs, generator_config, "validation_batch", norm=normalize, shuffle=False)
# train_batch = BatchGenerator(train_imgs, generator_config, "training_batch", shuffle=False)
# valid_batch = BatchGenerator(valid_imgs, generator_config, "validation_batch", shuffle=False)


##################################################
## Callbacks
##################################################

early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint('lgsvl_yolo3d.h5', 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)

tb_counter  = len([log for log in os.listdir(os.path.expanduser('logs/')) if 'lgsvl_' in log]) + 1
tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/') + 'lgsvl_' + '_' + str(tb_counter), 
                          histogram_freq=0, 
                          write_graph=True, 
                          write_images=False)

#####################################################
## Compile model and start training
#####################################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)

K.set_session(sess)

model = create_yolo3d_model()
#optimizer = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
optimizer = SGD(lr=1e-6, decay=0.0005, momentum=0.9)
model.compile(loss=my_yolo3d_loss, optimizer=optimizer)
model.summary()

print([n.name for n in tf.get_default_graph().as_graph_def().node])

model.fit_generator(generator        = train_batch, 
                    steps_per_epoch  = len(train_batch), 
                    epochs           = 150,
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = [early_stop, checkpoint, tensorboard], 
                    max_queue_size   = 3)