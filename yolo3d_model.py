from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
import os
import cv2

#################################################
## Setup
#################################################

# Setup GPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Setup parameters
LABELS = ['scooter', 'hoverboard', 'skateboard', 'segway', 'onewheel']

IMAGE_H, IMAGE_W = 608, 608
GRID_H,  GRID_W  = 38 , 38 #??
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.3 #0.5
NMS_THRESHOLD    = 0.3 #0.45
# ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHORS          = np.multiply([39, 52, 18, 27, 27, 52, 29, 29, 24, 32], (GRID_H/IMAGE_H)).tolist()
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0
YAW_SCALE        = 1.0

BATCH_SIZE       = 4
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 20

#################################################
## Construct Network
#################################################
# def space_to_depth_x2(x):
#     return tf.space_to_depth(x, block_size=2)

input_image = Input(shape=(IMAGE_H, IMAGE_W, 2)) # Originally 3
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 7)) # Originally 4

def create_yolo3d_model():
    # Layer 1
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 2
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4 - (3,3)? // written as (1,1)?
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7 - (3,3)? // written as (1,1)?
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # skip_connection = x # Skipped in YOLO3D

    # x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Removed in YOLO3D
    # skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
    # skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    # skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    # skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    # x = concatenate([skip_connection, x])
    # x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(x)
    # x = BatchNormalization(name='norm_21')(x)
    # x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 22
    # x = Conv2D(BOX * (7 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_22')(x)
    # x = Conv2D(1024, (1,1), name='conv_22', padding='same')(x)
    x = Conv2D(BOX * (7 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)

    # output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)
    output = Reshape((GRID_H, GRID_W, BOX, 7 + 1 + CLASS))(x)  # 8 regressed terms per BOX
    output = Lambda(lambda args: args[0])([output, true_boxes])

    model = Model([input_image, true_boxes], output)

    return model

def yolo3d_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4] # Shape matches the 7 terms
    
    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), tf.float32)
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])

    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)

    print("========================================================")
    """
    Adjust prediction
    """
    ### adjust x, y and z      
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    pred_box_z = tf.sigmoid(y_pred[..., 2])

    ### adjust w, l and h
    pred_box_wl = tf.exp(y_pred[..., 3:5]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
    pred_box_h = tf.exp(y_pred[..., 5])

    ### adjust yaw
    pred_box_yaw = tf.sigmoid(y_pred[..., 6])

    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 7])
    
    ### adjust class probabilities
    pred_box_class = y_pred[..., 8:]

    """
    Adjust ground truth
    """
    ### adjust x, y and z
    true_box_xy = y_true[..., 0:2] # relative position to the containing cell
    true_box_z = y_true[..., 2]
    
    ### adjust w, l and h
    true_box_wl = y_true[..., 3:5] # number of cells accross, horizontally and vertically
    true_box_h = y_true[..., 5]

    ### adjust yaw
    true_box_yaw = y_true[..., 6]

    ### adjust confidence(IOU)
    true_wl_half = true_box_wl / 2.
    true_mins    = true_box_xy - true_wl_half
    true_maxes   = true_box_xy + true_wl_half

    pred_wl_half = pred_box_wl / 2.
    pred_mins    = pred_box_xy - pred_wl_half
    pred_maxes   = pred_box_xy + pred_wl_half

    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wl    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wl[..., 0] * intersect_wl[..., 1] # overlapped area

    true_areas = true_box_wl[..., 0] * true_box_wl[..., 1]
    pred_areas = pred_box_wl[..., 0] * pred_box_wl[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    true_box_conf = iou_scores * y_true[..., 7] # 7th index is conf in YOLO3D
    
    ### adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 8:], -1) # 8:13th indices in YOLO3D are class probs

    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 7], axis=-1) * COORD_SCALE

    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_z = true_boxes[..., 2]

    true_wl = true_boxes[..., 3:5]
    true_h = true_boxes[..., 5]

    true_yaw = true_boxes[..., 6]

    true_wl_half = true_wl / 2.
    true_mins    = true_xy - true_wl_half
    true_maxes   = true_xy + true_wl_half

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wl = tf.expand_dims(pred_box_wl, 4)

    pred_wl_half = pred_wl / 2.
    pred_mins    = pred_xy - pred_wl_half
    pred_maxes   = pred_xy + pred_wl_half

    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wl    = tf.maximum(intersect_maxes - intersect_mins, 0.) 
    intersect_areas = intersect_wl[..., 0] * intersect_wl[..., 1]

    true_areas = true_wl[..., 0] * true_wl[..., 1]
    pred_areas = pred_wl[..., 0] * pred_wl[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.cast(best_ious < 0.6, tf.float32) * (1 - y_true[..., 7]) * NO_OBJECT_SCALE

    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 7] * OBJECT_SCALE

    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 7] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE

    """
    Warm-up training
    """
    # no_boxes_mask = tf.cast(coord_mask < COORD_SCALE/2., tf.float32)
    # seen = tf.assign_add(seen, 1.)
    
    # true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
    #                       lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
    #                                true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1,1,1,BOX,2]) * no_boxes_mask, 
    #                                tf.ones_like(coord_mask)],
    #                       lambda: [true_box_xy, 
    #                                true_box_wh,
    #                                coord_mask])

    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, tf.float32))
    nb_conf_box  = tf.reduce_sum(tf.cast(conf_mask  > 0.0, tf.float32))
    nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, tf.float32))

    # Add loss_z term
    loss_xy = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_z = tf.reduce_sum(tf.square(true_box_z-pred_box_z) * coord_mask) / (nb_coord_box + 1e-6) / 2.

    # Need to change to wlh
    loss_wl = tf.reduce_sum(tf.square(true_box_wl-pred_box_wl) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_h = tf.reduce_sum(tf.square(true_box_h-pred_box_h) * coord_mask) / (nb_coord_box + 1e-6) / 2.

    #loss_yaw = YAW_SCALE * tf.reduce_sum(tf.square(true_box_yaw - pred_box_yaw) * coord_mask) / (nb_coord_box + 1e-6) / 2.

    #loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2. #problematic? lambda_1_loss/mul_18
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    # loss = loss_xy + loss_z + loss_wl + loss_h + loss_yaw + loss_conf + loss_class
    loss = loss_xy + loss_z + loss_wl + loss_h + loss_class

    nb_true_box = tf.reduce_sum(y_true[..., 7])
    nb_pred_box = tf.reduce_sum(tf.cast(true_box_conf > 0.5, tf.float32) * tf.cast(pred_box_conf > 0.3, tf.float32))

    """
    Debugging code
    """    
    current_recall = nb_pred_box/(nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall) 

    # loss = tf.print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    # loss = tf.print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    # loss = tf.print(loss, [loss_z], message='Loss Z \t', summarize=1000)
    # loss = tf.print(loss, [loss_wl], message='Loss WL \t', summarize=1000)
    # loss = tf.print(loss, [loss_h], message='Loss H \t', summarize=1000)
    # loss = tf.print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    # loss = tf.print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    # loss = tf.print(loss, [loss], message='Total Loss \t', summarize=1000)
    # loss = tf.print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    # loss = tf.print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
    
    return loss
