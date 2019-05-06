import os
import cv2
import numpy as np
from keras.utils import Sequence
from utils import BoundBox, bbox_iou
from random import seed, randrange
from shutil import copy

def prune_gt_annotations(image_dir, input_gt, output_gt):
    count_after_pruning = 0
    list_images = sorted(os.listdir(image_dir))
    with open(input_gt, 'r') as f_in:
        for line in f_in:
            line_split = line.split(' ')
            image_name = line_split[0]
            image_name = image_name[:-4] + '.jpg'

            if image_name in list_images:
                with open(output_gt, 'a') as f_out:
                    f_out.write(line)
                count_after_pruning += 1
    print("Count after pruning: " + str(count_after_pruning))


# image_dir = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/raw/images/combined_bev/"
# input_gt = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/raw/annotations/combined_gt.txt"
# output_gt = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/raw/annotations/combined_gt_pruned.txt"

# prune_gt_annotations(image_dir, input_gt, output_gt)

def split_test_training(input_images_dir, 
                        input_ann, 
                        train_images_dir, 
                        train_ann, 
                        val_images_dir, 
                        val_ann, 
                        split=0.1):

    list_images = sorted(os.listdir(input_images_dir))
    val_len = int(split * len(list_images))
    val_indices = []

    seed(101) # For repeatability

    while len(val_indices) < val_len:
        idx = randrange(0, len(list_images)-1)
        if idx not in val_indices:
            val_indices.append(idx)
    
    train_num = 0
    val_num = 0

    row_num = 0
    with open(input_ann, 'r') as f_in:
        for line in f_in:
            if row_num in val_indices: # send to validation set
                with open(val_ann, 'a') as f_out:
                    f_out.write(line)
                filename = line.split(' ')[0]
                filename = filename[:-4] + '.jpg'
                copy(input_images_dir + filename, val_images_dir + filename)
                val_num += 1
            else: # send to train set
                with open(train_ann, 'a') as f_out:
                    f_out.write(line)
                filename = line.split(' ')[0]
                filename = filename[:-4] + '.jpg'
                copy(input_images_dir + filename, train_images_dir + filename)
                train_num += 1
            row_num += 1
    
    print("Training images: " + str(train_num))
    print("Validation images: " + str(val_num))
    print("Total images: " + str(row_num))

# input_images_dir = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/raw/images/combined_bev/"
# input_ann = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/raw/annotations/combined_gt_pruned.txt"
# train_images_dir = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/train/"
# train_ann = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/annotations/train_ann.txt"
# val_images_dir = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/val/"
# val_ann = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/annotations/val_ann.txt"

# split_test_training(input_images_dir, input_ann, train_images_dir, train_ann, val_images_dir, val_ann, split=0.1)

def parse_annotations(ann_file, img_dir):
    labels_map = {0: "scooter",
                  1: "hoverboard",
                  2: "skateboard",
                  3: "segway",
                  4: "onewheel"}

    all_imgs = []
    seen_labels = {}

    with open(ann_file, 'r') as f_in:
        for line in f_in:
            
            img = {'object':[]}

            line = line.strip('\r\n')
            line_split = line.split(' ')

            filename = line_split[0]
            filename = filename[:-4] + '.jpg'
            filepath = img_dir + filename

            img['filename'] = filepath
            img['width'] = 608
            img['height'] = 608
            
            objects = line_split[1:]
            
            for obj in objects:
                obj_dict = {}
                obj_split = obj.split(',')

                x_pos = int(obj_split[0])
                y_pos = int(obj_split[1])
                z_pos = float(obj_split[2])
                yaw = float(obj_split[3])
                x_size = int(obj_split[4])
                y_size = int(obj_split[5])
                z_size = float(obj_split[6])
                label = labels_map.get(int(obj_split[7]))

                if label in seen_labels:
                    seen_labels[label] += 1
                else:
                    seen_labels[label] = 1

                obj_dict['name'] = label
                obj_dict['xmin'] = int(x_pos - x_size//2)
                obj_dict['xmax'] = int(x_pos + x_size//2)
                obj_dict['ymin'] = int(y_pos - y_size//2)
                obj_dict['ymax'] = int(y_pos + y_size//2)
                obj_dict['yaw'] = yaw
                obj_dict['z'] = z_pos
                obj_dict['height'] = z_size

                img['object'] += [obj_dict]

    all_imgs += [img]
    return all_imgs, seen_labels


# train_ann = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/annotations/train_ann.txt"
# train_images_dir = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/train/"

# all_imgs, seen_labels = parse_annotations(train_ann, train_images_dir)