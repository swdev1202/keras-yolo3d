import os
import cv2
import numpy as np
from keras.utils import Sequence
from utils import BoundBox, bbox_iou
from random import seed, randrange
from shutil import copy
import copy

def prune_gt_annotations(image_dir, input_gt, output_gt):
    count_after_pruning = 0
    list_images = sorted(os.listdir(image_dir))
    with open(input_gt, 'r') as f_in:
        for line in f_in:
            line_split = line.split(' ')
            image_name = line_split[0]
            image_name = image_name[:-4] + '.png'

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
                filename = filename[:-4] + '.png'
                copy(input_images_dir + filename, val_images_dir + filename)
                val_num += 1
            else: # send to train set
                with open(train_ann, 'a') as f_out:
                    f_out.write(line)
                filename = line.split(' ')[0]
                filename = filename[:-4] + '.png'
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
    labels_map = {1: "smallcar",
                  2: "bigcar",
                  3: "pedestrian",
                  4: "cyclist"}

    all_imgs = []
    seen_labels = {}

    with open(ann_file, 'r') as f_in:
        for line in f_in:
            
            img = {'object':[]}

            line = line.strip('\r\n')
            line_split = line.split(' ')

            filename = line_split[0]
            filename = filename[:-4] + '.png'
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
                obj_dict['yaw'] = yaw / np.pi
                obj_dict['z'] = z_pos / 2.0 # height range (-2.0, 0.0)
                obj_dict['height'] = z_size / 4.0 # height

                img['object'] += [obj_dict]

            all_imgs += [img]
    return all_imgs, seen_labels


# train_ann = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/annotations/train_ann.txt"
# train_images_dir = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Spring 2019/CMPE 297/datasets/lidar_bev_1/dataset_split/train/"

# all_imgs, seen_labels = parse_annotations(train_ann, train_images_dir)

class BatchGenerator(Sequence):
    def __init__(self, images, config, name, shuffle=True, jitter=True, norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.name = name

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1], 0, 0, 0) for i in range(int(len(config['ANCHORS'])//2))]

        if shuffle: np.random.shuffle(self.images)
    
    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)
    
    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])[:,:,0:2]
    
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)
    
    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], 
                        obj['ymin'], 
                        obj['xmax'], 
                        obj['ymax'],
                        obj['yaw'],
                        obj['z'],
                        obj['height'], 
                        self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)
    
    def __getitem__(self, idx):

        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']
        
        instance_count = 0

        # Input images
        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 2))

        # list of self.config[self.config['TRUE_BOX_BUFFER'] GT boxes
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'], 7)) #??

        # Desired network output
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'], self.config['GRID_W'], self.config['BOX'], 7+1+len(self.config['LABELS'])))
        
        #print("*** DEBUG ***")
        #print("[" + self.name + "]" + " idx: " + str(idx))
        #print("[" + self.name + "]" + "BATCH_SIZE: " + str(self.config['BATCH_SIZE']))
        #print("[" + self.name + "]" + "Bounds: " + str(l_bound) + ", " + str(r_bound))

        count = 0

        for train_instance in self.images[l_bound:r_bound]:
            image_name = train_instance['filename']
            count += 1

            #print("File " + str(count) + ": " + image_name)
            img = cv2.imread(image_name)[:,:,0:2]

            if img is None: print('Cannot find ', image_name)

            all_objs = copy.deepcopy(train_instance['object'])
            
            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']: # sanity check
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])
                    center_z = obj['z']
                    yaw = obj['yaw']
                    height = obj['height']

                    # grid location -> GRID(x,y)
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']: # sanity check
                        obj_indx  = self.config['LABELS'].index(obj['name']) # get the index of a given object(name)
                        
                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                        center_l = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
                        
                        box = [center_x, center_y, center_z, center_w, center_l, height, yaw]

                         # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou     = -1
                        
                        shifted_box = BoundBox(0, 
                                               0,
                                               0,
                                               center_w,                                                
                                               center_l,
                                               0,
                                               0)
                        
                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou    = bbox_iou(shifted_box, anchor)
                            
                            if max_iou < iou:
                                best_anchor = i
                                max_iou     = iou
                        
                        # assign ground truth x, y, z, yaw, w, l, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:7] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 7  ] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 8+obj_indx] = 1

                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
            
            # assign input image to x_batch
            if self.norm != None: 
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    print(image_name + " : " + obj['name'])
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        print(obj['xmax'], obj['xmin'], obj['ymax'], obj['ymin'])
                        cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                        cv2.putText(img[:,:,::-1], obj['name'], 
                                    (obj['xmin']+2, obj['ymin']+12),
                                    0, 1.2e-3 * img.shape[0], 
                                    (0,255,0), 2)
                        # debug code
                        # fin_img = np.concatenate((img, np.zeros([608,608,1], dtype=np.uint8)), axis=2)
                        # cv2.imwrite(str(obj['name'])+'.jpg', fin_img)
                        
                x_batch[instance_count] = img
            
            # increase instance counter in current batch
            instance_count += 1
            # print(instance_count)

        print("*************")
        print('new batch created', idx)
        return [x_batch, b_batch], y_batch


class MyGenerator(Sequence):
    def __init__(self, images, config, name, shuffle=True, norm=None):
        
        self.images = images
        self.config = config
        self.shuffle = shuffle
        self.name = name
        self.norm = norm
        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1], 0, 0, 0) for i in range(int(len(config['ANCHORS'])//2))]
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))
    
    def num_classes(self):
        return len(self.config['LABELS'])
    
    def size(self):
        return len(self.images)
    
    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])[:,:,0:2]
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.images))
        if self.shuffle: np.random.shuffle(self.images)
    
    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], 
                        obj['ymin'], 
                        obj['xmax'], 
                        obj['ymax'],
                        obj['yaw'],
                        obj['z'],
                        obj['height'], 
                        self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)
    
    def __getitem__(self, idx):

        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        indices = self.indices[l_bound:r_bound]
        images_batch = [self.images[k] for k in indices]

        instance_count = 0

        # Input images
        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 2))

        # list of self.config[self.config['TRUE_BOX_BUFFER'] GT boxes
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'], 7)) #??

        # Desired network output
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'], self.config['GRID_W'], self.config['BOX'], 7+1+len(self.config['LABELS'])))
        
        #print("\n*** DEBUG ***")
        #print("[" + self.name + "]" + " idx: " + str(idx))
        #print("[" + self.name + "]" + "BATCH_SIZE: " + str(self.config['BATCH_SIZE']))
        #print("[" + self.name + "]" + "Bounds: " + str(l_bound) + ", " + str(r_bound))

        count = 0

        for train_instance in images_batch:
            count += 1
            image_name = train_instance['filename']
            # print("File " + str(count) + ": " + image_name)

            img = cv2.imread(image_name)[:,:,0:2]
            if img is None: print('Cannot find ', image_name)
            
            all_objs = copy.deepcopy(train_instance['object'])
            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']: # sanity check
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])
                    center_z = obj['z']
                    yaw = obj['yaw']
                    height = obj['height']

                    # grid location -> GRID(x,y)
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']: # sanity check
                        obj_indx  = self.config['LABELS'].index(obj['name']) # get the index of a given object(name)
                        
                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                        center_l = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
                        
                        box = [center_x, center_y, center_z, center_w, center_l, height, yaw]

                         # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou     = -1
                        
                        shifted_box = BoundBox(0, 
                                               0,
                                               0,
                                               center_w,                                                
                                               center_l,
                                               0,
                                               0)
                        
                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou    = bbox_iou(shifted_box, anchor)
                            
                            if max_iou < iou:
                                best_anchor = i
                                max_iou     = iou
                        
                        # assign ground truth x, y, z, yaw, w, l, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:7] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 7  ] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 8+obj_indx] = 1

                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
            
            # assign input image to x_batch
            if self.norm != None: 
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    print(image_name + " : " + obj['name'])
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        print(obj['xmax'], obj['xmin'], obj['ymax'], obj['ymin'])
                        cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                        cv2.putText(img[:,:,::-1], obj['name'], 
                                    (obj['xmin']+2, obj['ymin']+12),
                                    0, 1.2e-3 * img.shape[0], 
                                    (0,255,0), 2)
                        # debug code
                        # fin_img = np.concatenate((img, np.zeros([608,608,1], dtype=np.uint8)), axis=2)
                        # cv2.imwrite(str(obj['name'])+'.jpg', fin_img)
                        
                x_batch[instance_count] = img
            
            # increase instance counter in current batch
            instance_count += 1
            # print(instance_count)

        #print("*************")
        #print('new batch created', idx)

        assert not np.any(np.isnan(x_batch))
        assert not np.any(np.isnan(y_batch))
        return [x_batch, b_batch], y_batch
