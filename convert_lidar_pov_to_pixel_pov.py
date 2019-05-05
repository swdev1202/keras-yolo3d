import numpy as np
import os

def get_new_x(y_hat):
    return 608 - (y_hat+304)

def get_new_y(x_hat):
    return 608 - x_hat

def get_min_max_x(x, w):
    return int(x-(w/2)), int(x+(w/2))

def get_min_max_y(y, l):
    return int(y-(l/2)), int(y+(l/2))

filepath = 'annotations'
filename = 'gt_yolo3d_annotations_2.txt'

file_to_write = 'gt_yolo3d_annotations_converted_2.txt'
given_file_to_write = os.path.join(filepath, file_to_write)

file_w = open(given_file_to_write, "w")

given_annotation = os.path.join(filepath, filename)
# open and read file
f = open(given_annotation, "r")
for line in f:
    # first split the lines with " " ==> [filename, object1, object2, ...]
    split_line = line.split(" ")
    split_line = split_line[:len(split_line)-1]

    complete_line = ""
    for inner_line in split_line:
        if(inner_line[len(inner_line)-3:len(inner_line)] == 'pcd'):
            complete_line += inner_line
        else:
           split_inner_line = inner_line.split(",")
           _x = int(split_inner_line[0])
           _y = int(split_inner_line[1])

           new_y = get_new_y(_x)
           new_x = get_new_x(_y)

           complete_line += " " + str(new_x) + "," + str(new_y) + ","
           complete_line += split_inner_line[2] + ","
           complete_line += split_inner_line[3] + ","
           complete_line += split_inner_line[4] + ","
           complete_line += split_inner_line[5] + ","
           complete_line += split_inner_line[6] + ","
           complete_line += split_inner_line[7]
    print(complete_line)
    file_w.write(complete_line + "\n")
        
'''
    for inner_line in split_line:
        if('.pcd' not in inner_line):
            split_inner_line = inner_line.split(",")
            # now split_inner_line has each object's information
            _object = split_inner_line[-1]
            _width = split_inner_line[4]
            _length = split_inner_line[5]

            _info = _object + "," + _width + "," + _length
            DIMENSIONS.append(_info)
            '''