import csv
import numpy as np

input_file = "annotations/gt_3d_annotations.csv"
output_file = "annotations/gt_yolo3d_annotations.txt"

fwd_range = [0, 15.2]
side_range = [-7.6, 7.6]

x_res = 0.025 # m/px
y_res = 0.025 # m/px

# Input 3D Ground Truth format
# <filename> <x_pos, y_pos, z_pos, x_ori, y_ori, z_ori, w_ori, x_size, y_size, z_size, label>

# Output 3D Ground Truth format
# <filename> <x_pos, y_pos, z_pos, yaw, x_size, y_size, z_size, label>

def get_yaw_angle(x_ori, y_ori, z_ori, w_ori):
    siny_cosp = 2 * ((w_ori * z_ori) + (x_ori * y_ori))
    cosy_cosp = 1 - 2 * ((y_ori * y_ori) + (z_ori * z_ori))
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def convert_3d_annotations(input_file, output_file):
    converted_points = []
    with open(input_file, 'r') as f_in:
        for line in f_in:
            line = line.strip('\n')
            line = line.split(';')
            pcd_name = line[0]
            pcd_bboxes = line[1]
            pcd_bboxes = pcd_bboxes.split(' ')
            line_out = [pcd_name]
            bboxes_str = ""
            for bbox in pcd_bboxes:
                np_bbox = bbox.split(',')
                if not len(np_bbox) == 11:
                    # if not all points available, skip
                    continue
                x_pos = np_bbox[0]
                y_pos = np_bbox[1]
                z_pos = np_bbox[2]
                x_ori = np_bbox[3]
                y_ori = np_bbox[4]
                z_ori = np_bbox[5]
                w_ori = np_bbox[6]
                x_size = np_bbox[7]
                y_size = np_bbox[8]
                z_size = np_bbox[9]
                label = np_bbox[10]

                if not ((float(x_pos) < fwd_range[0]) or (float(x_pos) > fwd_range[1]) or (float(y_pos) < side_range[0]) or (float(y_pos) > side_range[1])):
                    # If not out of range
                    yaw = get_yaw_angle(float(x_ori), float(y_ori), float(z_ori), float(w_ori))
                    bboxes_str += str(int(float(x_pos)/x_res)) + "," + str(int(float(y_pos)/y_res)) + "," + str(z_pos) + "," + str(yaw) + "," + str(int(float(x_size)/x_res)) + "," + str(int(float(y_size)/y_res)) + "," + str(z_size) + ","
                    bboxes_str += str(label) + " "
                
            if bboxes_str:
                line_out.append(bboxes_str)
                converted_points.append(line_out)

        # Write to output file line by line
        with open(output_file, 'a') as f_out:
            for line in converted_points:
                f_out.write(line[0] + ' ' + line[1] + '\n')

convert_3d_annotations(input_file, output_file)      