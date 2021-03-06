import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
def point_cloud_2_top(points,
                      res=0.1,
                      zres=0.3,
                      side_range=(-10., 10.),  # left-most to right-most
                      fwd_range=(-10., 10.),  # back-most to forward-most
                      height_range=(-2., 2.),  # bottom-most to upper-most
                      ):
    """ Creates an birds eye view representation of the point cloud data for MV3D.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        zres:        (float)
                    Desired resolution on Z-axis in metres to use.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        numpy array encoding height features , density and intensity.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    # Assuming PCD file has (X, Y, Z) = (Y, -X, Z)
    x_points = points[:, 1]
    y_points = -points[:, 0]
    z_points = points[:, 2]

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    z_max = int((height_range[1] - height_range[0]) / zres)
    top = np.zeros([y_max+1, x_max+1, z_max], dtype=np.float32)

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and(
        (x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and(
        (y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)


    # # ASSIGN EACH POINT TO A HEIGHT SLICE
    # # n_slices-1 is used because values above max_height get assigned to an
    # # extra index when we call np.digitize().
    # bins = np.linspace(height_range[0], height_range[1], num=n_slices-1)
    # slice_indices = np.digitize(z_points, bins=bins, right=False)
    # # RESCALE THE REFLECTANCE VALUES - to be between the range 0-255
    # pixel_values = scale_to_255(r_points, min=0.0, max=1.0)
    # FILL PIXEL VALUES IN IMAGE ARRAY
    # -y is used because images start from top left
    # x_max = int((side_range[1] - side_range[0]) / res)
    # y_max = int((fwd_range[1] - fwd_range[0]) / res)
    # im = np.zeros([y_max, x_max, n_slices], dtype=np.uint8)
    # im[-y_img, x_img, slice_indices] = pixel_values
    for i, height in enumerate(np.arange(height_range[0], height_range[1], zres)):
        z_filt = np.logical_and((z_points >= height),
                                (z_points < height + zres))
        zfilter = np.logical_and(filter, z_filt)
        indices = np.argwhere(zfilter).flatten()

        # KEEPERS
        xi_points = x_points[indices]
        yi_points = y_points[indices]
        zi_points = z_points[indices]

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-yi_points / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-xi_points / res).astype(np.int32)  # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor & ceil used to prevent anything being rounded to below 0 after shift
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.floor(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = zi_points - height_range[0]

        # FILL PIXEL VALUES IN IMAGE ARRAY
        pixel_values = np.floor((zi_points-height_range[0]) * (225/2)).astype(np.int8)
        top[y_img-1, x_img-1, i] = pixel_values

    return top

def ascii_pcd_to_np(file):
    data_start_line = np.inf
    all_points_string = ""
    with open(file, 'r') as pcd_file:
        for i, line in enumerate(pcd_file):
            if line.startswith('DATA'):
                data_start_line = i + 1
            if i >= data_start_line:
                all_points_string += line
    scan = np.fromstring(all_points_string, dtype=np.float32, sep=' ')
    scan = np.reshape(scan, (-1, 5))
    return scan
################################################################################################

side_range = (-7.6, 7.6)
fwd_range = (0.0, 15.2)
height_range = (-2.0, 0.0)
resolution = 0.025
z_resolution = 0.03125

X_DIM = int((side_range[1]-side_range[0])/resolution)+1
Y_DIM = int((fwd_range[1]-fwd_range[0])/resolution)+1

root_dir = "."
name = "1556686293933383.pcd"
filename = os.path.join(root_dir, name)
# scan = np.fromfile(filename, dtype=np.float32, sep=" ") ## .pcd (ASCII)
scan = ascii_pcd_to_np(filename)
bird_view = point_cloud_2_top(scan, res=resolution, zres=z_resolution,
                                side_range=side_range,  # left-most to right-most
                                fwd_range=fwd_range,  # back-most to forward-most
                                height_range=height_range)
print("Bird View Dimension = " + str(bird_view.shape))


##################################################################################
# height map generation
##################################################################################
max_height_map = np.zeros([Y_DIM,X_DIM], dtype=np.uint8)
max_height_map = np.amax(bird_view, axis=2)
plt.imshow(max_height_map, cmap='gray')
plt.savefig('height_map_plt.png')
cv2.imwrite('height_map_cv2.jpg', max_height_map)

##################################################################################
# density map generation
##################################################################################
density_map = np.zeros([Y_DIM,X_DIM], dtype=np.uint8)
# we will need to change this (instead of iterating all)
for i in range(Y_DIM):
    for j in range(X_DIM):
        density_map[i][j] = np.count_nonzero(bird_view[i,j,:])
density_map = np.fmin(1.0, np.log(1+density_map)/np.log(64))
#rescale denstiy map
density_map = np.floor(density_map * (255)).astype(np.uint8)
plt.imshow(density_map, cmap='gray')
plt.savefig('density_map_plt.png')
cv2.imwrite('density_map_cv2.jpg', density_map)

##################################################################################
# 2 channel concatenated image
##################################################################################
two_channel_map = np.concatenate((max_height_map.reshape(Y_DIM,X_DIM,1), density_map.reshape(Y_DIM,X_DIM,1)), axis=2)
np.save("result", two_channel_map)

two_channel_RGB_map = np.concatenate((two_channel_map, np.zeros([Y_DIM,X_DIM,1], dtype=np.uint8)), axis=2)
plt.imshow(two_channel_RGB_map)
plt.savefig('two_channel_plt.png')
cv2.imwrite('two_channel_cv2.png', two_channel_RGB_map)

plt.show()