import sys, getopt, os
import numpy as np
from PIL import Image

def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))

def get_velo_points(velo):
    return velo[:,0], velo[:,1], velo[:,2]

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def main(arg1, arg2):

    inputfile = arg1
    outputformat = '.' + arg2
    print("input file = ", inputfile)
    print("output format = ", outputformat)
    
    #getting only the file name
    filename = os.path.splitext(inputfile)[0]

    # first, read kitti's point cloud formats
    velo_data = load_velo_scan(inputfile)

    # YOLO3D requirements
    side_range=(-30.4, 30.4)  # left-most to right-most
    fwd_range=(0, 60.8)       # back-most to forward-most

    # get x,y,z points
    x_points, y_points, z_points = get_velo_points(velo_data)

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS (only desired points left)
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    res = 0.1 # 0.1m/px resoultion
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor and ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # YOLO3D requirement
    height_range = (-2, 2)  # bottom-most to upper-most

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a = z_points,
                            a_min=height_range[0],
                            a_max=height_range[1])

                            
    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values  = scale_to_255(pixel_values, min=height_range[0], max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1+int((side_range[1] - side_range[0])/res)
    y_max = 1+int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img-1, x_img-1] = pixel_values

    # CONVERT FROM NUMPY ARRAY TO A PIL IMAGE
    outImage = Image.fromarray(im)
    outImage.save(filename+outputformat)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])