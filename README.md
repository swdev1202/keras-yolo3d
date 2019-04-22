# keras-yolo3d
keras implementation of [YOLO3D](https://arxiv.org/pdf/1808.02350.pdf)

## Introduction
This is a YOLO3D implementation.

## How to convert
kitti2birdeye.py will convert KITTI's .bin format point cloud to bird-eye-view 2D image
```
python kitti2birdeye.py [inputfilename] [desired output format]
```

For instance,
```
python kitti2birdeye.py 000000.bin jpg
```
will produce 000000.jpg  

The output format can be .bmp, .jpg, .png (whatever pillow supports).

## References
[Ronny Restrepo's Birdeye View](http://ronny.rest/tutorials/module/pointclouds_01/point_cloud_birdseye/)  
[Alex Staravoitau's Visualizing Lidar Data](https://navoshta.com/kitti-lidar/)
