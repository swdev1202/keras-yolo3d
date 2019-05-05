# keras-yolo3d
keras implementation of [YOLO3D](https://arxiv.org/pdf/1808.02350.pdf) based on YOLOv2 architecture.

## Introduction
This is a YOLO3D implementation in Keras using a TensorFlow Backend.

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
[MLBLR YoloV2](https://mlblr.com/includes/mlai/index.html#yolov2)
