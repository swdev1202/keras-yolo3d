from yolo3d_model import infer_image

weights = "../apollo_yolo3d_1.h5"
# image_path = "../dataset/val/1557021023820550.jpg"
image_path = "../dataset_front/val/result_9055_6_192.png"
output_path = "output/"

infer_image(weights, image_path, output_path, save_image=True)