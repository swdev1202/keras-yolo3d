from yolo3d_model import infer_image

weights = "lgsvl_yolo3d.h5"
# image_path = "../dataset/val/1557021023820550.jpg"
image_path = "../dataset/train/1556688656567380.jpg"
output_path = "output/"

infer_image(weights, image_path, output_path, save_image=True)