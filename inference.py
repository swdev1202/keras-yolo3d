from yolo3d_model import infer_image

weights = "lgsvl_yolo3d.h5"
image_path = "../dataset/val/1557021023820550.jpg"
output_path = "output/"

infer_image(weights, image_path, output_path, save_image=True)