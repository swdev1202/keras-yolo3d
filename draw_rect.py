import numpy as np
import cv2

img = cv2.imread('height_map_cv2.jpg', cv2.IMREAD_COLOR)

def get_new_x(y_hat):
    return 608 - (y_hat+304)

def get_new_y(x_hat):
    return 608 - x_hat

def get_min_max_x(x, w):
    return int(x-(w/2)), int(x+(w/2))

def get_min_max_y(y, l):
    return int(y-(l/2)), int(y+(l/2))

X_HAT = 549
Y_HAT = -208
WIDTH = 18
LENGTH = 27

X_PIXEL_CENTER = get_new_x(Y_HAT)
Y_PIXEL_CENTER = get_new_y(X_HAT)

X_PIXEL_MIN, X_PIXEL_MAX = get_min_max_x(X_PIXEL_CENTER, WIDTH)
Y_PIXEL_MIN, Y_PIXEL_MAX = get_min_max_y(Y_PIXEL_CENTER, LENGTH)

print(X_PIXEL_CENTER, Y_PIXEL_CENTER)
print(X_PIXEL_MIN)
print(X_PIXEL_MAX)
print(Y_PIXEL_MIN)
print(Y_PIXEL_MAX)

cnt = np.array([
    [[X_PIXEL_MIN,Y_PIXEL_MIN]],
    [[X_PIXEL_MAX,Y_PIXEL_MIN]],
    [[X_PIXEL_MIN,Y_PIXEL_MAX]],
    [[X_PIXEL_MAX,Y_PIXEL_MAX]]
])

rect_img = cv2.rectangle(img, (X_PIXEL_MIN,Y_PIXEL_MIN), (X_PIXEL_MAX,Y_PIXEL_MAX), (0,0,225), 1)
cv2.imshow('image', rect_img)
cv2.imwrite('temp.png',rect_img)
cv2.waitKey(0)