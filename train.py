import cv2
import pandas as pd
import os
import numpy as np
import operator

def distance_between(p1, p2): 
    a = p2[0] - p1[0] 
    b = p2[1] - p1[1] 
    return np.sqrt((a ** 2) + (b ** 2))



wd = os.getcwd()

data = pd.read_pickle(f"{wd}/data.pkl")

image_location = f"{wd}\images\image1.jpg"


# Using this article to invert image and then to crop.

image = cv2.imread(image_location)
grayscale = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

proc = cv2.GaussianBlur(grayscale.copy(), (9, 9), 0)
proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
Mask = cv2.bitwise_not(proc, proc)


contours, hierarchy = cv2.findContours(Mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)
polygon = contours[0]

bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                      polygon]), key=operator.itemgetter(1))
top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                  polygon]), key=operator.itemgetter(1))
bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                     polygon]), key=operator.itemgetter(1))
top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                   polygon]), key=operator.itemgetter(1))

crop_rect = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32') 
side = max([  distance_between(bottom_right, top_right), 
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),   
            distance_between(top_left, top_right) ])

dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
m = cv2.getPerspectiveTransform(src, dst)
cropped_image = cv2.warpPerspective(Mask, m, (int(side), int(side)))
# print(polygon)
cv2.imshow('Img', cropped_image)
cv2.waitKey(0)

