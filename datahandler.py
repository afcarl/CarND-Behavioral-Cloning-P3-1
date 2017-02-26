import os
import csv
import cv2
import numpy as np


DEFAULT_FILENAME = "data/driving_log.csv"
DEFAULT_IMG_DIR = "data/IMG"


def augment(image, steering_angle):
    reverse_image = np.fliplr(image)
    reverse_angle = -steering_angle
    return reverse_image, reverse_angle


def read_driving_log(filename=DEFAULT_FILENAME):
    with open(filename) as log:
        reader = csv.reader(log)
        next(reader)
        for entry in reader:
            image_name = os.path.join(DEFAULT_IMG_DIR, os.path.basename(entry[0]))
            image = cv2.resize(cv2.imread(image_name), dsize=(224,224))
            steering_angle = float(entry[3])
            yield image, steering_angle


def get():
    images = []
    angles = []
    for image, steering_angle in read_driving_log():
        images.append(image)
        angles.append(steering_angle)
        reverse_image, reverse_angle = augment(image, steering_angle)
        images.append(reverse_image)
        angles.append(reverse_angle)
    return np.array(images), np.array(angles)