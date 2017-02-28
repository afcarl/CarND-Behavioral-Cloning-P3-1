import os
import csv
import cv2
import numpy as np

DEFAULT_FILENAME = "samples/driving_log.csv"
DEFAULT_IMG_DIR = "samples/IMG"
DEFAULT_ANGLE_CORRECTION = 0.15


def augment(image, steering_angle):
    reverse_image = np.fliplr(image)
    reverse_angle = -steering_angle
    return reverse_image, reverse_angle


def read_driving_log(filename=DEFAULT_FILENAME):
    with open(filename) as log:
        reader = csv.reader(log)
        for entry in reader:
            image_name = os.path.join(DEFAULT_IMG_DIR, os.path.basename(entry[0]))
            image_name_left = os.path.join(DEFAULT_IMG_DIR, os.path.basename(entry[1]))
            image_name_right = os.path.join(DEFAULT_IMG_DIR, os.path.basename(entry[2]))
            image = cv2.resize(cv2.imread(image_name), dsize=(224,224)) #
            image_left = cv2.resize(cv2.imread(image_name_left), dsize=(224,224))
            image_right = cv2.resize(cv2.imread(image_name_right), dsize=(224,224))
            steering_angle = float(entry[3])
            yield image, steering_angle, image_left, steering_angle + DEFAULT_ANGLE_CORRECTION, image_right, steering_angle - DEFAULT_ANGLE_CORRECTION


def get():
    images = []
    angles = []
    for image, steering_angle, image_left, steering_left, image_right, steering_right  in read_driving_log():
        images.append(image)
        angles.append(steering_angle)
        images.append(image_left)
        angles.append(steering_left)
        images.append(image_right)
        angles.append(steering_right)
        reverse_image, reverse_angle = augment(image, steering_angle)
        images.append(reverse_image)
        angles.append(reverse_angle)
    return np.array(images), np.array(angles)
