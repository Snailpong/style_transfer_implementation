import torch
import numpy as np
import random
import cv2
import os


def init_device_seed(seed, cuda_visible):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    return device


def get_smoothed_image(img):
    img_blur = cv2.GaussianBlur(img, (5,5), 0)
    img_canny = cv2.Canny(img, 100, 200)

    img_result = np.copy(img)

    for i in range(img_canny.shape[0]):
        for j in range(img_canny.shape[1]):
            if img_canny[i, j] == 255:
                for a in range(-2, 3):
                    for b in range(-2, 3):
                        if 0 <= a < img.shape[0] and 0 <= b < img.shape[1]:
                            img_result[a, b] = img_blur[a, b]

    return img_result
