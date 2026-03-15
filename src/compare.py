
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

def compare_sample_frames(real_folder, fake_folder):

    real_images = sorted(os.listdir(real_folder))
    fake_images = sorted(os.listdir(fake_folder))

    limit = min(len(real_images), len(fake_images), 10)

    for i in range(limit):

        imgA = cv2.imread(os.path.join(real_folder, real_images[i]), 0)
        imgB = cv2.imread(os.path.join(fake_folder, fake_images[i]), 0)

        imgA = cv2.resize(imgA, (256,256))
        imgB = cv2.resize(imgB, (256,256))

        m = mse(imgA, imgB)
        s = ssim(imgA, imgB)

        print(f"Frame {i}: MSE={m:.2f} SSIM={s:.2f}")
