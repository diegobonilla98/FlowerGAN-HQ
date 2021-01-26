import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import glob
from scipy.io import loadmat
from tensorflow.keras.utils import to_categorical


class DataLoader:
    def __init__(self, size):
        self.ROOT = '/media/bonilla/HDD_2TB_basura/databases/102flowers/jpg'
        self.IMAGES = np.array(
            sorted(glob.glob(os.path.join(self.ROOT, '*')), key=lambda x: int(os.path.split(x)[-1][6: 11])))
        self.LABELS = loadmat('/media/bonilla/HDD_2TB_basura/databases/102flowers/imagelabels.mat')['labels'][0]
        self.NUM_IMAGES = len(self.IMAGES)
        self.size = size
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def augment(self, i, rand, rot, gamma):
        if rand[0] > 0.5:
            i = cv2.flip(i, 0)
        if rand[1] > 0.5:
            i = cv2.flip(i, 1)
        if rand[2] > 0.5:
            i = cv2.rotate(i, rot)
        if rand[3] > 0.5:
            l, a, b = cv2.split(cv2.cvtColor(i, cv2.COLOR_BGR2LAB))
            l = self.clahe.apply(l)
            i = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        if rand[4] > 0.5:
            invGamma = 1.0 / gamma
            table = np.array([((ii / 255.0) ** invGamma) * 255 for ii in np.arange(0, 256)]).astype("uint8")
            i = cv2.LUT(i, table)
        return i

    def load_image(self, path, rand, rot, gamma, size):
        image = cv2.imread(path)
        image = cv2.resize(image, (size, size))
        image = self.augment(image, rand, rot, gamma)
        return (image.astype('float32') - 127.5) / 127.5

    def load_batch(self, batch_size):
        rand_idx = np.random.choice(self.NUM_IMAGES, size=(batch_size,), replace=True)
        paths = self.IMAGES[rand_idx]
        rand = np.random.rand(5)
        rot = np.random.randint(0, 3)
        gamma = np.random.rand() * 2. + 0.5
        X = np.array([self.load_image(p, rand, rot, gamma, self.size) for p in paths])
        labels = to_categorical(self.LABELS[rand_idx] - 1, num_classes=102)
        return X, labels
