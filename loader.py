import os
import struct
import numpy as np

class Loader:

    def __init__(self, path):
        self.path = path

    def load_train(self):

        lbpath = os.path.join(self.path, 'train-labels-idx1-ubyte')
        imgpath = os.path.join(self.path, 'train-images-idx3-ubyte')

        with open(lbpath, 'rb') as lbfile:
            magic, n = struct.unpack('>II', lbfile.read(8))
            labels = np.fromfile(lbfile, dtype=np.uint8)

        with open(imgpath, 'rb') as imgfile:
            magic, n, rows, cols = struct.unpack('>IIII', imgfile.read(16))
            img = np.fromfile(imgfile, dtype=np.uint8).reshape(len(labels), 28, 28, 1)
            img = np.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)

        return img, labels

    def load_test(self):

        lbpath = os.path.join(self.path, 't10k-labels-idx1-ubyte')
        imgpath = os.path.join(self.path, 't10k-images-idx3-ubyte')

        with open(lbpath, 'rb') as lbfile:
            magic, n = struct.unpack('>II', lbfile.read(8))
            labels = np.fromfile(lbfile, dtype=np.uint8)

        with open(imgpath, 'rb') as imgfile:
            magic, n, rows, cols = struct.unpack('>IIII', imgfile.read(16))
            img = np.fromfile(imgfile, dtype=np.uint8).reshape(len(labels), 28, 28, 1)
            img = np.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)

        return img, labels
