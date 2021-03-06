from torch.utils.data import Dataset
import numpy as np
import pickle


class Cifar10Loader(Dataset):

    def __init__(self, filename):
        self.dataset = self.unpickle(filename)
        self.images = self.dataset[b'data']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]/255
        g_img = np.reshape(self.convert_rgb2gray(img), (1,32,32))
        return {'input': g_img, 'target': np.reshape(img, (3,32,32))}

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dictionary = pickle.load(fo, encoding='bytes')
        return dictionary

    def convert_rgb2gray(self, img):
        r = img[0:1024]
        g = img[1024:2048]
        b = img[2048:]
        # use grayscale converison of OpenCV
        return 0.299*r + 0.587*g + 0.114*b