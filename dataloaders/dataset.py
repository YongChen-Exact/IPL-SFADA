import itertools
import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from skimage import exposure
from torch.utils.data import Dataset
from torchvision import transforms


#  Normal fine-tuning or first training of the source model
class BaseDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
    ):
        self._base_dir = base_dir
        print(self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform

        if self.split == "train":
            self.sample_list = sorted(os.listdir(self._base_dir + "/training_set_mixed_DR_r0.3/"))

        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/val_set/")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/training_set_mixed_DR_r0.3/{}".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/val_set/{}".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        label[label > 0] = 1
        if self.transform == None:
            image = zoom(
                image, (1, 256 / image.shape[0], 256 / image.shape[1]), order=0)
            label = zoom(
                label, (1, 256 / label.shape[0], 256 / label.shape[1]), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.int16))

        if self.split == "train":
            sample = {"image": image, "label": label, "name": case}
            if self.transform != None:
                sample = self.transform(sample)
        else:
            sample = {"image": image, "label": label, "name": case}

        sample["idx"] = idx
        return sample


#  first fine-tuning
class BaseDataSets1(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None
    ):
        self._base_dir = base_dir
        print(self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform
        list_NPC = []
        with open(
                '/home/data/CY/codes/BDK-SFADA/Results/selection_list/Active_sample_CenterC_DR_r0.5.txt',
                'r') as file:
            lines = file.readlines()

        if self.split == "train":
            for line in lines:
                list_NPC.append(line.replace("\n", ""))
            self.sample_list = list_NPC
            print('len(self.sample_list)=', len(self.sample_list))

        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/val_set/")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/training_set/{}".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/val_set/{}".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        label[label > 0] = 1

        if self.split == "train":
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
        else:
            sample = {"image": image, "label": label.astype(np.int16)}

        sample["idx"] = idx
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        if len(label.shape) == 2:
            label = np.rot90(label, k)
            label = np.flip(label, axis=axis).copy()
            return image, label
        elif len(label.shape) == 3:
            new_label = np.zeros_like(label)
            for i in range(new_label.shape[0]):
                new_label[i, ...] = np.rot90(label[i, ...], k)
                new_label[i, ...] = np.flip(label[i, ...], axis=axis).copy()
            return image, new_label
        else:
            Exception("Error")
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    if len(label.shape) == 2:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label
    elif len(label.shape) == 3:
        new_label = np.zeros_like(label)
        for i in range(label.shape[0]):
            new_label[i, ...] = ndimage.rotate(
                label[i, ...], angle, order=0, reshape=False)
        return image, new_label
    else:
        Exception("Error")


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


def random_noise(image, label, mu=0, sigma=0.1):
    noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1]),
                    -2 * sigma, 2 * sigma)
    noise = noise + mu
    image = image + noise
    return image, label


def random_rescale_intensity(image, label):
    image = exposure.rescale_intensity(image)
    return image, label


def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
        # if random.random() > 0.5:
        #     image, label = random_rescale_intensity(image, label)
        # if random.random() > 0.5:
        #     image, label = random_equalize_hist(image, label)
        # print(image.shape)
        x, y = image.shape

        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int16))
        sample = {"image": image, "label": label}
        return sample


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
