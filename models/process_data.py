import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch
from collections import defaultdict

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def train_transform(crop_size):
	return Compose([
		# Resize() takes as input a PIL image, first argument is desired output size
		# https://pytorch.org/docs/stable/torchvision/transforms.html
		Resize((crop_size, crop_size), interpolation=Image.BICUBIC),
		ToTensor()
	])


class TrainDatasetFromFolder(Dataset):
	def __init__(self, dataset_dir, crop_size=224, test=False):
		"""

		:param dataset_dir:
		:param crop_size: fix to 224, for ResNet
		:param ratio:
		"""
		super(TrainDatasetFromFolder, self).__init__()
		self.dataset_dir = dataset_dir

		self.image_filenames = list()
		self.class2id = defaultdict(int)

		self.transform = train_transform(crop_size)

		self.test = test

		self.name2id = dict()

		self.fetch_file_names()

		self.id2name = dict()

		for k, v in self.name2id.items():
			self.id2name[v] = k

		assert len(self.name2id.items()) == len(self.id2name.items())

		self.id2class = dict()
		for k, v in self.class2id.items():
			self.id2class[v] = k

		assert len(self.class2id.items()) == len(self.id2class.items())

	def fetch_file_names(self):
		sub_dirs = os.listdir(self.dataset_dir)

		for sub_dir in sub_dirs:
			if sub_dir.startswith('.'):
				continue
			file_names = os.listdir(os.path.join(self.dataset_dir, sub_dir))
			for file_name in file_names:
				full_path = os.path.join(self.dataset_dir, sub_dir, file_name)
				if is_image_file(full_path):
					self.class2id[sub_dir] += 1
					self.class2id[sub_dir] = len(self.class2id.items()) - 1
					self.image_filenames.append((full_path, self.class2id[sub_dir]))
					self.name2id[full_path] = len(self.name2id.items())

	def __getitem__(self, index):

		image = self.transform(Image.open(self.image_filenames[index][0]))
		label = self.image_filenames[index][1]

		if image.size(0) == 1:

			image = image.repeat(3, 1, 1)

		if image.size(0) == 4:
			image = image[:3, :, :]

		if self.test:
			return image, self.name2id[self.image_filenames[index][0]]

		return image, label

	def __len__(self):
		return len(self.image_filenames)
