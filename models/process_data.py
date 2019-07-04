import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def train_transform(crop_size):
	return Compose([
		ToPILImage(),
		# Resize() takes as input a PIL image, first argument is desired output size
		# https://pytorch.org/docs/stable/torchvision/transforms.html
		Resize(crop_size, interpolation=Image.BICUBIC),
		ToTensor()
	])


class TrainDatasetFromFolder(Dataset):
	def __init__(self, dataset_dir, crop_size=224):
		"""

		:param dataset_dir:
		:param crop_size: fix to 224, for ResNet
		:param ratio:
		"""
		super(TrainDatasetFromFolder, self).__init__()
		self.dataset_dir = dataset_dir

		self.image_filenames = list()
		self.class2id = dict()

		self.transform = train_transform(crop_size)

	def fetch_file_names(self):
		sub_dirs = os.listdir(self.dataset_dir)

		for sub_dir in sub_dirs:
			file_names = os.listdir(os.path.join(self.dataset_dir, sub_dir))
			for file_name in file_names:
				full_path = os.path.join(self.dataset_dir, sub_dir, file_name)
				if is_image_file(full_path):
					self.class2id[sub_dir] = len(self.class2id.items())
					self.image_filenames.append((full_path, self.class2id[sub_dir]))

	def __getitem__(self, index):

		image = self.transform(Image.open(self.image_filenames[index][0]))
		label = self.image_filenames[index][1]

		return image, label

	def __len__(self):
		return len(self.image_filenames)
