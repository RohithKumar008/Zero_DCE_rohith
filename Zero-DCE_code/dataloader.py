import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(lowlight_images_path):




	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")

	train_list = image_list_lowlight

	random.shuffle(train_list)

	return train_list

def populate_train_list_v2(lowlight_images_path):
	"""Robust file collector: supports multiple extensions and returns absolute paths."""
	lowlight_images_path = os.path.expanduser(lowlight_images_path)
	if not lowlight_images_path.endswith(os.sep):
		lowlight_images_path = lowlight_images_path + os.sep

	patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]
	image_list = []
	for p in patterns:
		image_list.extend(glob.glob(os.path.join(lowlight_images_path, p)))

	# remove duplicates and make absolute
	image_list = sorted(list({os.path.abspath(p) for p in image_list}))
	random.shuffle(image_list)
	return image_list

	

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path):

		self.train_list = populate_train_list_v2(lowlight_images_path)
		self.size = 256

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))


		

	def __getitem__(self, index):

		data_lowlight_path = self.data_list[index]
		
		data_lowlight = Image.open(data_lowlight_path)
		
		data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)

		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()

		return data_lowlight.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)

