import os
import torch
from torchvision import datasets as dsets
from torchvision import transforms
import datasets.grayfolderloader as grayfolderloader
import utils

"""
Dataloaders for the MNIST dataset
"""
class LoadMNIST():
	'''
	Downloads and loads the MNIST dataset.
	Preprocessing -> Data is normalized in Transforms.
	'''
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			dsets.MNIST('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(28, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,))
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			dsets.MNIST('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
			  **kwargs)

"""
Dataloaders for the CIFAR-10 dataset
"""
class LoadCIFAR10():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			dsets.CIFAR10('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			dsets.CIFAR10('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
		  **kwargs)

"""
Dataloaders for the CIFAR-100 dataset
"""
class LoadCIFAR100():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			dsets.CIFAR100('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
std=[x/255.0 for x in [68.2, 65.4, 70.4]])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			dsets.CIFAR100('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
std=[x/255.0 for x in [68.2, 65.4, 70.4]])
					   ])),
		  **kwargs)

"""
Dataloaders for the TUBerlin dataset
"""
class LoadTuberlin():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		valkwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.testbatchsize,
		  'shuffle' : True,
		  'pin_memory': True}

		crop_size = opt.inpsize

		# Scale down images to 256x256, perform a random size crop, randomly flip images and then rotate in a range of 10 degrees
		# Invert images to ensure white borders on black background. Normalize the data to ensure it is mean centered and in range.
		train_transform = transforms.Compose([
			transforms.Scale(256),
			transforms.RandomSizedCrop(crop_size),
			transforms.RandomHorizontalFlip(),
			utils.RandomRotate(10),
			transforms.ToTensor(),
			utils.Invert(),
			transforms.Normalize(mean=[0.06,], std=[0.93,])
		])

		if opt.tenCrop:
			val_transform = transforms.Compose([
				transforms.Scale(256),
				transforms.ToTensor(),
				utils.Invert(),
				# TenCrop performs normalization internally
				utils.TenCrop(crop_size, opt)
			])
		else:
			val_transform = transforms.Compose([
				transforms.Scale(256),
				transforms.CenterCrop(crop_size),
				transforms.ToTensor(),
				utils.Invert(),
				transforms.Normalize(mean=[0.06,], std=[0.93])
			])

		data_transforms = {
			'train': train_transform,
			'val': val_transform
		}

		data_dir = opt.data_dir + '/tuberlin'
		# Grayfolderloader ensures images are loaded as grayscale
		dsets = {x: grayfolderloader.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
		self.dsets = dsets
		self.train_loader = torch.utils.data.DataLoader(dsets["train"], **kwargs)
		self.val_loader = torch.utils.data.DataLoader(dsets["val"], **valkwargs)

"""
Dataloaders for the Sketchy (Recognition) dataset
"""
class LoadSketchyRecognition():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		valkwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.testbatchsize,
		  'shuffle' : True,
		  'pin_memory': True}

		crop_size = opt.inpsize

		# Scale down images to 256x256, perform a random size crop, randomly flip images and then rotate in a range of 10 degrees
		# Invert images to ensure white borders on black background. Normalize the data to ensure it is mean centered and in range.
		train_transform = transforms.Compose([
			transforms.Scale(256),
			transforms.RandomSizedCrop(crop_size),
			utils.RandomRotate(10),
			transforms.ToTensor(),
			utils.Invert(),
			transforms.Normalize(mean=[0.0465,], std=[0.9])
		])

		if opt.tenCrop:
			val_transform = transforms.Compose([
				transforms.Scale(256),
				transforms.ToTensor(),
				utils.Invert(),
				# TenCrop performs normalization internally
				utils.TenCrop(crop_size, opt)
			])
		else:
			val_transform = transforms.Compose([
				transforms.Scale(256),
				transforms.CenterCrop(crop_size),
				transforms.ToTensor(),
				utils.Invert(),
				transforms.Normalize(mean=[0.0465,], std=[0.9])
			])

		data_transforms = {
			'train': train_transform,
			'val': val_transform
		}

		data_dir = opt.data_dir + '/sketchy_recognition'
		# Grayfolderloader ensures images are loaded as grayscale
		dsets = {x: grayfolderloader.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
		self.dsets = dsets
		self.train_loader = torch.utils.data.DataLoader(dsets["train"], **kwargs)
		self.val_loader = torch.utils.data.DataLoader(dsets["val"], **valkwargs)

"""
Dataloaders for the SVHN dataset
"""
class LoadSVHN():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			dsets.SVHN('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.ToTensor(),
						#Normalization pending
						transforms.Normalize([0,0,0], [0,0,0])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			dsets.SVHN('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   #Normalization pending
						   transforms.Normalize([0,0,0], [0,0,0])
					   ])),
		  **kwargs)

"""
Dataloaders for the STL-10 dataset
"""
class LoadSTL10():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			dsets.STL10('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.ToTensor(),
						#Normalization pending
						transforms.Normalize([0,0,0], [0,0,0])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			dsets.STL10('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   #Normalization pending
						   transforms.Normalize([0,0,0], [0,0,0])
					   ])),
		  **kwargs)

"""
Dataloaders for the Imagenet12 dataset
"""
class LoadImagenet12():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		# Perform a random size crop of 224x224 and randomly flip images.
		# Invert images to ensure white borders on black background. Normalize the data to ensure it is mean centered and in range.
		data_transforms = {
			'train': transforms.Compose([
				transforms.RandomSizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]),
			'val': transforms.Compose([
				transforms.Scale(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		}

		data_dir = opt.data_dir
		dtsets = {x: dsets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

		self.train_loader = torch.utils.data.DataLoader(dtsets["train"], **kwargs)
		self.val_loader = torch.utils.data.DataLoader(dtsets["val"], **kwargs)
