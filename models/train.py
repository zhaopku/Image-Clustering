from models.process_data import TrainDatasetFromFolder
import argparse
import torch
from torch.utils.data import DataLoader
from models.model import Model
from models import utils
import os
from tensorboardX import SummaryWriter

class Train:
	def __init__(self):
		self.args = None
		self.training_set = None
		self.val_set = None
		self.train_loader = None
		self.val_loader = None
		self.model = None

		self.result_dir = None
		self.writer = None

		self.global_step = 0
		self.global_val_step = 0

	@staticmethod
	def parse_args(args):
		parser = argparse.ArgumentParser()

		data_args = parser.add_argument_group('Dataset options')
		data_args.add_argument('--data_dir', type=str, default='./data/Architectural_Style')
		data_args.add_argument('--summary_dir', type=str, default='summaries')
		data_args.add_argument('--result_dir', default='./result')

		# neural network options
		nn_args = parser.add_argument_group('Network options')
		nn_args.add_argument('--crop_size', default=224, type=int, help='training images crop size')
		nn_args.add_argument('--model', type=str, default='resnet', help='string to specify model')
		nn_args.add_argument('--num_classes', type=int, default=17)

		# training options
		training_args = parser.add_argument_group('Training options')
		training_args.add_argument('--pretrained', action='store_true')
		training_args.add_argument('--batch_size', type=int, default=3)
		training_args.add_argument('--n_save', type=int, default=50, help='number of test images to save on disk')
		training_args.add_argument('--epochs', type=int, default=100, help='number of training epochs')
		training_args.add_argument('--lr', type=float, default=0.001, help='learning rate')
		training_args.add_argument('--resume', action='store_true')
		training_args.add_argument('--load_model', action='store_true')

		return parser.parse_args(args)
	def load_model(self, model_path):

		if torch.cuda.is_available():
			(model_state_dict, optimizer_state_dict, self.global_step, self.global_val_step) = torch.load(model_path)
		else:
			(model_state_dict, optimizer_state_dict, self.global_step, self.global_val_step) = torch.load(model_path, map_location='cpu')

		self.model.load_state_dict(model_state_dict)

		# only load optimizer when we are resuming from a checkpoint
		if self.args.resume > 0:
			self.optimizer.load_state_dict(optimizer_state_dict)
			if torch.cuda.is_available():
				for state in self.optimizer.state.values():
					for k, v in state.items():
						if isinstance(v, torch.Tensor):
							state[k] = v.cuda()

	def construct_data(self):
		self.dataset = TrainDatasetFromFolder(dataset_dir=self.args.data_dir,
		                                           crop_size=self.args.crop_size)

		self.training_set, self.val_set \
			= torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*0.9), len(self.dataset) - int(len(self.dataset)*0.9)])

		self.train_loader = DataLoader(dataset=self.training_set, num_workers=1, batch_size=self.args.batch_size, shuffle=True)

		# images are of different size, hence test batch size must be 1
		self.val_loader = DataLoader(dataset=self.val_set, num_workers=1, batch_size=self.args.batch_size, shuffle=False)

	def construct_model(self):
		if self.args.model == 'resnet':
			self.model = Model(self.args)

	def construct_out_dir(self):
		self.result_dir = utils.construct_dir(prefix=self.args.result_dir, args=self.args)
		self.model_dir = os.path.join(self.result_dir, 'models')
		self.out_path = os.path.join(self.result_dir, 'result.txt')
		self.summary_dir = utils.construct_dir(prefix=self.args.summary_dir, args=self.args)

		if not os.path.exists(self.summary_dir):
			os.makedirs(self.summary_dir)
		self.writer = SummaryWriter(log_dir=self.summary_dir)

		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

	def main(self, args=None):
		print('PyTorch Version {}, GPU enabled {}'.format(torch.__version__, torch.cuda.is_available()))
		self.args = self.parse_args(args=args)

		self.construct_data()

		self.construct_model()

		self.construct_out_dir()

		with open(self.out_path, 'w') as self.out:
			self.train_loop()

	def train_loop(self):
		pass