from models.process_data import TrainDatasetFromFolder
import argparse
import torch
from torch.utils.data import DataLoader
from models.model import Model
from models import utils
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.optim as optimizer
from sklearn.metrics import f1_score
import numpy as np

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
		training_args.add_argument('--batch_size', type=int, default=100)
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

		self.train_loader = DataLoader(dataset=self.training_set, num_workers=2, batch_size=self.args.batch_size, shuffle=True)

		# images are of different size, hence test batch size must be 1
		self.val_loader = DataLoader(dataset=self.val_set, num_workers=2, batch_size=self.args.batch_size, shuffle=False)

	def construct_model(self):
		if self.args.model == 'resnet':
			self.model = Model(self.args)

		self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
		self.optimizer = optimizer.Adam(self.model.parameters(), lr=self.args.lr)

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
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'
		# torch.backends.cudnn.deterministic = True
		# torch.backends.cudnn.benchmark = False
		torch.manual_seed(1209)
		np.random.seed(32389)

		print('PyTorch Version {}, GPU enabled {}'.format(torch.__version__, torch.cuda.is_available()))

		self.args = self.parse_args(args=args)

		self.construct_data()

		self.construct_model()

		self.construct_out_dir()

		with open(self.out_path, 'w') as self.out:
			self.train_loop()

	def train_loop(self):

		if torch.cuda.is_available():
			self.model.cuda()
			self.loss.cuda()

		for e in range(self.args.epochs):
			# switch to training mode
			self.model.train()

			train_results = {'total_loss': 0.0, 'n_samples': 0, 'n_corrects': 0}

			all_predictions = []
			all_labels = []

			for idx, (images, labels) in enumerate(tqdm(self.train_loader)):
				self.global_step += 1
				if torch.cuda.is_available():
					images = images.cuda()
					labels = labels.cuda()

				cur_batch_size = images.size(0)
				train_results['n_samples'] += cur_batch_size

				# [batch_size, 17]
				logits = self.model(images)

				predictions = torch.argmax(logits, -1)

				all_predictions.extend(list(predictions.data.cpu().numpy()))
				all_labels.extend(list(labels.data.cpu().numpy()))

				corrects = torch.sum(torch.eq(labels, predictions)).data.cpu().numpy()
				train_results['n_corrects'] += corrects
				loss = self.loss(logits, labels)

				train_results['total_loss'] += cur_batch_size*loss.data.cpu().numpy()

				self.writer.add_scalar('train/step.loss', loss, self.global_step)
				self.writer.add_scalar('train/step.corrects', corrects, self.global_step)

				self.model.zero_grad()
				loss.backward()
				self.optimizer.step()

			f1_micro, f1_macro = self.compute_F1(predictions=all_predictions, labels=all_labels)

			result_line = 'Epoch {}, '.format(e)
			result_line += 'total_loss = {}, '.format(train_results['total_loss'])
			result_line += 'avg_loss = {}, '.format(train_results['total_loss']/train_results['n_samples'])
			result_line += 'acc = {}, '.format(train_results['n_corrects']*1.0/train_results['n_samples'])
			result_line += 'f1_micro = {}, '.format(f1_micro)
			result_line += 'f1_macro = {}\n\t'.format(f1_macro)
			print(result_line)
			self.writer.add_scalar('train/total_loss', train_results['total_loss'], e)
			self.writer.add_scalar('train/avg_loss', train_results['total_loss']/train_results['n_samples'], e)
			self.writer.add_scalar('train/acc', train_results['n_corrects']*1.0/train_results['n_samples'], e)
			self.writer.add_scalar('train/f1_micro', f1_micro, e)
			self.writer.add_scalar('train/f1_macro', f1_macro, e)

			self.validate(e)
			torch.save(self.model.state_dict(), os.path.join(self.model_dir, str(e)+'.pth'))

	def validate(self, e):
		with torch.no_grad():
			if torch.cuda.is_available():
				self.model.cuda()
				self.loss.cuda()

			# switch to eval mode
			self.model.eval()

			val_results = {'total_loss': 0.0, 'n_samples': 0, 'n_corrects': 0}
			all_predictions = []
			all_labels = []

			for idx, (images, labels) in enumerate(tqdm(self.val_loader)):
				self.global_step += 1
				if torch.cuda.is_available():
					images = images.cuda()
					labels = labels.cuda()

				cur_batch_size = images.size(0)
				val_results['n_samples'] += cur_batch_size

				# [batch_size, 17]
				logits = self.model(images)

				predictions = torch.argmax(logits, -1)

				all_predictions.extend(list(predictions.data.cpu().numpy()))
				all_labels.extend(list(labels.data.cpu().numpy()))

				corrects = torch.sum(torch.eq(labels, predictions)).data.cpu().numpy()
				val_results['n_corrects'] += corrects
				loss = self.loss(logits, labels)

				val_results['total_loss'] += cur_batch_size*loss.data.cpu().numpy()

				self.writer.add_scalar('val/step.loss', loss, self.global_step)
				self.writer.add_scalar('val/step.corrects', corrects, self.global_step)

			f1_micro, f1_macro = self.compute_F1(predictions=all_predictions, labels=all_labels)

			result_line = '\ttotal_loss = {}, '.format(val_results['total_loss'])
			result_line += 'avg_loss = {}, '.format(val_results['total_loss']/val_results['n_samples'])
			result_line += 'acc = {}, '.format(val_results['n_corrects']*1.0/val_results['n_samples'])
			result_line += 'f1_micro = {}, '.format(f1_micro)
			result_line += 'f1_macro = {}\n\t'.format(f1_macro)
			print(result_line)

			self.writer.add_scalar('val/total_loss', val_results['total_loss'], e)
			self.writer.add_scalar('val/avg_loss', val_results['total_loss']/val_results['n_samples'], e)
			self.writer.add_scalar('val/acc', val_results['n_corrects']*1.0/val_results['n_samples'], e)
			self.writer.add_scalar('val/f1_micro', f1_micro, e)
			self.writer.add_scalar('val/f1_macro', f1_macro, e)

	def compute_F1(self, predictions, labels):
		f1_micro = f1_score(y_true=labels, y_pred=predictions, average='micro')
		f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')

		return f1_micro, f1_macro
