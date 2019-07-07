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
from sklearn.cluster import KMeans
from shutil import copyfile
from collections import defaultdict
import copy
from torch.autograd.gradcheck import zero_gradients
import torchvision

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
		training_args.add_argument('--clustering', action='store_true')
		training_args.add_argument('--load_model', type=str, default='dummy')
		training_args.add_argument('--saliency', action='store_true')

		return parser.parse_args(args)
	def load_model(self, model_path):

		if torch.cuda.is_available():
			model_state_dict = torch.load(model_path)
		else:
			model_state_dict = torch.load(model_path, map_location='cpu')

		self.model.load_state_dict(model_state_dict)

	def construct_data(self):
		self.dataset = TrainDatasetFromFolder(dataset_dir=self.args.data_dir,
		                                           crop_size=self.args.crop_size)

		self.training_set, self.val_set \
			= torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*0.9), len(self.dataset) - int(len(self.dataset)*0.9)])

		self.data_loader = DataLoader(dataset=self.dataset, num_workers=2, batch_size=self.args.batch_size, shuffle=False)

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
		self.image_dir = utils.construct_dir(prefix='images', args=self.args)

		if not os.path.exists(self.summary_dir):
			os.makedirs(self.summary_dir)
		self.writer = SummaryWriter(log_dir=self.summary_dir)

		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

	def clustering(self):
		self.test_set = TrainDatasetFromFolder(dataset_dir='./data/test',
		                                           crop_size=self.args.crop_size, test=True)
		self.test_loader = DataLoader(dataset=self.test_set, num_workers=2, batch_size=self.args.batch_size, shuffle=False)

		# first obtain the image embeddings

		if torch.cuda.is_available():
			self.model.cuda()

		all_embeddings = []
		all_ids = []
		all_predictions = []

		self.model.eval()
		for idx, (images, ids) in enumerate(tqdm(self.test_loader)):
			if torch.cuda.is_available():
				images = images.cuda()
				ids = ids.cuda()

			embeddings = self.model(images, test=True)
			logits = self.model(images, test=False)
			predictions = torch.argmax(logits, -1)

			all_embeddings.extend(list(embeddings.data.cpu().numpy()))
			all_ids.extend(list(ids.data.cpu().numpy()))
			all_predictions.extend(list(predictions.data.cpu().numpy()))

		all_embeddings = np.asarray(all_embeddings)
		all_ids = np.asarray(all_ids)

		# do predictions
		dst = os.path.join(self.image_dir, 'predictions')
		for i in range(17):
			label = self.dataset.id2class[i]
			if not os.path.exists(os.path.join(dst, label)):
				os.makedirs(os.path.join(dst, label))

		for id_ in all_ids:
			name = self.test_set.id2name[id_]
			prediction = all_predictions[id_]
			label = self.dataset.id2class[prediction]
			file_name = name.split('/')[-1]
			copyfile(name, os.path.join(dst, label, file_name))

		# do clustering
		n_clusters = [2, 3, 5, 10, 15, 17, 20]
		id2name = self.test_set.id2name
		for n in n_clusters:
			print('clustering, {}'.format(n))
			kmeans = KMeans(n_clusters=n, n_jobs=20).fit(all_embeddings)
			labels = kmeans.labels_

			# move images to cluster folders
			dst = os.path.join(self.image_dir, str(n)+'_clusters')
			if not os.path.exists(dst):
				os.makedirs(dst)

			for i in range(n):
				if not os.path.exists(os.path.join(dst, str(i))):
					os.makedirs(os.path.join(dst, str(i)))

			for id_, name in id2name.items():
				label = labels[id_]
				file_name = name.split('/')[-1]
				copyfile(name, os.path.join(dst, str(label), file_name))

	def statistics(self):
		all_labels = []
		for idx, (images, labels) in enumerate(tqdm(self.data_loader)):

			all_labels.extend(list(labels.data.cpu().numpy()))

		id2class = self.dataset.id2class

		class2cnt = defaultdict(int)

		for label in all_labels:
			c = id2class[label]
			class2cnt[c] += 1

		l = []
		for k, v in class2cnt.items():
			l.append((k, v))

		l = sorted(l, key=lambda x: x[0])

		for item in l:
			print(item[0], item[1])

	def saliency_map(self):
		self.test_set = TrainDatasetFromFolder(dataset_dir='./data/test',
		                                           crop_size=self.args.crop_size, test=True)
		self.test_loader = DataLoader(dataset=self.test_set, num_workers=2, batch_size=self.args.batch_size, shuffle=False)

		# first obtain the image embeddings

		if torch.cuda.is_available():
			self.model.cuda()

		self.model.eval()
		all_grads = []
		all_predictions = []
		all_ids = []
		all_images = []

		for idx, (images, ids) in enumerate(tqdm(self.test_loader)):
			if torch.cuda.is_available():
				images = images.cuda()
				ids = ids.cuda()
			all_images.append(images.cpu().data)
			images.requires_grad_(True)

			# [batch_size, 17]
			logits = self.model(images, test=False)
			predictions = torch.argmax(logits, -1)
			all_predictions.extend(list(predictions.data.cpu().numpy()))
			all_ids.extend(list(ids.data.cpu().numpy()))

			max_logits, _ = torch.max(logits, dim=1)

			zero_gradients(images)
			max_logits.sum().backward(retain_graph=True)

			grads = copy.deepcopy(images.grad.data)
			grads = torch.abs(grads)
			grads, _ = torch.max(grads, dim=1, keepdim=True)
			all_grads.append(grads)

		all_grads = torch.cat(all_grads)
		all_ids = np.asarray(all_ids)
		all_images = torch.cat(all_images)

		# make saliency dirs
		dst = os.path.join(self.image_dir, 'cropped')
		for i in range(17):
			label = self.dataset.id2class[i]
			if not os.path.exists(os.path.join(dst, label)):
				os.makedirs(os.path.join(dst, label))

		for idx, id_ in enumerate(tqdm(all_ids, desc='saving cropped images')):
			name = self.test_set.id2name[id_]
			prediction = all_predictions[id_]
			label = self.dataset.id2class[prediction]
			file_name = name.split('/')[-1]
			path = os.path.join(dst, label, file_name)
			# save image with full_name
			image_cropped = all_images[idx]
			self.save_image(image_cropped, path)
		exit()

		# make saliency dirs
		dst = os.path.join(self.image_dir, 'saliency')
		for i in range(17):
			label = self.dataset.id2class[i]
			if not os.path.exists(os.path.join(dst, label)):
				os.makedirs(os.path.join(dst, label))

		for idx, id_ in enumerate(tqdm(all_ids, desc='saving saliency maps')):
			name = self.test_set.id2name[id_]
			prediction = all_predictions[id_]
			label = self.dataset.id2class[prediction]
			file_name = name.split('/')[-1]
			path = os.path.join(dst, label, file_name)
			# save image with full_name
			image_normalized = all_grads[idx]/all_grads[idx].max()
			self.save_image(image_normalized, path)

	def save_image(self, image, path):
		torchvision.utils.save_image(image, path)

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

		# self.statistics()
		# exit(0)

		if self.args.load_model != 'dummy':
			self.load_model(model_path=self.args.load_model)

		if self.args.clustering:
			with torch.no_grad():
				self.clustering()
				exit()

		if self.args.saliency:
			self.saliency_map()
			exit()

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
				logits = self.model(images, test=False)

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
			self.out.write(result_line+'\n')
			self.out.flush()

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
				logits = self.model(images, test=False)

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
			self.out.write(result_line+'\n')
			self.out.flush()

			self.writer.add_scalar('val/total_loss', val_results['total_loss'], e)
			self.writer.add_scalar('val/avg_loss', val_results['total_loss']/val_results['n_samples'], e)
			self.writer.add_scalar('val/acc', val_results['n_corrects']*1.0/val_results['n_samples'], e)
			self.writer.add_scalar('val/f1_micro', f1_micro, e)
			self.writer.add_scalar('val/f1_macro', f1_macro, e)

	def compute_F1(self, predictions, labels):
		f1_micro = f1_score(y_true=labels, y_pred=predictions, average='micro')
		f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')

		return f1_micro, f1_macro
