from networks.resnet import *
from torch import nn

class Model(nn.Module):
	def __init__(self, args):

		super(Model, self).__init__()
		self.args = args

		if self.args.model == 'resnet':
			print('ResNet-18')
			self.net = resnet18(pretrained=self.args.pretrained, num_classes=self.args.num_classes)
		elif self.args.model == 'resnet101':
			print('ResNet-101')
			self.net = resnet101(pretrained=self.args.pretrained, num_classes=self.args.num_classes)
		elif self.args.model == 'resnet50':
			print('ResNet-50')
			self.net = resnet50(pretrained=self.args.pretrained, num_classes=self.args.num_classes)
		else:
			print('Invalid network {}'.format(self.args.model))
			exit(-1)

	def forward(self, x, test=False):
		batch_size = x.size(0)

		logits = self.net(x, test).view(batch_size, -1)

		# returning logits should be enough
		return logits


