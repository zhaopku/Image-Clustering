import os

def construct_dir(prefix, args):
	path = ''
	path += str(args.model)
	path += '_lr_'
	path += str(args.lr)
	path += '_bt_'
	path += str(args.batch_size)
	path += '_cr_' + str(args.crop_size)

	if args.pretrained:
		path += '_p'

	return os.path.join(prefix, path)