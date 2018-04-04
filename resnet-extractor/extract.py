# coding=utf-8
# need model from http://models.tensorpack.com/ResNet/

import sys,os,argparse,math,cv2
from nn import pretrained_resnet_conv4,resnet_conv5
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from glob import glob
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("imgpath")
	parser.add_argument("weights",help="pre-trained network weights")
	parser.add_argument("featpath")
	parser.add_argument("--batch_size",type=int,default=30)
	parser.add_argument("--resize",type=int,default=448,help="rescale image before into the cnn")
	parser.add_argument("--skip",action="store_true")
	parser.add_argument("--depth",type=int,default=101)

	return parser.parse_args()

CFG = {
	50: [3, 4, 6, 3],
	101: [3, 4, 23, 3],
	152: [3, 8, 36, 3]
}
def build_model(depth=101):
	# [batch,H,W,C]
	image_input = tf.placeholder(tf.float32,[None,None,None,3],name="image")
	image = image_input
	image = image*(1.0/255)
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	bgr = True # cv2 imread is bgr
	if bgr:
		mean = mean[::-1]
		std = std[::-1]
	image_mean = tf.constant(mean, dtype=tf.float32)
	image_std = tf.constant(std,dtype=tf.float32)
	image = (image - image_mean) / image_std

	image = tf.transpose(image,[0, 3, 1, 2])

	resnet_num_block = CFG[depth] # resnet 

	featuremap = pretrained_resnet_conv4(image,resnet_num_block[:3])
	#print featuremap.get_shape()
	featuremap = resnet_conv5(featuremap,resnet_num_block[-1])
	#print featuremap.get_shape()
	featout = tf.transpose(featuremap,[0,2,3,1])
	#print featout.get_shape()
	#sys.exit()

	return image_input,featout

def get_op_tensor_name(name):
	"""
	Will automatically determine if ``name`` is a tensor name (ends with ':x')
	or a op name.
	If it is an op name, the corresponding tensor name is assumed to be ``op_name + ':0'``.

	Args:
		name(str): name of an op or a tensor
	Returns:
		tuple: (op_name, tensor_name)
	"""
	if len(name) >= 3 and name[-2] == ':':
		return name[:-2], name
	else:
		return name, name + ':0'

def chunk(lst,n):
	for i in xrange(0,len(lst),n):
		yield lst[i:i+n]

def load_weights(path,sess):
	tf.global_variables_initializer().run()
	weights = np.load(path)
	params = {get_op_tensor_name(n)[1]:v for n,v in dict(weights).iteritems()}
	param_names = set(params.iterkeys())

	variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	variable_names = set([k.name for k in variables])

	intersect = variable_names & param_names

	restore_vars = [v for v in variables if v.name in intersect]

	with sess.as_default():
		for v in restore_vars:
			vname = v.name
			v.load(params[vname])
	print "restored %s vars "%(len(restore_vars))

if __name__ == "__main__":
	args = get_args()

	if not os.path.exists(args.featpath):
		os.makedirs(args.featpath)

	image_input,outfeat = build_model(depth=args.depth)

	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary gpu, not take all
	with tf.Session(config=tfconfig) as sess:
		# restore the cnn weights

		load_weights(args.weights,sess)

		imgpaths = glob(os.path.join(args.imgpath,"*.jpg"))

		for imgs in tqdm(chunk(imgpaths,args.batch_size),total=int(math.ceil(len(imgpaths)/float(args.batch_size)))):

			if args.skip:
				allExists = True
				for img in imgs:
					imgname = os.path.splitext(os.path.basename(img))[0]
					target = os.path.join(args.featpath,"%s.npy"%imgname)
					if not os.path.exists(target):
						allExists=False
						break
				if allExists:
					continue

			bs = len(imgs)
			h = w = args.resize
			# load the image
			imgfeats = np.zeros((bs,h,w,3),dtype="float")
			imgnames = []
			for i,img in enumerate(imgs):
				imgnames.append(os.path.splitext(os.path.basename(img))[0])
				im = cv2.imread(img,cv2.IMREAD_COLOR)
				im = cv2.resize(im,(w,h),interpolation=cv2.INTER_LINEAR)
				imgfeats[i,:,:,:] = im

			feed_dict = {}
			feed_dict[image_input] = imgfeats
			# (14,14,2048)
			output, = sess.run([outfeat],feed_dict=feed_dict)
			#print output.shape
			#sys.exit()

			for i,imgfeat in enumerate(output):
				imgname = imgnames[i]
				target = os.path.join(args.featpath,"%s.npy"%imgname)
				np.save(target,output[i])
