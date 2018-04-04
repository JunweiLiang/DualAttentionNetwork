# coding=utf-8
# tester, given the config with model path


import tensorflow as tf
import numpy as np

class Tester():
	def __init__(self,model,config,sess=None):
		self.config = config
		self.model = model

		self.z_u = self.model.z_u # the output of the embedding for text
		self.z_v = self.model.z_v # the output of the embedding for images


	def step(self,sess,batch):
		# give one batch of Dataset, use model to get the result,
		assert isinstance(sess,tf.Session)
		batchIdxs,batch_data = batch
		feed_dict = self.model.get_feed_dict(batch_data,is_train=False)
		z_u,z_v = sess.run([self.z_u,self.z_v],feed_dict=feed_dict)
		# clip the output
		#print z_v.shape,z_u.shape
		z_u = z_u[:batch_data.num_examples]
		z_v = z_v[:len(batch_data.data['imgs'])]
		return z_u,z_v
