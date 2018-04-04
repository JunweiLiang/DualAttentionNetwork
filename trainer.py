# coding=utf-8
# trainer class, given the model (model has the function to get_loss())



import tensorflow as tf


class Trainer():
	def __init__(self,model,config):
		self.config = config
		self.model = model # this is an model instance		

		self.global_step = model.global_step # 

		learning_rate = config.init_lr

		if config.learning_rate_decay is not None:
			learning_rate = tf.train.exponential_decay(
			  config.init_lr,
			  self.global_step * config.batch_size,
			  config.learning_rate_decay_examples, # decay every k samples used in training
			  config.learning_rate_decay,
			  staircase=True)

		#self.opt = tf.train.AdadeltaOptimizer(learning_rate) # only adadelta works
		#self.opt = tf.train.MomentumOptimizer(learning_rate,momentum=0.9)
		#self.opt = tf.train.AdamOptimizer(learning_rate)
		if config.optimizer == "momentum":
			opt_emb = tf.train.MomentumOptimizer(learning_rate*100.0,momentum=0.9)
			opt_rest = tf.train.MomentumOptimizer(learning_rate,momentum=0.9)
		elif config.optimizer == "adadelta":
			opt_emb = tf.train.AdadeltaOptimizer(learning_rate*100.0)
			opt_rest = tf.train.AdadeltaOptimizer(learning_rate)
		elif config.optimizer == "adam":
			opt_emb = tf.train.AdamOptimizer(learning_rate*100.0)
			opt_rest = tf.train.AdamOptimizer(learning_rate)
		else:
			raise Exception("Optimizer not implemented")

		self.loss = model.loss # get the loss funcion

		# valist for embding layer
		var_emb = [var for var in tf.trainable_variables() if var.name.startswith("emb/")]
		var_rest = [var for var in tf.trainable_variables() if not var.name.startswith("emb/")]

		# for training, we get the gradients first, then apply them
		#self.grads = tf.gradients(self.loss,var_emb+var_rest) # will train all trainable in Graph
		self.grads = tf.gradients(self.loss,var_emb+var_rest)
		#print self.grads[0]
		#config.clip_gradient_norm = 1
		if config.clip_gradient_norm is not None:
			# this is from opt.compute_gradients
			#self.grads = [(tf.clip_by_value(grad, -1*config.clip_gradient_norm, config.clip_gradient_norm), var) for grad, var in self.grads]
			self.grads = [tf.clip_by_value(grad, -1*config.clip_gradient_norm, config.clip_gradient_norm) for grad in self.grads]

		grads_emb = self.grads[:len(var_emb)]
		grads_rest = self.grads[len(var_emb):]

		# process gradients
		#self.train_op = self.opt.apply_gradients(self.grads,global_step=self.global_step)

		train_emb = opt_emb.apply_gradients(zip(grads_emb,var_emb))
		train_rest = opt_rest.apply_gradients(zip(grads_rest,var_rest),global_step=self.global_step)
		self.train_op = tf.group(train_emb,train_rest)


	def step(self,sess,batch): # we train 100 step then we save model
		assert isinstance(sess,tf.Session)
		# idxs is a tuple (23,123,33..) index for sample
		batchIdx,batch_data = batch
		feed_dict = self.model.get_feed_dict(batch_data,is_train=True)
		loss, train_op = sess.run([self.loss,self.train_op],feed_dict=feed_dict)
		return loss, train_op

