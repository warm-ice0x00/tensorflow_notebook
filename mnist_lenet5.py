import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
from PIL import Image

INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=800

BATCH_SIZE=100
LEARNING_RATE_BASE=0.005
LEARNING_RATE_DECAY=0.99
REGULARIZER=0.0002
STEPS=50000
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH="./model5/"
MODEL_NAME="mnist_model5"

IMG_SIZE=28
NUM_CHANNELS=1
CONV1_SIZE=5
CONV1_KERNEL_NUM=32
CONV2_SIZE=5
CONV2_KERNEL_NUM=64
FC_SIZE=512

#TEST_INTERVAL_SECS=5

def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

def conv2d(x,w):
        return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def forward(x,train, regularizer):
        conv1_w=get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
	conv1_b=get_bias([CONV1_KERNEL_NUM])
	conv1=conv2d(x, conv1_w)
	relu1=tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
	pool1=max_pool_2x2(relu1)
	
	conv2_w=get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
	conv2_b=get_bias([CONV2_KERNEL_NUM])
	conv2=conv2d(pool1, conv2_w)
	relu2=tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
	pool2=max_pool_2x2(relu2)
	
	pool_shape=pool2.get_shape().as_list()
	nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
	reshaped=tf.reshape(pool2,[pool_shape[0],nodes])
	
	fcl_w=get_weight([nodes,FC_SIZE],regularizer)
	fcl_b=get_bias([FC_SIZE])
	fc1=tf.nn.relu(tf.matmul(reshaped,fcl_w)+fcl_b)
        if train:fc1 = tf.nn.dropout(fc1,0.5)
	
	fc2_w=get_weight([FC_SIZE,OUTPUT_NODE],regularizer)
	fc2_b=get_bias([OUTPUT_NODE])
	y=tf.matmul(fc1, fc2_w) + fc2_b
	return y

def backward(mnist):
	x = tf.placeholder(tf.float32, [BATCH_SIZE,IMG_SIZE,IMG_SIZE,NUM_CHANNELS])
	y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
	y = forward(x,True, REGULARIZER)
	global_step = tf.Variable(0, trainable=False)
	
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	cem = tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection("losses"))
	
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples / BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase=True)
		
	train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss)
	
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name="train")
		
	saver=tf.train.Saver()
	
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		for i in range(STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			reshaped_xs=np.reshape(xs,(BATCH_SIZE,IMG_SIZE,IMG_SIZE,NUM_CHANNELS))
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:reshaped_xs, y_:ys})
			if i%100 == 0:
				print("After %s steps, loss on training batch is %g." % (i, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                                test(mnist)

def test(mnist):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32, [mnist.test.num_examples,IMG_SIZE,IMG_SIZE,NUM_CHANNELS])
		y_=tf.placeholder(tf.float32, [None, OUTPUT_NODE])
		y=forward(x,False,None)
		
		ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
		ema_restore=ema.variables_to_restore()
		saver=tf.train.Saver(ema_restore)
		
		correct_prediction=tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                                reshaped_x=np.reshape(mnist.test.images,(mnist.test.num_examples,IMG_SIZE,IMG_SIZE,NUM_CHANNELS))
				accuracy_score = sess.run(accuracy, feed_dict={x:reshaped_x, y_:mnist.test.labels})
				print("The test accuracy is %g." % (accuracy_score))
			else:
				print("No checkpoint file found")
				return
				
def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28,28),Image.LANCZOS)
	im_arr = np.array(reIm.convert('L'))
	threshold = 30
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255 - im_arr[i][j]
			if(im_arr[i][j]<threshold):
				im_arr[i][j]=0
			else: im_arr[i][j]=255
			
	nm_arr = im_arr.reshape([1,28,28,1])
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr, 1.0/255.0)
	#print(img_ready)
	return img_ready

def application():
	testNum = input("Number of test pictures:")
	for i in range(testNum):
		testPic = raw_input("Path of test image:")
		testPicArr = pre_pic(testPic)
		preValue = restore_model(testPicArr)
		print("The prediction number is " + str(preValue))

def restore_model(testPicArr):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [1,IMG_SIZE,IMG_SIZE,NUM_CHANNELS])
		y = forward(x, False, None)
		preValue = tf.argmax(y, 1)
		
		variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)
		
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
                                #reshaped_x=np.reshape(x,(1,IMG_SIZE,IMG_SIZE,NUM_CHANNELS))
				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
			else:
				print("No checkpoint file found!")
				return -1
				
def main():
    if raw_input("Type \"train\" to train, or press ENTER for application:")=='train':
		mnist = input_data.read_data_sets("./data/", one_hot=True)
		backward(mnist)
    else:
                application()
	
if __name__=='__main__':
	main()
