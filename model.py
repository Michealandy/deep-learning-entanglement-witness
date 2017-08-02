import csv
import numpy as np
import tensorflow as tf
from functions import *

print("Separating train and test data...")

# Loaging data and separating features from labels
features = csv_to_array('features.csv',"complex")
labels = csv_to_array('labels.txt',"int")

features = np.column_stack((features.real, features.imag))

features_labels = np.c_[features, labels]
np.random.shuffle(features_labels)

# Dividing dataset in 80% train set and 20% test set
num_features = int(len(features[0]))
num_labels = int(len(labels[0]))
n = int(0.8*len(features_labels))
x_train, y_train = features_labels[:n,:num_features], features_labels[:n,num_features:]
x_test, y_test = features_labels[n:,:num_features], features_labels[n:,num_features:]

######################################################################################
# Here the learning process effectively begins
# Using a 3-layered Neural Network, each with 3 neurons initialized with a random 
# Gaussian distribution
######################################################################################

# Network Parameters
num_features = int(len(features[0]))
num_labels = int(len(labels[0]))
num_h1 = 32

print("Starting Learning Process")
tf.reset_default_graph()

# Defining Input, Weights and Biase
with tf.name_scope('input'):
	x = tf.placeholder("float", shape = [None, num_features], name = "density_matrix")
	y = tf.placeholder("float", shape = [None, num_labels], name = "entanglement_witness")

with tf.name_scope('weights'):
	w1 = tf.Variable(tf.random_normal([num_features,num_h1]), name = 'w1')
	w2 = tf.Variable(tf.random_normal([num_h1,num_labels]), name = 'w2')

with tf.name_scope('biases'):
	b1 = tf.Variable(tf.random_normal([1,num_h1]), name = 'b1')
	b2 = tf.Variable(tf.random_normal([1,num_labels]), name = 'b2')

# Multilayered Neural Network
with tf.name_scope('layer_1'):
	layer_1 = tf.nn.tanh(tf.add(tf.matmul(x,w1), b1))
with tf.name_scope('layer_2'):
	layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1,w2), b2))

# Training Parameters
num_iter = int(1e3)
batch_size = 4
Lambda = 1e-2
learning_rate = 1e-2

y_ = tf.add(tf.matmul(layer_1,w2), b2)

# Regularization
with tf.name_scope('regularization'):
	regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)

# Loss function
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.square(y_ - y)) + Lambda*regularization
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_, labels = y))
	loss = tf.Print(loss, [loss], "loss_")

# Optimization
with tf.name_scope('train'):
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	#summary_op = tf.summary.merge_all()

with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Lauching Model
init = tf.global_variables_initializer()
#saver = tf.train.Saver()

with tf.Session() as session:
	#writer = tf.summary.FileWriter('tensorboard', graph = tf.get_default_graph())
	session.run(init)

	for i in range(num_iter):
		offset = (i * batch_size) % (len(x_train) - batch_size)
		batch_x = x_train[offset:(offset+batch_size),:]
		batch_y = y_train[offset:(offset+batch_size),:]

		_, current_loss = session.run([train_op, loss], feed_dict = {x: batch_x, y: batch_y})
		#writer.add_summary(summary, i)
		if i%50 ==0 or i == num_iter-1:
			train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
			print("i: {}, Current Loss: {:0.3f}, Current Accuracy: {:0.3f}".format(i, current_loss, train_accuracy))

	# Testing the network
	print("Prediction")
	print("Accuracy: {:0.3f}".format(accuracy.eval(feed_dict = {x: x_test, y: y_test})))

writer = tf.summary.FileWriter('tensorboard', graph = tf.get_default_graph())