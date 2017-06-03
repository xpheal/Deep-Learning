import csv
import numpy as np
import sys

def get_features_labels(file):
	examples = []
	with open(file, 'r') as file:
		c_r = csv.reader(file)

		for row in c_r:
			examples += [row]

	examples = np.array(examples)

	features = examples[:,:-1]
	labels = examples[:,-1]

	return features, labels

def generate_mapping(labels, num_class): 
	i = 0
	label_map = {}

	for l in labels:
		if not l in label_map:
			label_map[l] = i
			i += 1

		if len(label_map) == num_class:
			break

	return label_map

def convert_label(labels, label_map):
	return np.array([label_map[l] for l in labels])

def one_hot(labels, num_class):
	arr = np.zeros((labels.shape[0], num_class))
	arr[np.arange(labels.shape[0]), labels] = 1

	return arr

NUM_CLASS = 6
train_f1, train_l1 = get_features_labels('train1.csv')
train_f2, train_l2 = get_features_labels('train2.csv')
train_f = np.concatenate([train_f1])
train_l = np.concatenate([train_l1])
label_map = generate_mapping(train_l, NUM_CLASS)
train_l = convert_label(train_l, label_map)																								
																																	
test_f, test_l = get_features_labels('test.csv')
test_l = convert_label(test_l, label_map)

train_l = one_hot(train_l, NUM_CLASS)
test_l = one_hot(test_l, NUM_CLASS)

print("LOAD TF")
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 30								
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 128 # 2nd layer number of features
n_input = 2067 * 17 # MNIST data input (img shape: 28*28)
n_classes = 6 # MNIST total classes (0-9 dataigits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

patience = 10
min_cost = sys.maxsize
epoch = 0
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    while True:
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: train_f, y: train_l})

        if c < min_cost:
       		min_cost = c
       		patience = 10
        else:
        	patience -= 1

        if patience == 0:
        	break

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(c))

        epoch += 1

        np.random.shuffle(train_f)
        np.random.shuffle(train_l)
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_f, y: test_l}))
