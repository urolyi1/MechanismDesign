import tensorflow as tf
import numpy as np


def compute_utility(allocation):
    tf.reduce_sum(tf.multiply(alloc, allowed), axis=-1)

learn_rate = 0.01
n_hos = 3
n_types = 16

n_features = n_hos * n_types
n_out = n_hos * n_types * n_types 

init = tf.keras.initializers.glorot_uniform()

num_layers = 5

neurons = [n_features, 20, 20, 20, n_out]

weights = []
biases = []

activation = [tf.nn.relu] * (num_layers - 1) + [tf.nn.relu]

# Creating input layer weights
print("Creating weights for layer: 1")
print([neurons[0], neurons[1]])
weights.append(tf.compat.v1.get_variable('w_a_1', [neurons[0], neurons[1]], initializer=init))


# Creating hidden layers
for i in range(1, num_layers - 2):
    print("Creating weights for layer: {}".format(i+1))
    print([neurons[i], neurons[i + 1]])
    weights.append(tf.compat.v1.get_variable('w_a_' + str(i+1), [neurons[i], neurons[i + 1]], initializer=init))

# need two outputs
print("Creating weights for layer: {}".format(num_layers - 1))
print([neurons[-2], neurons[-1]])
weights.append(tf.compat.v1.get_variable('w_a_' + str(num_layers - 1), [neurons[-2], neurons[-1]], initializer=init))
w_prime = tf.compat.v1.get_variable('wi_a_' + str(num_layers - 1), [neurons[-2], neurons[-1]], initializer=init)


# Biases
for i in range(num_layers - 2):
    print("Creating biases for layer: {}".format(i+1))
    biases.append(tf.compat.v1.get_variable('b_a_' + str(i+1), [neurons[i + 1]], initializer=init))

print("Creating biases for layer: {}".format(num_layers - 1))
biases.append(tf.compat.v1.get_variable('b_a_' + str(num_layers - 1), [neurons[-1]], initializer=init))
b_prime = tf.compat.v1.get_variable('bi_a_' + str(num_layers - 1), [neurons[-1]], initializer=init)


# Putting it together to get output
X = tf.compat.v1.placeholder(tf.float32, [None, n_features], name='features')
x_in = tf.reshape(X, [-1, neurons[0]])

a = tf.nn.relu(tf.matmul(X, weights[0]) + biases[0], 'alloc_act_0')

for i in range(1, num_layers - 2):
    a = tf.matmul(a, weights[i]) + biases[i]
    a = tf.nn.relu(a, 'alloc_act_' + str(i))

pair = tf.matmul(a, weights[-1]) + biases[-1]
pool = tf.matmul(a, w_prime) + b_prime

# Softmax over pairs and total pool
pair = tf.reshape(tf.nn.softmax(tf.reshape(pair, [-1, n_types]), axis=1), [-1, n_types, n_types])
pool = tf.reshape(tf.nn.softmax(tf.reshape(pool, [-1, n_types]), axis=0), [-1, n_types, n_types])

# Weight softmax values
pair = tf.math.multiply(pair, tf.reshape(X, [n_hos, n_types, 1]))
pool = tf.math.multiply(pool, tf.math.reduce_sum(tf.reshape(X, [n_hos, n_types]), axis=0))

# Minimum of two allocations
alloc = tf.math.floor(tf.math.minimum(pair, pool))

# optimizer = tf.train.AdamOptimizer(learn_rate)
init_op = tf.compat.v1.global_variables_initializer()

test_in = np.random.randint(1, 100, size=(1, n_features))
sess = tf.compat.v1.InteractiveSession()
writer = tf.compat.v1.summary.FileWriter('./graphs', sess.graph)
sess.run(init_op)
example_alloc = sess.run(alloc, feed_dict={X:test_in})
print(example_alloc)
writer.close()
