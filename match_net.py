import tensorflow as tf
import numpy as np

class SingleHospital:
    def __init__(self, n_types, dist_lst):
        ''' Takes in number of pair types along with a list of functions that
        generate the number of people in that hospital with pair type.
        '''
        self.n_types = n_types
        self.dists = dist_lst
    def generate(self, batch_size):
        '''generate a report from this hospital'''
        X = np.zeros((batch_size, self.n_types))
        for i, dist in enumerate(self.dists):
            X[:, i] = dist(size=batch_size)
        return X
        
class ReportGenerator:
    def __init__(self, hos_lst, single_report_shape):
        self.hospitals = hos_lst
        self.single_shape = single_report_shape
    def generate_report(self, batch_size):
        X = np.zeros((batch_size,) + self.single_shape)
        for i, hos in enumerate(self.hospitals):
            X[:, i, :] = hos.generate(batch_size)
        yield X

def randint(low, high):
    return lambda size: np.random.randint(low, high, size)

def create_simple_generator(low_lst_lst, high_lst_lst, n_hos, n_types):
    hos_lst = []
    for h in range(n_hos):
        tmp_dist_lst = []
        for t in range(n_types):
            tmp_dist_lst.append(randint(low_lst_lst[h][t], high_lst_lst[h][t]))
        hos_lst.append(SingleHospital(n_types, tmp_dist_lst))
    gen = ReportGenerator(hos_lst, (n_hos, n_types))
    return gen


low_lst_lst = [[5, 10],[20, 40],[40, 80]]
high_lst_lst = [[10, 20], [40, 80], [80, 160]]
gen = create_simple_generator(low_lst_lst, high_lst_lst, 3, 2)

def create_u_mask(compat_lst, n_types, n_hos):
    '''
    Create mask matrix that will only reward valid matchings making the rest 0
    return shape (n_hos, n_types, n_types)
    '''
    mask = np.zeros([n_hos, n_types, n_types])
    print(mask.shape)
    for t1, t2 in compat_lst:
        mask[:, t1, t2] = 1.0
        mask[:, t2, t1] = 1.0
    return mask

learn_rate = 0.01
n_hos = 3
n_types = 2
batch_size = 10

n_features = n_hos * n_types
n_out = n_hos * n_types * n_types 

init = tf.keras.initializers.glorot_uniform()

num_layers = 5

neurons = [n_features, 100, 100, 100, n_out]

weights = []
biases = []

activation = [tf.nn.tanh] * (num_layers - 1) + [tf.nn.tanh]

# Creating input layer weights
print("Creating weights for layer: 1")
print([neurons[0], neurons[1]])
weights.append(tf.compat.v1.get_variable('w_a_1', [neurons[0], neurons[1]], initializer=init, dtype=tf.float64))


# Creating hidden layers
for i in range(1, num_layers - 2):
    print("Creating weights for layer: {}".format(i+1))
    print([neurons[i], neurons[i + 1]])
    weights.append(tf.compat.v1.get_variable('w_a_' + str(i+1), [neurons[i], neurons[i + 1]], initializer=init, dtype=tf.float64))

# need two outputs
print("Creating weights for layer: {}".format(num_layers - 1))
print([neurons[-2], neurons[-1]])
weights.append(tf.compat.v1.get_variable('w_a_' + str(num_layers - 1), [neurons[-2], neurons[-1]], initializer=init, dtype=tf.float64))
w_prime = tf.compat.v1.get_variable('wi_a_' + str(num_layers - 1), [neurons[-2], neurons[-1]], initializer=init, dtype=tf.float64)


# Biases
for i in range(num_layers - 2):
    print("Creating biases for layer: {}".format(i+1))
    biases.append(tf.compat.v1.get_variable('b_a_' + str(i+1), [neurons[i + 1]], initializer=init, dtype=tf.float64))

print("Creating biases for layer: {}".format(num_layers - 1))
biases.append(tf.compat.v1.get_variable('b_a_' + str(num_layers - 1), [neurons[-1]], initializer=init, dtype=tf.float64))
b_prime = tf.compat.v1.get_variable('bi_a_' + str(num_layers - 1), [neurons[-1]], initializer=init, dtype=tf.float64)

def feedforward(X):
    x_in = tf.reshape(X, [-1, neurons[0]])

    a = tf.nn.relu(tf.matmul(x_in, weights[0]) + biases[0], 'alloc_act_0')

    # push through hidden layers
    for i in range(1, num_layers - 2):
        a = tf.matmul(a, weights[i]) + biases[i]
        a = tf.nn.relu(a, 'alloc_act_' + str(i))

    # final layer
    pair = tf.matmul(a, weights[-1]) + biases[-1]
    pool = tf.matmul(a, w_prime) + b_prime

    # Softmax over pairs and total pool
    pair = tf.reshape(tf.nn.softmax(tf.reshape(pair, [-1, n_types * n_hos, n_types]), axis=-1), [-1, n_hos, n_types, n_types])
    pool = tf.reshape(tf.nn.softmax(tf.reshape(pool, [-1, n_types * n_hos, n_types]), axis=1), [-1, n_hos, n_types, n_types])

    ## Weighting softmax values ##

    # Weight softmax values by hospital's reported needs
    pair = tf.math.multiply(pair, tf.reshape(X, [-1, n_hos, n_types, 1]))

    # Sums over each hospitals for for all pair types then reshapes to allow broadcasting
    tot_reshaped = tf.reshape(tf.math.reduce_sum(tf.reshape(X, [-1, n_hos, n_types]), axis=1), [-1, 1, 1, n_types])

    # Weight softmax values by total available pairs in pool
    pool = tf.math.multiply(pool, tot_reshaped)

    # Floor of minimum of two allocations
    alloc = tf.math.floor(tf.math.minimum(pair, pool))
    return alloc
def create_misreports(x, curr_mis, self_mask):
    '''
    x and curr_mis should have dimensions (B, h, p)
    combined and og has dimensions (n_hos, batch_size, n_hos, n_types)
    '''
    tiled_curr_mis = tf.tile(tf.expand_dims(curr_mis, 0), [n_hos, 1, 1, 1])
    og_tiled = tf.reshape(tf.tile(x, [n_hos, 1, 1]), [n_hos, -1, n_hos, n_types])
    
    # Filter original reports to only keep reports from all other hospitals
    other_hos = og_tiled * (1 - self_mask)
    
    # Filter reports to only keep the mis reports
    only_mis = tiled_curr_mis * self_mask
    
    # Add the two to get inputs where only one hospital misreports
    combined = only_mis + other_hos
    return combined, og_tiled

def best_sample_misreport(x, alt_sample, self_mask, alt_sample_size):
    # Expand x and tile for each sample and hos
    expanded_x = tf.expand_dims(tf.expand_dims(x, 0), 0)
    tiled_x = tf.tile(expanded_x, [alt_sample_size, n_hos , 1, 1, 1])
    
    # Expand and tile sample for each sample and hos
    expanded_sample = tf.expand_dims(tf.expand_dims(alt_sample, 1), 1)
    tiled_sample = tf.tile(expanded_sample, [1, n_hos, batch_size, 1, 1])
    
    
    # Only keep the specific hospitals misreport
    only_mis = tf.math.maximum(tiled_sample * self_mask, tiled_x * self_mask)
    # Keep other hospitals reports the same
    only_other = tiled_x * (1 - self_mask)
    
    combined = only_mis + only_other
    return combined, tiled_x

def compute_sample_util(alloc, alloc_mask, mis_mask, sample_internal):
    # Calculate utility from mechanism
    mech_util = tf.reduce_sum(tf.multiply(alloc, alloc_mask), axis=(2, 3))
    mech_util = tf.reshape(mech_util, [-1, n_hos, batch_size, n_hos])
    
    # Calculate utility from internal matchings
    internal_util = compute_internal_util(sample_internal)
    
    return (mech_util + internal_util) # [alt_size, n_hos, batch_size, n_hos]

def get_best_mis(tot_sample_util, mis_mask):
    hos_util = tot_sample_util * mis_mask
    return tf.reduce_max(hos_util, axis=0) # will only keep the best misreport
    
def compute_util(alloc, mask, internal=None):
    if internal is None:
        return tf.reduce_sum(tf.multiply(alloc, mask), axis=(2, 3))
    else:
        mech_util = tf.reshape(tf.reduce_sum(tf.multiply(alloc, mask), axis=(2, 3)), [n_hos, -1, n_hos])
        return mech_util + compute_internal_util(internal)
def compute_internal_util(internal):
    '''
    internal has dim [n_hos, batch_size, n_hos, n_types]
    '''
    return tf.reduce_min(internal, axis=-1) # for two types, utility is just the minimum of two types
def compute_internal(misreports, og):
    ''' Computes the difference between '''
    return (og - misreports) #maybe just keep the specific misreport bidder's difference

# This mask is to manipulate the reports and will only have 1 for the specific hospital misreporting
self_mask = np.zeros([n_hos, batch_size, n_hos, n_types])
self_mask[np.arange(n_hos), :, np.arange(n_hos), :] = 1.0

# This mask will only count the utility from the specific hospital that is misreporting
mis_u_mask = np.zeros((n_hos, batch_size, n_hos))
mis_u_mask[np.arange(n_hos), :, np.arange(n_hos)] = 1.0

# Mask to only count valid matchings 
u_mask = create_u_mask([(0,1)], n_types, n_hos)

## Sample Based Algorithm ##
# TODO: test regret max stuff, check that misreports are not overreporting, lagrange

alt_sample_size = 10000
X = tf.compat.v1.placeholder(tf.float64, [batch_size, n_hos * n_types], name='features')
alt_sample = np.reshape(next(gen.generate_report(alt_sample_size)), (alt_sample_size, n_hos, n_types))

misreports, og = best_sample_misreport(tf.reshape(X, [batch_size, n_hos, n_types]),
                                       alt_sample, self_mask, alt_sample_size)
non_reported = compute_internal(misreports, og)

# Get feedforward output
actual_alloc = feedforward(X)

# Get misreport alloc
mis_alloc = feedforward(tf.reshape(misreports, [-1, n_hos* n_types]))

# Calculate Utilities
util = compute_util(actual_alloc, u_mask) # [batch_size, n_hos]
mis_util = compute_sample_util(mis_alloc, u_mask, mis_u_mask, non_reported)
best_mis = get_best_mis(mis_util, mis_u_mask) # [n_hos, batch_size, n_hos]

mis_diff = tf.nn.relu(tf.tile(tf.expand_dims(util, 0), [n_hos, 1, 1]) - best_mis)
rgt = tf.reduce_mean(tf.reduce_max(mis_diff, axis=0), axis=0) # reduce max since only utility is from bidder i 


# optimizer = tf.train.AdamOptimizer(learn_rate)
init_op = tf.compat.v1.global_variables_initializer()

#test_in = np.random.randint(1, 100, size=(10, n_features))
sess = tf.compat.v1.InteractiveSession()

writer = tf.compat.v1.summary.FileWriter('./graphs', sess.graph)
sess.run(init_op)

test_in = np.reshape(next(gen.generate_report(batch_size)), (10, -1))
ex_util = sess.run(util, feed_dict={X:test_in})
ex_best_util = sess.run(best_mis, feed_dict={X:test_in})
ex_rgt = sess.run(rgt, feed_dict={X:test_in})