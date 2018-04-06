import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 100
K = 10
M = 40000
dt = 0.005
T = M*dt

inp = tf.placeholder(dtype = tf.float32, shape = (None, 1))

x_training = np.random.uniform(-4,4,size=(N,1))
x_test = np.random.uniform(-4,4,size=(100,1))
target_fcn = tf.square(tf.abs(inp - 0.5))

def neural_network_model(inp):
    initializer = tf.initializers.random_normal
    activation_fcn_1 = tf.nn.sigmoid
    activation_fcn_output = None
    num_output_nodes = 1
    
    hidden_layer_1 = tf.layers.dense(inputs=inp,
                                     units=K,
                                     activation=activation_fcn_1,
                                     kernel_initializer=initializer())
    
    output = tf.layers.dense(inputs=hidden_layer_1,
                             units=num_output_nodes,
                             activation=activation_fcn_output,
                             use_bias=False,
                             kernel_initializer=initializer())
    return output

def train_neural_network(sess):
    alpha = neural_network_model(inp)
    cost = tf.reduce_mean(tf.square(alpha - target_fcn))
    optimizer = tf.train.GradientDescentOptimizer(dt)
    train = optimizer.minimize(cost)
    E_1 = []
    
    sess.run(tf.global_variables_initializer())
    
    for m in range(0, M):
        n = np.random.randint(0, N)
        x_in = x_training[n:n + 1]
        sess.run(train, feed_dict = {inp:x_in})
        
        if m % int(M/100) == 0:
            E_1.append(cost.eval(feed_dict={inp:x_test}))
        if m % int(M/10) == 0:
            print(m)
            
    return alpha, E_1

pts = np.linspace(-4,4,300).reshape(-1,1)

with tf.Session() as sess:
    alpha, E_1 = train_neural_network(sess)
    training_data_vals = target_fcn.eval(feed_dict={inp:x_training})
    target_fcn_vals = target_fcn.eval(feed_dict={inp:pts})
    alpha_vals = alpha.eval(feed_dict={inp:pts})

plt.figure('Learned function', figsize=(15,10));
plt.plot(x_training, training_data_vals, 'o', label='Training data')
plt.plot(pts, target_fcn_vals, label='Target function')
plt.plot(pts, alpha_vals, label='alpha')
plt.legend()
plt.show()

plt.figure('Test error', figsize=(15,10));
plt.semilogy(np.linspace(0, T, len(E_1)), E_1, label='E_1')
plt.legend()
plt.show()