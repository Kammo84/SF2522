import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

K = 500
M = 5000
dt = 1
N = 100
n_pixels = 784
T = M*dt

inp = tf.placeholder(dtype = tf.float32, shape = (None, n_pixels))
label = tf.placeholder(dtype = tf.float32, shape = (None, 10))

def neural_network_model(inp):
    initializer = tf.initializers.random_normal
    activation_fcn_1 = tf.nn.sigmoid
    activation_fcn_output = None
    num_output_nodes = 10
    
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
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=alpha, labels=label))
    optimizer = tf.train.GradientDescentOptimizer(dt)
    train = optimizer.minimize(cost)
    E_1 = []
    
    sess.run(tf.global_variables_initializer())
    for m in range(0, M):
        train_images, train_labels = mnist.train.next_batch(N)
        sess.run(train, feed_dict={inp: train_images, label: train_labels})
        if m % int(M/10) == 0:
            print(m)
        if m % int(M/100) == 0:
            E_1.append(cost.eval(feed_dict={inp: mnist.test.images, label: mnist.test.labels}))
    return alpha, E_1

def evaluate_model(sess, alpha):
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    
    alpha_is_correct = tf.equal(tf.argmax(alpha, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(alpha_is_correct, tf.float32))
    alpha_values = alpha.eval({inp:test_images, label:test_labels})
    
    index_first_mistaken_number = alpha_is_correct.eval({inp:test_images, label:test_labels}).tolist().index(False)
    mistaken_alpha = tf.argmax(alpha_values[index_first_mistaken_number,:])
    mistaken_label = tf.argmax(test_labels[index_first_mistaken_number,:])
    
    print('Accuracy:',accuracy.eval({inp:test_images, label:test_labels}))
    print('The first mistaken number')
    print('-------------------------')
    print('Alpha:', mistaken_alpha.eval())
    print('Label:', mistaken_label.eval())
    
    mistaken_number = test_images[index_first_mistaken_number,:].reshape((28,28))
    plt.imshow(mistaken_number, cmap='gray')

with tf.Session() as sess:
    alpha, E_1 = train_neural_network(sess)
    evaluate_model(sess, alpha)

plt.figure('Test error', figsize=(15,10));
plt.semilogy(np.linspace(0, T, len(E_1)), E_1, label='E_1')
plt.legend()
plt.show()