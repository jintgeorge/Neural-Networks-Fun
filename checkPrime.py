# Check for Prime Number in Tensorflow!
# I got approximately 75% accuracy. Feel free to let me know if you find anything wrong 
# or ways the performance can be improved

#Inspired by Joel Grus (http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/)

import numpy as np
import tensorflow as tf
from math import sqrt
from itertools import count, islice

NUM_DIGITS = 10

# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def isPrime(n):
    return n > 1 and all(n%i for i in islice(count(2), int(sqrt(n)-1)))

# One-hot encode the desired outputs: [number, "prime"]
def encodeIsPrime(n):
    if isPrime(n): return np.array([0,1])
    else: return np.array([1, 0])

# Produce synthetic Training data for numbers from 101 tp 1024 
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([encodeIsPrime(i)             for i in range(101, 2 ** NUM_DIGITS)])

# Randomly initialize weights.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Our model is a standard 1-hidden-layer multi-layer-perceptron with ReLU
# activation. The softmax (which turns arbitrary real-valued outputs into
# probabilities) gets applied in the cost function.
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

TARGET_SIZE = 2

# Our variables. The input has width NUM_DIGITS, and the output has width 2.(Prime or NotPrime)
X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, TARGET_SIZE])

# How many units in the hidden layer.
NUM_HIDDEN = 100


# Initialize the weights.
w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, TARGET_SIZE])

# Predict y given x using the model.
py_x = model(X, w_h, w_o)

# We'll train our model by minimizing a cost function. 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(py_x, 1)

# Finally, we need a way to turn a prediction (and an original number)
# into a fizz buzz output
def Prime(i, prediction):
    return [str(i), "Prime"][prediction]

BATCH_SIZE = 128

# Launch the graph in a session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch in range(10000):
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        # Train in batches of 128 inputs.
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        # And print the current accuracy on the training data.
        print(epoch, np.mean(np.argmax(trY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: trX, Y: trY})))

    # And now for real test (Test for number from 1 - 100)
    numbers = np.arange(1, 101)
    teX = np.transpose(binary_encode(numbers, NUM_DIGITS)) #testX
    teY = sess.run(predict_op, feed_dict={X: teX}) #testY
    output = np.vectorize(Prime)(numbers, teY)
    
    y1 = np.array([1 if i == "Prime" else 0 for i in output])
    y2 = np.array([1 if isPrime(i) else 0 for i in numbers])
    print(output)
    print('Accuracy = ', np.sum(y1==y2), '%')
