# First model
import tensorflow as tf

# Model parameters
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

# Inputs and outputs
x = tf.placeholder(tf.float32)
linear_model = W * x + b

y = tf.placeholder(tf.float32)

# loss function
square_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(square_delta)

# Optimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
# Before optimization loss
print(sess.run(loss, {x:[1,2,3,4], y:[0, -1, -2, -3]}))

for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0, -1, -2, -3]})
print(sess.run([W, b]))

# After optimization loss
print(sess.run(loss, {x:[1,2,3,4], y:[0, -1, -2, -3]}))

sess.close()
