import tensorflow as tf

# Build computational graph

# Constant
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
print(node1)
print(node2)
c = node1 * node2

# Placeholder
node3 = tf.placeholder(tf.float32)
node4 = tf.placeholder(tf.float32)
print(node3)
print(node4)
addr_node = node3 + node4

# Variable with initialization
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W*x + b
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    output1 = sess.run([node1, node2, c])
    print(output1)
    
    output2 = sess.run(addr_node, {node3:[1,3], node4:[2,4]})
    print(output2)
    
    sess.run(init)
    output3 = sess.run(linear_model, {x:[1,2,3,4]})
    print(output3)
    
    visualizing graph
    File_Writter = tf.summary.FileWriter('C:\\graph', sess.graph)
    
    
  
  
%load_ext tensorboard
%tensorboard --logdir logs
