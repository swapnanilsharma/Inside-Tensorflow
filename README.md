# Inside-Tensorflow



**ReLU implementation, h = ReLU(W*x + b)


1. Create weights, including initialization
		W ~ Uniform(-1, 1); b=0
2. Create input placeholder x
		m * 784 imput matrix
3. Build flow graph



```
import tensorflow as tf

b=tf.Variable(tf.zeros((100,)))
W=tf.Variable(tf.random_uniform((784, 100), -1, 1))

x=tf.placeholder(tf.float32, (100, 784))

h=tf.nn.relu(tf.matmul(x, W) + b)
```


