'''
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import cv2
import numpy as np

# Import MNIST data

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

num_classes = 2

######################################################################################
count = 1000
validation_size = 100
train_images = []
train_labels = []
validation_images = []
validation_labels = []
label_doors = []
label_windows = []
image_doors = []
image_windows = []
temp = np.zeros([1800,2])


for i in range (0,count):
    #img = Image.open('C:\\Users\\Kestutis\\Desktop\\Images_HW11\\Train\\Cats\\cat.'+str(i)+'.jpg')
    #img = 'C:\\Users\\Kestutis\\Desktop\\Images_HW11\\Train\\Cats\\cat.'+str(i)+'.jpg'
#    img = '../images/doors/door.'+str(i)+'.jpg'
#    img = '/home/chad/ECE479/ece479hw6/images/doors/door.'+str(i)+'.jpg'
    img = '../images/doors/door.'+str(i)+'.jpg'
    if i > validation_size-1:
        train_images.append(img)
        train_labels.append(0)
        label_doors.append(0)
        image_doors.append(img)
    else:
        validation_images.append(img)
        validation_labels.append(0)
for i in range (0,count):
    #img = Image.open('C:\\Users\\Kestutis\\Desktop\\Images_HW11\\Train\\Dogs\\dog.'+str(i)+'.jpg')
    #img = 'C:\\Users\\Kestutis\\Desktop\\Images_HW11\\Train\\Dogs\\dog.'+str(i)+'.jpg'
    img = '../images/windows/window.'+str(i)+'.jpg'
#    img = '/home/chad/ECE479/ece479hw6/images/windows/window.'+str(i)+'.jpg'
    if i > validation_size-1:
        train_images.append(img)
        train_labels.append(1)
        label_windows.append(1)
        image_windows.append(img)
    else:
        validation_images.append(img)
        validation_labels.append(1)

label_list = np.hstack((label_doors, label_windows))
image_list = np.hstack((image_doors, image_windows))
final_temp = np.array([image_list, label_list])
final_temp = final_temp.transpose()
np.random.shuffle(final_temp)
image_list = list(final_temp[:,0])
label_list = list(final_temp[:,1])
label_list = [int(i) for i in label_list]
print("here is label_list:")
print(label_list)

for i in range(0, 1800):
    if(train_labels[i] == 0):
        temp[i,0] = 1
    if(train_labels[i] == 1):
        temp[i,1] = 1
train_labels = temp
temp = np.zeros([1800,2])
for i in range(0, 1800):
    if(label_list[i] == 0):
        temp[i,0] = 1
    if(label_list[i] == 1):
        temp[i,1] = 1
label_list = temp
print("images:")

#np.reshape(train_labels, (900, 2))
#print("here is training labels reshaped")
#print(train_labels)
print("here is temp:")
print(label_list)
#train_labels = np.hstack((label_doors, label_windows))
#print("here are the training labels:")
#print(train_labels)
train_labels = temp
train_images = image_list

train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels))

validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

def _parse_function(filename,filelable):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    lable = tf.cast(filelable, tf.float32)
    tf.reshape(image, [-1,50*50])
    return image, lable

train_dataset = train_dataset.map(_parse_function)

train_dataset = train_dataset.batch(1)

iterator = train_dataset.make_one_shot_iterator()

next_element = iterator.get_next()

training_init_op = iterator.make_initializer(train_dataset)
#validation_init_op = iterator.make_initializer(validation_dataset)

'''
with tf.Session() as sess:
    sess.run(training_init_op)
    while True:
        try:
            elem = sess.run(next_element[0])
            print(elem)
        except tf.errors.OutOfRangeError:
            print("end of dataset")
            break
 '''       

######################################################################################



# Parameters
starter_learning_rate = 0.001
global_step = tf.Variable(0)
#global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.95, staircase=True)
training_epochs = 17
batch_size = 128
display_epoch = 1
logs_path = 'tmp/tensorflow_logs/doors_windows/'

# tf Graph Input
# mnist data image of shape 28*28=784
X = tf.placeholder(tf.float32, [None, 50,50,1], name='InputData')
x = tf.reshape(X, [-1,50*50])
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 2], name='LabelData')
#y = tf.placeholder(tf.float32, [1,], name='LabelData')

#Parameters:
K = 500
L = 100
M = 60
N = 30

#Set model weights & bias:
W1 = tf.Variable(tf.random_normal([50*50, K], stddev=0.1), name='Weights')
b1 = tf.Variable(tf.zeros([K]), name='Bias')
W2 = tf.Variable(tf.random_normal([K, L], stddev=0.1), name='Weights')
b2 = tf.Variable(tf.zeros([L]), name='Bias')
W3 = tf.Variable(tf.random_normal([L, M], stddev=0.1), name='Weights')
b3 = tf.Variable(tf.zeros([M]), name='Bias')
W4 = tf.Variable(tf.random_normal([M, N], stddev=0.1), name='Weights')
b4 = tf.Variable(tf.zeros([N]), name='Bias')
W5 = tf.Variable(tf.random_normal([N, 2], stddev=0.1), name='Weights')
b5 = tf.Variable(tf.zeros([2]), name='Bias')

#Activation functions & layer connection:
Y1 = tf.nn.leaky_relu(tf.matmul(x,W1)+b1)
Y2 = tf.nn.leaky_relu(tf.matmul(Y1,W2)+b2)
Y3 = tf.nn.leaky_relu(tf.matmul(Y2,W3)+b3)
Y4 = tf.nn.leaky_relu(tf.matmul(Y3,W4)+b4)
Output = tf.matmul(Y4, W5) + b5

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    pred = tf.nn.softmax(tf.matmul(Y4, W5) + b5) # Softmax
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
   # cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    cost = -tf.reduce_sum(y * tf.log(pred))
with tf.name_scope('SGD'):
    # Gradient Descent
    #optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
   optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    sess.run(training_init_op)
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_images)/batch_size)
        # Loop over all batches
        for i in range(100):            #total_batch
            batch_xs, batch_ys = sess.run(next_element)
           # batch_ys = tf.cast(train_labels[i], tf.string)

            #batch_ys = train_labels[i]

            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                     feed_dict={X:batch_xs, y:batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    sess.run(validation_init_op)
    print("Accuracy:", acc.eval({x: validation_dataset[0], y: validation_dataset[1]}))
    print("Run the command line:\n" \
        "--> tensorboard --logdir=/tmp/tensorflow_logs " \
        "\nThen open http://0.0.0.0:6006/ into your web browser")

