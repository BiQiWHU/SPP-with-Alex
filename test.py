
# coding: utf-8

# In[1]:


from network import alex_net
from tfdata import *
import numpy as np
from training import accuracy_of_batch


# In[2]:


import tensorflow as tf


# In[3]:


# Dataset path
train_tfrecords = 'train.tfrecords'
test_tfrecords = 'test.tfrecords'


# In[4]:


batch_size=20
img,label=input_pipeline(test_tfrecords,batch_size,is_shuffle=False,is_train=False)
with tf.variable_scope('model_definition'):
    prediction=alex_net(img,train=False)
accuracy=accuracy_of_batch(prediction,label)
# confusionmatrix=tf.confusion_matrix(label,prediction)


# In[5]:


saver=tf.train.Saver()


# In[7]:


with tf.Session() as sess:
    saver.restore(sess,'checkpoint/my-model.ckpt-2000')
    
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    acc2=0
    for i in range(21):
        acc=sess.run(accuracy)
        print(' accuracy={:.2f}%.'.format(100. * acc))
        # print(acc)
        acc2+=acc
    print('overallaccuracy={:.2f}%.'.format(100. * acc2/21))
    # cm=sess.run(confusionmatrix)
    # print(cm)
    coord.request_stop()
    coord.join(threads)

