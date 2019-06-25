# SPP with Alex
This is an implementation of Spatial pyramid pooling (SPP) and the backbone is AlexNet. 

Here is the function of each python file:
my_ops.py:defining some basic operaions and functions.
network.py:defining the alexnet and the alexnet with SPP.
tfdata.py:generate tf.record format file for training and testing.
training.py:training the spp with alexnet on your own dataset (after you use tfdata.py generate your own tfrecord file)
test.py: report the test accuracy on your dataset.

requirement:
tensorflow>1.5
python3
opencv3
