import tensorflow_datasets as tfds
import numpy as np

(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    batch_size=-1
)

train_images, train_labels = ds_train
test_images, test_labels = ds_test

x_train = train_images.numpy()
y_train = train_labels.numpy()
x_test = test_images.numpy()
y_test = test_labels.numpy()

remove_indices = []
# find all digits
for i in range (0, len(x_train)):
    if y_train[i] <= 9:
        remove_indices.append(i)

# remove all digits
x_train = np.delete(x_train, remove_indices, axis=0)
y_train = np.delete(y_train, remove_indices, axis=0)

remove_indices = []
for i in range (0, len(x_test)): #find all digits
    if y_test[i] <= 9:
        remove_indices.append(i)
x_test = np.delete(x_test, remove_indices, axis=0)
y_test = np.delete(y_test, remove_indices, axis=0)

# offset train data by 10
# now: 0-25 are A-Z, 26-36 are lowercase letters 
# (a b d e f g h n q r t)
y_train = y_train - 10
y_test = y_test - 10

# the images in this dataset are rotated.
x_train = np.transpose(x_train, (0, 2, 1, 3))
x_test  = np.transpose(x_test,  (0, 2, 1, 3))

# normalize the image values
x_train = x_train / 255.0
x_test = x_test / 255.0

def get_train_set():
    return x_train, y_train

def get_test_set():
    return x_test, y_test