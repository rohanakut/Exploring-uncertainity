import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

num_classes = 10
input_shape = (28, 28, 1)

##code refered from https://pythonistaplanet.com/cifar-10-image-classification-using-keras/
##lines 16-33 are refered from the above mentioned website. From line 33 he entire code is written by me
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



##from her on the entire code is mine
visible = Input(shape=(28,28,1))
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
dropout1 = keras.layers.Dropout(0.5)(conv1, training=True)
pool2 = MaxPooling2D(pool_size=(2, 2))(dropout1)
dropout2 = keras.layers.Dropout(0.5)(pool2, training=True)
flat = Flatten()(pool2)
output = Dense(10, activation='sigmoid')(flat)
model = Model(inputs=visible, outputs=output)
print(model.summary())
x = x_train
y = y_train
x_train = x[0:20]
y_train=y[0:20]
x_pool = x[20::]
y_pool = y[20::]
print(x_train.shape)
print(x_pool.shape)

batch_size = 128
epochs = 5

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
sc = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", sc[0])
print("Test accuracy:", sc[1])

def find_var(sample):
  entropy = 0
  val = sample.max()
  return 1-val
#   sample_log = np.log(sample)
#   entropy=(np.sum(sample*sample_log))
#  return -entropy 

import tensorflow as tf
soft = tf.keras.layers.Softmax()
iter = []
values_mean = []
entr =[]
for k in range(100):
    for j in range(len(x_pool)):
        for i in range(3):
            test_image  = np.expand_dims(x_pool[j], axis=0)
            test=soft(model.predict(test_image))
            test = test.numpy()
            iter.append(test)
        print(j)
        values_mean.append(np.mean(iter,axis=0))
        iter.clear()
        print(j)
    for i in range(len(values_mean)):
        entr.append(find_var(values_mean[i]))
    values=np.argsort(entr)[-5:]
    x_train = x_train.tolist()
    y_train = y_train.tolist()
    x_pool = x_pool.tolist()
    y_pool = y_pool.tolist()
    for i in range(len(values)):
        x_train.append(x_pool[values[i]])
        y_train.append(y_pool[values[i]])
    print(len(x_train))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(x_train.shape)
    for i in range(len(values)):
        x_pool.pop(values[i])
        y_pool.pop(values[i])
    entr.clear()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    score = model.evaluate(x_test, y_test, verbose=0)
    num_arr = []
    num_arr.append(score[1])
np.save('data.npy', num_arr)
    