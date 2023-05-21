import csv
import os
import cv2
import numpy as np
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split 
import tensorflow as tf
#x_train, x_test, y_train, y_test = train_test_split(images, target, test_size=0.22, random_state=42, stratify=target)


with open('/content/ISBI2016_ISIC_Part3_Training_GroundTruth.csv', mode='r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    gt = {rows[0]:rows[1] for rows in reader}

print(gt)
path = '/content/ISBI2016_ISIC_Part3_Training_Data'
images = [] 
target = [] 
images_arr = np.asarray(images)
#print(gt['ISIC_0011203'])

for root, dirs, files in os.walk(path):
    for file in files:
        with open(os.path.join(root, file), "r") as auto: 
           # print(root+'/'+file)
            if file.endswith('.png'):
              print("hello")
              continue   
            im = cv2.imread(root+'/'+file)
            file = file.replace('.jpg','')  
            im = cv2.resize(im,(400,400))
            images.append(im)
            if(gt[file]=='benign'):
              target.append(0)
            else:
              target.append(1)

images=np.array(images)
target = np.array(target)
print(images.shape)
print(target.shape)
target = keras.utils.to_categorical(target, 2)
x_train, x_test, y_train, y_test = train_test_split(images, target, test_size=0.22, random_state=42, stratify=target)
x = x_train
y = y_train
x_train = x[0:20]
y_train=y[0:20]
x_pool = x[20::]
y_pool = y[20::]

visible = Input(shape=(400,400,3))
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
dropout1 = keras.layers.Dropout(0.5)(conv1, training=True)
pool2 = MaxPooling2D(pool_size=(2, 2))(dropout1)
dropout2 = keras.layers.Dropout(0.5)(pool2, training=True)
flat = Flatten()(pool2)
output = Dense(2, activation='sigmoid')(flat)
model = Model(inputs=visible, outputs=output)
print(model.summary())

batch_size = 128
epochs = 5

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.AUC()])
print(x_train.shape)
print(y_train.shape)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
sc = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", sc[0])
print("Test accuracy:", sc[1])

def find_var(sample):
  entropy = 0
  #val = sample.max()
  #print(val)
  #return 1-val
  sample_log = np.log(sample)
  entropy=(np.sum(sample*sample_log))
  return -entropy 

import tensorflow as tf
soft = tf.keras.layers.Softmax()
iter = []
values_mean = []
entr =[]
num_arr = []
x_pool = x_pool.tolist()
y_pool = y_pool.tolist()
for k in range(3):
    print("ietartion started",k)
    for j in range(len(x_pool)):
        if(j==5000):
          break
        for i in range(3):
            test_image  = np.expand_dims(x_pool[j], axis=0)
            test=soft(model.predict(test_image))
            test = test.numpy()
            iter.append(test)
       # print(j)
        values_mean.append(np.mean(iter,axis=0))
        iter.clear()
        #print(j)
    for i in range(len(values_mean)):
        entr.append(find_var(values_mean[i]))
    values=np.argsort(entr)[-5:]
    #print(values)
    x_train = x_train.tolist()
    y_train = y_train.tolist()
    for i in range(len(values)):
        x_train.append(x_pool[values[i]])
        y_train.append(y_pool[values[i]])
    #print(len(x_train))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #print(x_train.shape)
    for i in range(len(values)):
        x_pool.pop(values[i])
        y_pool.pop(values[i])
    entr.clear()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.AUC()])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    score = model.evaluate(x_test, y_test, verbose=0)
    num_arr.append(score[1])
    print(score[1])
np.save('data_entr_isic.npy', num_arr)