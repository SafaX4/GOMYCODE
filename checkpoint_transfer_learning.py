#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive') # You can either use your drive or work directly on colab with temporary import


# In[ ]:


#@title Run Me Please
get_ipython().system('pip -q install pydot_ng')
get_ipython().system('pip -q install graphviz')
get_ipython().system('apt install graphviz > /dev/null')

from __future__ import absolute_import, division, print_function

import tensorflow as tf
# import tensorflow.contrib.eager as tfe
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow import keras
# try:
#   tf.enable_eager_execution()
#   print('Running in Eager mode.')
# except ValueError:
#   print('Already running in Eager mode')

from __future__ import print_function, division
from builtins import range, input
from keras.layers import Input, Lambda, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications.inception_v3 import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from tensorflow.contrib.layers import flatten

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle
from sklearn.utils import shuffle

# Feel free to import more packages
WARNING


# # Step 0: Load The Data

# In[ ]:


# Load pickled data

# TODO: Fill this in based on where you saved the training and testing data
# If you have a folder in your Drive named traffic-signs-data you do so, else change directory

training_file = '/content/gdrive/My Drive/traffic-signs-data/train.p'
validation_file= '/content/gdrive/My Drive/traffic-signs-data/valid.p'
testing_file = '/content/gdrive/My Drive/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)


# # Step 1: Dataset Summary & Exploration
# 

# In[ ]:


# Please complete None with your code
#-------------------------------------------------------------------------
signs = []
signnames = pd.read_csv('/content/gdrive/My Drive/traffic-signs-data/signnames.csv', delimiter=',', header=0)
for row in range(signnames.shape[0]):
     signs.append(signnames.iloc[row, 1])    
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# # Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas
#  

# In[ ]:


# Please complete None with your code
#-------------------------------------------------------------------------
#  Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of test examples.
n_test = X_test.shape[0]

#  What's the shape of a traffic sign image?
image_shape = X_test[0].shape

#  How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of valid examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# In[ ]:


# In this part you should One-hot encode all Y-s vectors using
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encodedtrain_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
Ytrain = np_utils.to_categorical(encodedtrain_Y)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_valid)
encodedvalid_Y = encoder.transform(y_valid)
# convert integers to dummy variables (i.e. one hot encoded)
Yvalid = np_utils.to_categorical(encodedvalid_Y)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_test)
encodedtest_Y = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
Ytest = np_utils.to_categorical(encodedtest_Y)


# # Include an exploratory visualization of the dataset
# 

# In[ ]:


### Data exploration visualization code goes here.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')
def show_images(X,Y,r,c):
  fig, axs = plt.subplots(r,c, figsize=(15, 6))
  fig.subplots_adjust(hspace = .2, wspace=.001)
  axs = axs.ravel()
  for i in range((X*Y)): # if rows = 2 and columns = 5 i should take 10 values
      index = random.randint(0, len(X))
      image = X[index]
      axs[i].axis('off')
      axs[i].imshow(image)
      axs[i].set_title(Y[index])
  plt.show()    

# show image of 10 random data points
rows = 2
columns = 5
show_images (rows,columns,r,c)


# In[ ]:


plt.figure(figsize=(20,5))
item, count = np.unique(y_train, return_counts=True)
#names is a list of traffic signs, Remember that we already have a list : signs
names = signs 
y_pos = np.arange(len(names))
plt.bar(item, count, alpha=0.6, color = (0.3,0.9,0.4,0.6) )

plt.xticks(y_pos, names, fontsize=15, rotation=90)

plt.show()


# In[ ]:


plt.figure(figsize=(20,5))
item, count = np.unique(y_test, return_counts=True)
item1, count1 = np.unique(y_valid, return_counts=True)

names = signs
names1 = signs

y_pos = np.arange(len(names))
plt.bar(item, count, alpha=0.6, color = (0.3,0.5,0.4,0.2), label="Validation Data" )

plt.bar(item1, count1, alpha=0.6, color = (0.9,0.1,0.3,0.2), label="Train Data" )

plt.xticks(y_pos, names, fontsize=15, rotation=90)
plt.legend()
plt.show()


# # Pre-process the Data Set (normalization, grayscale, etc.)
# 

# In[ ]:


# gray scale
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)
X_valid_gry = np.sum(X_valid/3, axis=3, keepdims=True)


# In[ ]:


# normalization Here 

# re-scale the image data to values between [0,1]
X_test_gry = X_test_gry/255.
X_train_gry = X_train_gry/255.
X_valid_gry = X_valid_gry/255.


# In[ ]:


# Shuffle your data here 

X_test_gry,X_train_gry,X_valid_gry=shffle(X_test_gry,X_train_gry,X_valid_gry)


# In[ ]:


import cv2

def random_translate(img):
    rows,cols,_ = img.shape
    
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst


# In[ ]:


def random_scaling(img):   
    rows,cols,_ = img.shape

    # transform limits
    px = np.random.randint(-2,2)

    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])

    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(rows,cols))
    
    dst = dst[:,:,np.newaxis]
    
    return dst


# In[ ]:


def random_warp(img):
    
    rows,cols,_ = img.shape

    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06   # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06

    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4

    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst


# In[ ]:


def random_brightness(img):
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    return dst


# In[ ]:


# Data augmentation 
# This may take too much time...
# i've set 2000 as a minimum number of images for each class
# if u don't already have a file containing generated Data, Run ME please
# --------------------------!!!!-----------------------------------------

# input_indices = []
# output_indices = []

# for class_n in range(n_classes):
#     class_indices = np.where(y_train == class_n)
#     n_samples = len(class_indices[0])
#     if n_samples < 2000:
#         for i in range(2000 - n_samples):
#             input_indices.append(class_indices[0][i%n_samples])
#             output_indices.append(new_X_train.shape[0])
#             new_img = new_X_train[class_indices[0][i % n_samples]]
#             new_img = random_translate(random_scaling(random_warp(random_brightness(new_img))))
#             new_X_train = np.concatenate((new_X_train, [new_img]), axis=0)
#             y_train = np.concatenate((y_train, [class_n]), axis=0)
# data_file = "/content/gdrive/My Drive/traffic-signs-data/new_train.p"
# pickle.dump({"images":new_X_train,"labels":y_train},open(data_file,"wb"),protocol=4)


# In[ ]:


# if u do have a file that contains new_train data RUN ME
with open("/content/gdrive/My Drive/traffic-signs-data/new_train.p","rb") as f:
    data = pickle.load(f)
new_X_train,y_train = data["images"],data["labels"]


# In[ ]:


# shuffle data  
new_X_train,y_train=shffle(new_X_train,y_train)

# CODE


# In[ ]:


def plot_figures_no_labels(figures, nrows = 1, ncols=1):
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 15))
    axs = axs.ravel()
    for index, title in zip(range(len(figures)), figures):
        axs[index].imshow(figures[title], plt.gray())
        axs[index].set_axis_off()
        
    plt.tight_layout()
    plt.show()
    
for class_n in range(n_classes): # you should range for all classes 
  figures = {}

  class_indices = list (np.where(y_train == class_n)[0])
  
  for i in range(8):
    
        figures[i] = new_X_train[class_indices[-i]].squeeze()
      
  plot_figures_no_labels(figures, 1, 8)


# In[ ]:


IMAGE_SIZE = [32, 32]
epochs = 100
batch_size = 64
image_input = Input(shape=(32, 32, 1))
# Feel free to use as many code cells as needed.
# define your model
# Use pretrained model 

vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_input) 
#we will include weights learned with imagenet dataset
output = vgg.layers[-1].output
output = Flatten()(output)
vgg_model = Model(vgg.input, output)
# we can chose which layer to train 
vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False
# all layers are not trainable

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


#Training
history = model.fit(new_X_train, y_train, 
                    epochs=epochs, batch_size=batch_size,
                    validation_data=(X_valid, Yvalid))

test_loss, test_accuracy = model.evaluate(X_test, Ytest, batch_size=batch_size)
print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

plt.plot(Yvalid, label='Validation accuracy', color = palette(2))
plt.plot(new_X_train, label='Train accuracy', color = palette(1))
plt.title("Training Performance")
plt.xlabel("Epoch")
plt.legend()
plt.show()


# In[ ]:


rom sklearn.metrics import confusion_matrix

predicted_classes = Ytrain
y_true = Ytest


cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot = True)
# PLOT IMAGES WITH PREDICTED SIGNS

rows, cols = Ytrain.shape
cols += 1

plt.figure(1, figsize=(48,48))
for row in range(rows):
    for col in range(cols):
        plt.subplot(rows, cols, row*cols + col + 1)
        if col == 0:
            plt.title("[{0}] True".format(y_new[row]), fontsize=30)
            plt.imshow(X_new0[row, :, :, :])
        else:
            plt.title("[{0}] {1:.4f}".format(indices[row, col-1], Ytrain[row, col-1]), fontsize=30)  
            plt.imshow(sign_examples[indices[row, col-1]])
            
        plt.axis('off')
    
plt.tight_layout(pad=0., w_pad=0., h_pad=1.0)

