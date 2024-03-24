# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

![image](https://github.com/PriyankaAnnadurai/mnist-classification/assets/118351569/39c33773-b4a5-43ae-a225-f60940e375bd)

## Neural Network Model

![image](https://github.com/PriyankaAnnadurai/mnist-classification/assets/118351569/9aeb627d-acdd-4007-98e1-12422f0264ef)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
Download and load the dataset

### STEP 3:
Scale the dataset between it's min and max values

### STEP 4:
Using one hot encode, encode the categorical values

### STEP 5:
Split the data into train and test

### STEP 6:
Build the convolutional neural network model

### STEP 7:
Train the model with the training data

### STEP 8:
Plot the performance plot

### STEP 9:
Evaluate the model with the testing data

### STEP 10:
Fit the model and predict the single input

## PROGRAM

### Name: Priyanka A
### Register Number: 212222230113

```python
## Load the dataset from the tensorflow library.
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

## Preprocess the dataset.
(X_train, y_train), (X_test, y_test) = mnist.load_data()
     
X_train.shape

X_test.shape

single_image= X_train[0]
     
single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

## Create and train your model.
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

## Compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

## Fit the model    
model.fit(X_train_scaled ,y_train_onehot, epochs=8,batch_size=128, validation_data=(X_test_scaled,y_test_onehot))

## Include the training loss, validation loss vs iteration plot.
metrics = pd.DataFrame(model.history.history)
metrics.head()

print("Priyanka A - 212222230113")
metrics[['accuracy','val_accuracy']].plot()

print("Priyanka A - 212222230113")
metrics[['loss','val_loss']].plot()

## Test the model for your handwritten scanned images.
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print("Priyanka A - 212222230113")
print(confusion_matrix(y_test,x_test_predictions))

print("Priyanka A - 212222230113")
print(classification_report(y_test,x_test_predictions))

# Prediction for a single input
img = image.load_img('3.jpg')
type(img)

img = image.load_img('3.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
     
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

print("Priyanka A - 212222230113")
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
     
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print("Priyanka A - 212222230113")
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/PriyankaAnnadurai/mnist-classification/assets/118351569/339d3f3d-5b79-49b0-95b5-0d160530e607)

![image](https://github.com/PriyankaAnnadurai/mnist-classification/assets/118351569/fdf4cb72-ee4f-4baa-b142-c2b47b5376df)


### Classification Report

![image](https://github.com/PriyankaAnnadurai/mnist-classification/assets/118351569/3d76ec8a-e713-4582-ad15-ac293e9698fa)


### Confusion Matrix

![image](https://github.com/PriyankaAnnadurai/mnist-classification/assets/118351569/f6c86093-1d96-48d7-863e-4df382ee37e4)


### New Sample Data Prediction

![image](https://github.com/PriyankaAnnadurai/mnist-classification/assets/118351569/3e4675fb-ebe3-4e36-90ac-35aa00e51774)

![image](https://github.com/PriyankaAnnadurai/mnist-classification/assets/118351569/6338f066-452a-45d1-a85b-097fbc21dc60)

## RESULT
Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
