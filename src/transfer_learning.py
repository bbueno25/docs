# MIT License
#
# Copyright (c) 2017 François Chollet                                                                          
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import matplotlib.pyplot as pyplot
import numpy 
import os
import tensorflow
import tensorflow_datasets

# data download
keras = tensorflow.keras

tensorflow_datasets.disable_progress_bar()

SPLIT_WEIGHTS = (8, 1, 1)
splits = tensorflow_datasets.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tensorflow_datasets.load(
    'cats_vs_dogs', split=list(splits), with_info=True, as_supervised=True)

print(raw_train)
print(raw_validation)
print(raw_test)

get_label_name = metadata.features['label'].int2str
for image, label in raw_train.take(2):
    pyplot.figure()
    pyplot.imshow(image)
    pyplot.title(get_label_name(label))

# format data
IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
    image = tensorflow.cast(image, tensorflow.float32)
    image = (image/127.5) - 1
    image = tensorflow.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
   pass

image_batch.shape

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tensorflow.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, weights='imagenet'
    )

feature_batch = base_model(image_batch)
print(feature_batch.shape)

# freeze the convolutional base 
base_model.trainable = False
# Let's take a look at the base model architecture
base_model.summary()

# add classification head
global_average_layer = tensorflow.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tensorflow.keras.Sequential(
    [base_model, global_average_layer, prediction_layer]
    )

# compile the model
base_learning_rate = 0.0001
model.compile(
    optimizer=tensorflow.keras.optimizers.RMSprop(lr=base_learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

model.summary()

len(model.trainable_variables)

# train the modrl
num_train, num_val, num_test = (
    metadata.splits['train'].num_examples * weight / 10
    for weight in SPLIT_WEIGHTS
    )
initial_epochs = 10
steps_per_epoch = round(num_train) // BATCH_SIZE
validation_steps = 20
loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
history = model.fit(
    train_batches, 
    epochs=initial_epochs,
    validation_data=validation_batches
    )

# learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

pyplot.figure(figsize=(8, 8))
pyplot.subplot(2, 1, 1)
pyplot.plot(acc, label='Training Accuracy')
pyplot.plot(val_acc, label='Validation Accuracy')
pyplot.legend(loc='lower right')
pyplot.ylabel('Accuracy')
pyplot.ylim([min(pyplot.ylim()),1])
pyplot.title('Training and Validation Accuracy')

pyplot.subplot(2, 1, 2)
pyplot.plot(loss, label='Training Loss')
pyplot.plot(val_loss, label='Validation Loss')
pyplot.legend(loc='upper right')
pyplot.ylabel('Cross Entropy')
pyplot.ylim([0, 1.0])
pyplot.title('Training and Validation Loss')
pyplot.xlabel('epoch')
pyplot.show()   

# # fine-tuning
# base_model.trainable = True
# # Let's take a look to see how many layers are in the base model
# print("Number of layers in the base model: ", len(base_model.layers))
# # Fine tune from this layer onwards
# fine_tune_at = 100
# # Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable =  False

# # recompile model
# model.compile(
#     loss='binary_crossentropy',
#     optimizer = tensorflow.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
#     metrics=['accuracy']
#     )
# model.summary()
# len(model.trainable_variables)

# # retrain model
# fine_tune_epochs = 10
# total_epochs =  initial_epochs + fine_tune_epochs
# history_fine = model.fit(
#     train_batches,
#     epochs=total_epochs,
#     initial_epoch = initial_epochs,
#     validation_data=validation_batches
#     )

# # learning curves
# acc += history_fine.history['accuracy']
# val_acc += history_fine.history['val_accuracy']

# loss += history_fine.history['loss']
# val_loss += history_fine.history['val_loss']

# pyplot.figure(figsize=(8, 8))
# pyplot.subplot(2, 1, 1)
# pyplot.plot(acc, label='Training Accuracy')
# pyplot.plot(val_acc, label='Validation Accuracy')
# pyplot.ylim([0.8, 1])
# pyplot.plot([initial_epochs-1,initial_epochs-1], pyplot.ylim(), label='Start Fine Tuning')
# pyplot.legend(loc='lower right')
# pyplot.title('Training and Validation Accuracy')

# pyplot.subplot(2, 1, 2)
# pyplot.plot(loss, label='Training Loss')
# pyplot.plot(val_loss, label='Validation Loss')
# pyplot.ylim([0, 1.0])
# pyplot.plot([initial_epochs-1,initial_epochs-1], pyplot.ylim(), label='Start Fine Tuning')
# pyplot.legend(loc='upper right')
# pyplot.title('Training and Validation Loss')
# pyplot.xlabel('epoch')
# pyplot.show()
