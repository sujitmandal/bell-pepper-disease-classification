# Author : Sujit Mandal
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import backend
import tensorflow_model_optimization as tfmot
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Github: https://github.com/sujitmandal
# Pypi : https://pypi.org/user/sujitmandal/
# LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/


EPOCHS = 10
CHANNELS = 3
INIT_LR = 1e-3 # 0.001
Image_Size = 0
BATCH_SIZE = 32
IMAGE_SIZE = 256
dataset_dir = 'dataset'
AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_IMAGE_SIZE = tuple((IMAGE_SIZE, IMAGE_SIZE))


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    seed = 123,
    image_size=DEFAULT_IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print(class_names)

total_class = len(class_names)


# plt.figure(figsize=(10, 10))
# for batch, label in dataset.take(1):
#     for i in range(16):
#         ax = plt.subplot(4, 4, i + 1)
#         plt.imshow(batch[i].numpy().astype("uint8"))
#         plt.title(class_names[label[i]])
#         plt.axis("off")


def split_dataset(dataset, train_split, val_split, test_split):
    # shuffle = True
    # shuffle_size = 10000
    assert(train_split + val_split + test_split) == 1

    dataset_size = len(dataset)

    # if shuffle:
    #     dataset = dataset.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size).skip(val_size)

    return(train_dataset, val_dataset, test_dataset)


train_split = 0.8
val_split = 0.1
test_split = 0.1

train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_split, val_split, test_split)


print('\n')
print('Total dataset : {}'.format(len(dataset)))
print('Train dataset : {}'.format(len(train_dataset)))
print('Val dataset : {}'.format(len(val_dataset)))
print('Test dataset : {}'.format(len(test_dataset)))


# Checking Batch Size
for image_batch, labels_batch in train_dataset:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# Cache, Shuffle, and Prefetch the Dataset
# train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
# val_dataset = val_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
# test_dataset = test_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# Resizing and Normalization
data_resize_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1./255)
])

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

# Checking what is the expected dimension order for channel
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
batch_input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
channel_dim = -1

if backend.image_data_format() == 'channels_first':
    input_shape = (CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    batch_input_shape = (BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    channel_dim = 1


# build CNN model

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

model = models.Sequential([
    data_resize_rescale,
    data_augmentation,

    layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape= input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'), 
    layers.Dense(total_class, activation = 'softmax'),
])

model.build(input_shape=input_shape)

print('\n')
print('Model :')
model.summary()

# compiling the model
model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    batch_size = BATCH_SIZE,
    validation_data = val_dataset,
    verbose = 1,
    epochs = EPOCHS,
)

model_score = model.evaluate(test_dataset)

print('\n')
print(model_score)
print(history.params)
print(history.history.keys())


# Testing the Model
print("Calculating model accuracy")
print(f"Test Accuracy: {round(model_score[1],4)*100}%")




model_performance = history.history
print(json.dumps(model_performance, indent=4))


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), accuracy, label='Training Accuracy')
plt.plot(range(EPOCHS), val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Quantize only the Dense, MaxPool2D, Conv2D Layers
def apply_quantization(layer):
    if (
        isinstance(layer, layers.Dense)
        or isinstance(layer, layers.MaxPool2D)
        or isinstance(layer, layers.Conv2D)
    ):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    
    return(layer)


# Clone the Model and Make Quantization
annotated_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_quantization,
)

quant_model = tfmot.quantization.keras.quantize_apply(annotated_model)
quant_model.summary()


# Compile Quantization Model before Fine Tuning
quant_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Fine Tuning the Quantization Model
quant_history = quant_model.fit(train_dataset,
    batch_size=BATCH_SIZE,
    validation_data=val_dataset,
    verbose=1,
    epochs=EPOCHS,
)

# Evaluate the Model Accuracy
print("Calculating Quant model accuracy")
quant_model_score = quant_model.evaluate(test_dataset)
print(f"Test Accuracy: {round(quant_model_score[1],4)*100}%")


print('\n')
print(quant_model_score)
print(quant_history.params)
print(quant_history.history.keys())

quant_model_performance = quant_history.history

print(json.dumps(model_performance, indent=4))


quant_accuracy = quant_history.history['accuracy']
quant_val_accuracy = quant_history.history['val_accuracy']

quant_loss = quant_history.history['loss']
quant_val_loss = quant_history.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), quant_accuracy, label='Training Accuracy')
plt.plot(range(EPOCHS), quant_val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), quant_loss, label='Training Loss')
plt.plot(range(EPOCHS), quant_val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Convert Quanitzation Model to TF Lite Mode

# Convert the Model
converter = tf.lite.TFLiteConverter.from_keras_model(quant_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()


# Evaluate the TF Lite Model
def evaluate_tflite_model(dataset, interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    prediction_digits = []
    test_labels = []
    for image, label in dataset.unbatch().take(dataset.unbatch().cardinality()):

        test_image = np.expand_dims(image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)
        interpreter.invoke()
        
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)
        test_labels.append(label)

    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()
    
    return(accuracy)


interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

test_accuracy = evaluate_tflite_model(dataset, interpreter)

print('Quant TFLite test_accuracy:', test_accuracy)


# Test model
for image_batch, labels in test_dataset.take(1):
    image = image_batch[0].numpy().astype("uint8")
    label = labels[0].numpy()
    
    print('Image to Predict :')
    plt.imshow(image)
    print('Actual Lable : ', class_names[label])
    
    batch_prediction = model.predict(image_batch)
    print('Predicted Label : ', class_names[np.argmax(batch_prediction[0])])


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    
    accuracy = round(100 * (np.max(predictions[0])), 2)
    
    return(predicted_class, accuracy)


plt.figure(figsize=(15, 15))
for images, labels in test_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, accuracy = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Accuracy: {accuracy}%")
        
        plt.axis("off")

# Saving the TF Lite Model¶
model_version = max([int(i) for i in (os.listdir("tf-lite-models")+[0])]) + 1

with open(
    f"tf-lite-models/tf-lite-models-" + str(model_version) + ".tflite", 'wb') as f:
        f.write(quantized_tflite_model)