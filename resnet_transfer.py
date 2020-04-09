\

import os
from keras import applications
import keras
import tensorflow as tf
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras import backend as k



DATASET_PATH  = '/deepLearning/jschuppan/brc/chest_xray/'
IMAGE_SIZE    = (150,150)
#IMAGE_SIZE    = (32, 32)                                                                                                              \
                                                                                                                                        
NUM_CLASSES   = 2
BATCH_SIZE    = 32  # try reducing batch size or freeze more layers if your GPU runs out of memory                                     \
                                                                                                                                        
NUM_EPOCHS    = 20
WEIGHTS_FINAL = 'model-transfer-Chest-ResNet50-final.h5'


train_datagen = ImageDataGenerator( rescale=1.0 / 255.0,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                    channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/test',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)
lrelu = lambda x: tensorflow.keras.activations.relu(x, alpha=0.1)

#Now make our first convolutional layer, 32 filters, 3x3, default stride and padding                                                   \
                                                                                                                                        
model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape=[150,150,3])
#model = applications.MobileNet(weights =None, include_top=False, input_shape=(150,150,3), pooling='avg', alpha=.5, depth_multiplier=1, dropout=.2)
# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.                                               
for layer in model.layers[:100]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(128, activation="sigmoid")(x)
predictions = Dense(2, activation="softmax")(x)

#model.load_weights("MobileNet_Gleason_weights.h5",by_name=True)
#Use Adam optimizer (instead of plain SGD), set learning rate to explore.                                                              \
                                                                                                                                        
adam = Adam(lr=.00001)

#instantiate model                                                                                                                     \
                                                                                                                                        
model = Model(input=model.input, output=predictions)
#Compile model                                                                                                                         \
                                                                                                                                        
model.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['accuracy'])

#Print layers for resulting model                                                                                                      \
                                                                                                                                        
model.summary()

#Log training data into csv file                                                                                                       \
                                                                                                                                        
csv_logger = CSVLogger(filename="transfer-Chest-ResNet5000001-log.csv"+str(int(time.time())))
#checkpointer = ModelCheckpoint(filepath='deepLearning/jamccomb/weights/resnet50/weights.{epoch:02d}-{val_acc:.2f}.hdf5',monitor='val_loss', verbose=1, save_best_only=True, mode='min')
cblist = [csv_logger]

# train the model                                                                                                                      \
                                                                                                                                        
model.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
            callbacks=cblist)

# save trained model and weights                                                                                                       \
                                                                                                                                        
model.save(WEIGHTS_FINAL)

