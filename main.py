import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils import load_img
from sklearn.model_selection import train_test_split
import os
import pylab
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

Image_width = 128
Image_height = 128
Image_size = (Image_width, Image_height)
Image_channel = 3
Image_rgb_size = (Image_width, Image_height, 3)

df_train = pd.read_csv(r'D:\comp 4471 project\Training_Set\RFMiD_Training_Labels.csv')

df_train['category'] = df_train['Disease_Risk'].apply(lambda x: 'No Disease' if x == 0 else 'Have Disease')
df_train.head()
df_train.drop(['Disease_Risk'], inplace = True, axis = 1)
df_train.rename(columns = {'category' : 'Disease_Risk'}, inplace = True)
df_train.head()

df_validation = pd.read_csv(r'D:\comp 4471 project\Evaluation_Set\RFMiD_Validation_Labels.csv')

df_validation['category1'] = df_validation['Disease_Risk'].apply(lambda x: 'No Disease' if x == 0 else 'Have Disease')
df_validation.head()
df_validation.drop(['Disease_Risk'], inplace = True, axis = 1)
df_validation.rename(columns = {'category1' : 'Disease_Risk'}, inplace = True)
df_validation.head()

train_df = df_train.reset_index(drop = True)
val_df = df_validation.reset_index(drop = True)

batch_size = 40
epochs = 14
total_train = train_df.shape[0]
total_validate = val_df.shape[0]

train_dategen = ImageDataGenerator(rotation_range= 15,
                                   rescale= 1.0/255,
                                   shear_range= 0.1,
                                   zoom_range = 0.2,
                                   horizontal_flip= True,
                                   width_shift_range= 0.1,
                                   height_shift_range=0.1)

train_generator = train_dategen.flow_from_dataframe(
    train_df,
    "./Training_Set/Training",
    x_col='ID',
    y_col='Disease_Risk',
    target_size=Image_size,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    val_df,
    "./Evaluation_Set/Validation",
    x_col='ID',
    y_col='Disease_Risk',
    target_size=Image_size,
    class_mode='categorical',
    batch_size=batch_size
)

base = tensorflow.keras.applications.resnet50.ResNet50(weights= 'imagenet', include_top= False, input_shape= Image_rgb_size)

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(2, activation= 'softmax')(x)
model = Model(inputs= base.input, outputs= predictions)

from keras.optimizers import Adam

adam = Adam(learning_rate=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

callbacks= [earlystop, learning_rate_reduction]

history = model.fit(
    train_generator,
    epochs = epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

model.save('model.h5')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

test_filenames = sorted(os.listdir('./Test_Set/Test'), key = len)
test = pd.DataFrame({'ID' : test_filenames})
nb_samples = test.shape[0]
test.head()
print(test)

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test,
    './Test_Set/Test',
    x_col='ID',
    y_col=None,
    class_mode=None,
    target_size=Image_size,
    batch_size=batch_size,
    shufflee=False
)

predict = model.predict(test_generator, steps= np.ceil(nb_samples/batch_size))
test['category'] = np.argmax(predict, axis = -1)
test.head()

output_df = test.copy()
output_df.to_csv('submission.csv', index=False)