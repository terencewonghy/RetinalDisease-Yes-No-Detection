import pandas as pd
import numpy as np
import datetime as dt
import os
import os.path
from pathlib import Path
import glob
#import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, SpatialDropout2D
from keras import layers
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.optimizers import RMSprop
from keras.models import Model
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from keras.preprocessing import image
from PIL import Image

##2. Organizing Training and Testing Dataframes##

# Selecting Dataset Folder Paths
zero_dir = Path(r'D:\FYP\0\0_eyes')
one_dir = Path(r'D:\FYP\1\1_eyes')
zero_filepaths = list(zero_dir.glob(r'**/*.png'))
one_filepaths = list(one_dir.glob(r'**/*.png'))

# Mapping the labels
zero_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], zero_filepaths))
one_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], one_filepaths))

# Paths & labels zero eyes
zeros_filepaths = pd.Series(zero_filepaths, name= 'File').astype(str)
zeros_labels = pd.Series(zero_labels, name='Label')

# Paths & labels one eyes
ones_filepaths = pd.Series(one_filepaths, name= 'File').astype(str)
ones_labels = pd.Series(one_labels, name='Label')

# Concatenating...
zero_df = pd.concat([zeros_filepaths, zeros_labels], axis=1)
one_df = pd.concat([ones_filepaths, ones_labels], axis=1)

df = pd.concat([zero_df, one_df])

df = df.sample(frac = 1, random_state= 56).reset_index(drop = True)

vc = df['Label'].value_counts()
plt.figure(figsize= (9, 5))
sns.barplot(x= vc.index, y=vc)
plt.title("Number of images for each category in the Training Dataset", fontsize = 11)
plt.show()
##3. Img Observation##
plt.style.use("dark_background")
figure = plt.figure(figsize=(2,2))
x = plt.imread(df["File"][34])
plt.imshow(x)
plt.xlabel(x.shape)
plt.title(df["Label"][34])

figure = plt.figure(figsize=(2, 2))
x = plt.imread(df["File"][11])
plt.imshow(x)
plt.xlabel(x.shape)
plt.title(df["Label"][11])

fig, axes = plt.subplots(nrows = 5,
                        ncols = 5,
                        figsize = (7, 7),
                        subplot_kw = {"xticks":[],"yticks":[]})

for i,ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df["File"][i]))
    ax.set_title(df["Label"][i])
plt.tight_layout()
plt.show()

##3. Dividing into training and testing sets##
_, testset_df = train_test_split(df, train_size = 0.75, random_state = 4)



testset_df.head()

# converting the Label to a numeric format for testing later...
LE = LabelEncoder()

y_test = LE.fit_transform(testset_df["Label"])

print(testset_df)

# Viewing data in test dataset
print('Test Dataset:')

print(f'Number of images: {testset_df.shape[0]}')

print(f'Number of images with one eyes: {testset_df["Label"].value_counts()[0]}')
print(f'Number of images with zero eyes: {testset_df["Label"].value_counts()[1]}\n')

##4. Data Augmentation##


test_datagen = ImageDataGenerator(rescale = 1./255)

##5. Preparing datagen for training, validation and test datasets##


print("Preparing the test dataset ...")
test_set = test_datagen.flow_from_dataframe(
    dataframe = testset_df,
    x_col = "File",
    y_col = "Label",
    target_size = (75, 75),
    color_mode ="rgb",
    class_mode = "binary",
    shuffle = False,
    batch_size = 32)
print('Data generators are ready!')


print("Test: ")
print(test_set.class_indices)
print(test_set.image_shape)



from keras.models import load_model

model = load_model('disease-classifier-v12.h5')


score_CNN = model.evaluate(test_set)
print("Test Loss:", score_CNN[0])
print("Test Accuracy:", score_CNN[1])

y_pred_CNN = model.predict(test_set)
y_pred_CNN = np.round(y_pred_CNN)
recall_CNN = recall_score(y_test, y_pred_CNN)
precision_CNN = precision_score(y_test, y_pred_CNN)
f1_CNN = f1_score(y_test, y_pred_CNN)
roc_CNN = roc_auc_score(y_test, y_pred_CNN)

print(classification_report(y_test, y_pred_CNN))


plt.figure(figsize = (6, 4))

sns.heatmap(confusion_matrix(y_test, y_pred_CNN),annot = True, fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()

df = pd.DataFrame(y_pred_CNN, columns=['Prediction Label'])

result = pd.concat([testset_df, df.set_index(testset_df.index)], axis=1)
result['Label'] = result['Label'].str.replace('_eyes', '')
result['Prediction Label'] = result['Prediction Label'].astype(int)
result.to_csv('prediction_result_v12.csv', index=False)