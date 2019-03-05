#!/usr/bin/python
# coding:utf8

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
img = load_img('C:\\Users\\lenovo\\Desktop\\cat4.png')
x = img_to_array(img)
x = x.reshape((1,)+x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='C:\\Users\\lenovo\\Desktop\\dataset\\cat', save_prefix='cat', save_format='png'):
    i += 1
    if i>60:
        break