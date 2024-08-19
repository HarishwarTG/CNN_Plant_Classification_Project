import numpy as np
import PIL
import os
from datasets import load_from_disk
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

    

def preprocess_images(img):
    img_size=(224,224)
    data_augmentation = ImageDataGenerator(
                        rotation_range=20,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True,
                        fill_mode='nearest')
    img=image.img_to_array(img)
    img=data_augmentation.random_transform(img)
    img=image.smart_resize(img,img_size)
    img=img/255
    return img


def CNN_Model(train_pp_images,train_labels_one_hot,val_pp_images,validation_labels_one_hot,test_pp_images,test_labels):
    model = Sequential()

    model.add(Conv2D(32,(2,2),activation='relu',input_shape=(224,224,3) , kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64,(2,2),activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(14,activation='softmax'))


    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    Train = model.fit(np.array(train_pp_images),
                        train_labels_one_hot,
                        epochs=1,
                        batch_size=32,
                        validation_data=(np.array(val_pp_images),validation_labels_one_hot),
                        verbose=1)

    model.save('Z:/CNN_Crop_Classification/Notebook/plant_classification_tf', save_format='tf')
    
    ans=model.predict(np.array(test_pp_images))
    ans=tf.nn.softmax(ans)
    y_pred=[np.argmax(res) for res in ans]
    acc=accuracy_score(test_labels,y_pred)
    print('The Accuracy of the trained CNN model is : ',acc)


def main():
    folder= 'Z:\CNN_Crop_Classification\Dataset'
    ds = load_from_disk(folder)

    train_images = ds['train']['image']
    train_labels = ds['train']['label']

    test_images = ds['test']['image']
    test_labels = ds['test']['label']

    validation_images = ds['validation']['image']
    validation_labels = ds['validation']['label']
    
    train_pp_images=[preprocess_images(img) for img in train_images]
    test_pp_images=[preprocess_images(img) for img in test_images]
    val_pp_images=[preprocess_images(img) for img in validation_images]
    
    train_labels_one_hot = to_categorical(train_labels,14)
    validation_labels_one_hot = to_categorical(validation_labels,14)
    
    CNN_Model(train_pp_images,train_labels_one_hot,val_pp_images,validation_labels_one_hot,test_pp_images,test_labels)

if __name__== '__main__':
    main()