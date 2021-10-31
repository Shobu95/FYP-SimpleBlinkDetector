import csv
import numpy as np 
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Activation,Dropout,MaxPooling2D
from keras.activations import relu
from tensorflow.keras.optimizers import Adam

def readDataset(path):
    
    with open(path,'r') as f:
        #read the csv dataset file
        reader = csv.DictReader(f)
        rows = list(reader)
    
    #define numpy array with all images
    imgs = np.empty((len(list(rows)),26,34,1),dtype=np.uint8)
    
    #define numpy array with all tags of images
    tgs = np.empty((len(list(rows)),1))
    
    for row, i in zip(rows,range(len(rows))):
        
        #converting dataset to image format
        img = row['image']
        img = img.strip('[').strip(']').split(', ')
        im = np.array(img, dtype=np.uint8)
        im = im.reshape((26,34))
        im = np.expand_dims(im, axis=2)
        imgs[i] = im
        
        #open eye tag = 1, close eye tag = 0
        tag = row['state']
        if tag == 'open':
            tgs[i] = 1
        else:
            tgs[i] = 0
            
    #shuffle dataset
    index = np.random.permutation(imgs.shape[0])
    imgs = imgs[index]
    tgs = tgs[index]

    return imgs,tgs

def makeModel():
    model = Sequential()
    
    model.add(Conv2D(32, (3,3), padding = 'same', input_shape = (26,34,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, (2,2), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Conv2D(128,(2,2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(optimizer=Adam(lr=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    
    return model 

def main():
    
    xTrain, yTrain = readDataset('training/dataset.csv')
    
    #scale the values of the images between 0 and 1
    xTrain = xTrain.astype('float32')
    xTrain /= 255
    
    model = makeModel()
    
    #Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2)
    
    datagen.fit(xTrain)
    
    #training the model
    model.fit_generator(datagen.flow(xTrain,yTrain,batch_size=32),
                        steps_per_epoch=len(xTrain) / 32, epochs=50)
    
    #saving the model
    model.save('training/trainedBlinkModel.hdf5')
    

if __name__ == "__main__":
    main()