
# coding: utf-8

# In[ ]:

import os, time
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras import optimizers
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import callbacks
from keras import backend as K
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
K.set_image_dim_ordering('th')


# # Train

# In[ ]:

# path to the model weights files.
weights_path = './weights/REPLAY-ftweights18.h5'
top_model_weights_path = './model'

# dimensions of images. (less than 224x 224)
img_width, img_height = 96,96

# number of layers to freeze
nFreeze = 0

train_data_dir = './train'
validation_data_dir = './val'
nb_train_samples = 10524
nb_validation_samples = 2090
nb_epoch = 10

def get_tr_vgg_model(weights_path, img_width, img_height):
    
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    if weights_path:
        model.load_weights(weights_path, by_name=True)
        print('Model loaded.')
        return model
    #print ('Model loaded.')
    else:
        print('model not loaded...')

def add_top_layers(model):

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # add the model on top of the convolutional base
    model.add(top_model)
    
    return model

def run_train(model):
    
    start_time = time.time()
    
    # freeze layers
    for layer in model.layers[:nFreeze]:
        layer.trainable = False

    # compile model
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6),
              metrics=['accuracy'])
    
    print ('Model Compiled.')
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip = True,
            fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=256,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=256,
            class_mode='binary')

    print ('Fine-tuning top layers...')

    earlyStopping = callbacks.EarlyStopping(monitor='val_acc',
                                           patience=10, 
                                           verbose=0, mode='auto')

    #fit model
    model.fit_generator(
           train_generator,
           callbacks=[earlyStopping],
           samples_per_epoch=nb_train_samples,
           steps_per_epoch=nb_train_samples/256,
           nb_epoch=nb_epoch,
           validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

    #model.save_weights(top_model_weights_path)
    model_structure = model.to_json()
    open('model.json','w').write(model_structure)
    model.save_weights('model.h5',overwrite=True)    
    print ('Done fine-tuning, have a nice day!')
    print("Execution time %s seconds" % (time.time() - start_time))


# In[ ]:
'''
if __name__ == "__main__":
   
    vgg16_tr_model = get_tr_vgg_model(weights_path, img_width, img_height)
    vgg16_tr_model = add_top_layers(vgg16_tr_model)
    
    # fine-tuning the model 
    run_train(vgg16_tr_model)
'''

# # Test

# In[ ]:

def load_model(weightsPath,img_width,img_height):
    
    #VGG-16 model
    model = Sequential()
   
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Top-model for anti-spoofing
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    #
    
    model.add(top_model)
    
    if weightsPath:
        model.load_weights(weightsPath)
    else:
        print ('Could not load model!')
    
    return model

def read_preprocess_image(imgPath,img_width,img_height):
      
    img = load_img(imgPath,target_size=(img_width,img_height))
    imgArray = img_to_array(img)
    imgArray = imgArray.reshape(1,3,img_width, img_height)
    imgArray = imgArray/float(255)
    
    return imgArray


# In[ ]:

if __name__ == "__main__":
    
    # load Parameters
    pos_path = './1/'
    neg_path = './0/'
    test_model_path = './model.json'
    img_width,img_height = 96,96
    
    # read and Pre-processing image
    # load weights
    #model = load_model(test_model_path,img_width,img_height)
    ori_model_path = './weights/REPLAY-ftweights18.h5'
    #ori_model_path = './weights/3DMAD-ftweights18.h5'
    use_ori_model = True
    # load weights
    if use_ori_model:
        model = load_model(ori_model_path,img_width,img_height)
    else:
        model_path_json = 'model.json'
        model_path_h5 = 'model.h5'
        with open(model_path_json) as file_constant:
            model = model_from_json(file_constant.read())
        model.load_weights(model_path_h5)
    # predict Class
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    pos_dirs = os.listdir( pos_path )
    neg_dirs = os.listdir( neg_path )
    pos_cnt = 0
    neg_cnt = 0
    pos_true_cnt = 0
    neg_true_cnt = 0
    #threhold = 0.99998
    time_start=time.time()
    print("positive samples:")
    for item in pos_dirs:
        if os.path.isfile(pos_path+item):
            #print("%s:" %item)
            pos_cnt = pos_cnt+1
            img = read_preprocess_image(pos_path+item,img_width,img_height)
            #outLabel = model.predict(img,verbose=0)
            outLabel = model.predict_classes(img,verbose=0)
            #print(outLabel)
            #if outLabel>=threhold:
            if outLabel==1:
                pos_true_cnt = pos_true_cnt+1

    #print("negative samples:")
    for item in neg_dirs:
        if os.path.isfile(neg_path+item):
            #print("%s:" %item)
            neg_cnt = neg_cnt+1
            img = read_preprocess_image(neg_path+item,img_width,img_height)
            #outLabel = model.predict(img,verbose=0)
            outLabel = model.predict_classes(img,verbose=0)
            #print(outLabel)
            #if outLabel<threhold:
            if outLabel==0:
                neg_true_cnt = neg_true_cnt+1
    time_end=time.time()
    print('totally cost',time_end-time_start)
    print('average cost',(time_end-time_start)/(pos_cnt+neg_cnt))
    pt_acc = pos_true_cnt/pos_cnt
    nt_acc = neg_true_cnt/neg_cnt
    total_acc = (pos_true_cnt+neg_true_cnt)/(pos_cnt+neg_cnt)
    print("pt_acc: %f" % pt_acc)
    print("nt_acc: %f" % nt_acc)
    print("total_acc: %f" % total_acc)
