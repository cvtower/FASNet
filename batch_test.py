
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
from FASNet import load_model
from FASNet import read_preprocess_image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
K.set_image_dim_ordering('th')


if __name__ == "__main__":
    
    # load Parameters
    pos_path = './test/1/'
    neg_path = './test/0/'
    img_width,img_height = 96,96
    
    # read and Pre-processing image
    # load the given weights or your finetuned model
    ori_model_path = './weights/REPLAY-ftweights18.h5'
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
    threshold = 0.99998
    time_start=time.time()
    #since we use Sequential model, call either predict or predict_classes is ok.
    #guess you will find out the perpose
    print("positive samples:")
    for item in pos_dirs:
        if os.path.isfile(pos_path+item):
            #print("%s:" %item)
            pos_cnt = pos_cnt+1
            img = read_preprocess_image(pos_path+item,img_width,img_height)
            #outLabel = model.predict(img,verbose=0)
            outLabel = model.predict_classes(img,verbose=0)
            #print(outLabel)
            #if outLabel>=threshold:
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
            #if outLabel<threshold:
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
