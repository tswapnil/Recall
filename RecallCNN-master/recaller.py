import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Input,Dense,Activation
import numpy as np
import datetime
import os
import shutil
import PIL
from imagenet_utils import decode_predictions, preprocess_input
import imagenet_utils
from keras.preprocessing import image as image_utils

mat = dict()
for line in truth:
    if line in mat: 
        mat[line] +=1
    else :
        mat[line] = 1
#This just stores the count of an index
predMat = dict()

#This stores the mapping of fileName vs label

#print (truthLabelDict)

queryImagePath = "C:\\study\\Second Quarter\\Recall\\RecallCNN-master\\Query\\ILSVRC2012_val_00000201.JPEG";
img = image_utils.load_img(imagePath,target_size=(224, 224)) 
temp = image_utils.img_to_array(img)
temp = np.expand_dims(temp, axis=0)
temp = preprocess_input(temp)
preds = TopModel.predict(temp)
P = decode_predictions(preds,par)
(qimagenetID, qlabel, qprob, qindex) = P[0][0]

queryLabelFile = open("C:\\study\\Second Quarter\\Recall\\RecallCNN-master\\Query\\label.txt" ,'r')
queryTrueLabel = queryLabelFile.readline()




#fraction of instances that have the same label as Query Label
fTInstance = predMat(queryTrueLabel)

#fraction of instances that are retrieved
fInstance = predMat(qindex)
