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

#Read Predictions

predictionFile = open("C:\\study\\Second Quarter\\Recall\\RecallCNN-master\\Predictions.txt",'r');

matByIndex = dict()
matByLabel = dict()

i = 0
for line in predictionFile :
    
    if line in matByLabel: 
        matByLabel[line] +=1
    else :
        matByLabel[line] = 1
    
    matByIndex[i] = line
    
    i = i + 1


#Read Truth Data
truthFile = open('C:\\study\\Second Quarter\\Recall\\newData\\labels.txt', 'r')


matTruthByLabel = dict()
matTruthByIndex = dict()
i = 0 
for line in truthFile:
    if line in matTruthByLabel: 
        matTruthByLabel[line] +=1
    else :
        matTruthByLabel[line] = 1
    matTruthByIndex[i] = line
    i = i + 1


#Read Queries

queryTruthFile = open("C:\\study\\Second Quarter\\Recall\\RecallCNN-master\\Query\\labels.txt",'r')
queryTruthByIndex = dict()
i=0
for line in queryTruthFile :
    queryTruthByIndex[i] = line
    i = i+1

fileIndex = 10001
iterIndex = 0
for iterIndex in range(200):
    
    queryLabelFile = open("C:\\study\\Second Quarter\\Recall\\RecallCNN-master\\Query\\CNNlabel_0"+str(fileIndex)+".txt",'r')

    matQuery = dict()
    i = 0
    instanceRetrieved = 0

    for line in queryLabelFile :
        #instanceRetrieved = instanceRetrieved + int(matByIndex[int(line)])
        matQuery[i] = line
        i = i+1

    print ("line is " + str(type(matQuery[0])))

    #Information Retrieval
    irIndex = dict()
    k=0
    for lab in range(len(matByIndex)):
    
        for p in range(len(matQuery)):
        
            if matQuery[p] == matByIndex[lab] :
            
                irIndex[k] = lab
                k = k + 1
            

    #Recalling Begins
    prIR = 0
    print(len(irIndex))
    

    for pil in range(len(irIndex)):
    
        truth = matTruthByIndex[irIndex[pil]]
    
        if queryTruthByIndex[iterIndex] == truth :
            prIR = prIR + 1

    print ("Precision is " + str(prIR / len(irIndex)))

    print(iterIndex)
    if queryTruthByIndex[iterIndex] in matTruthByLabel:
        temp = matTruthByLabel[str(queryTruthByIndex[iterIndex])]
        
    print ("Recall is " + str(prIR/temp ))
    fileIndex = fileIndex+ 1

