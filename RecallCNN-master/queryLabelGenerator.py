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

def getModel(index):
    '''
        * output_dim: the number of classes (int)

        * return: compiled model (keras.engine.training.Model)
    '''
    print ("Inside getModel")
    vgg_model = VGG16( weights='imagenet', include_top=True )
    print (" Got the model ")
    #vgg_out = vgg_model.layers[-2].output #Last FC layer's output
    #print ("Got vgg_out")
    #softmax_layer = Dense(output_dim, activation='softmax')(vgg_out); #Create softmax layer taking input as vgg_ou
    #print("Got the softmax layer")
    #Create new transfer learning model
    #tl_model = Model( input=vgg_model.input, output=softmax_layer )
    #print ("Generated the tl_model")
    #tl_model = vgg_model;
    #Freeze all layers of VGG16 and Compile the model
    #Confirm the model is appropriate
    for layer in vgg_model.layers:
        layer.trainable = False
    return Model(input = vgg_model.input , output= vgg_model.layers[index].output)

#FlatModel = getModel(-4)
TopModel = VGG16( weights='imagenet', include_top=True )



v_data_dir = "C:\\study\\Second Quarter\\Recall\\RecallCNN-master\\Query\\"

#n = len([nam for nam in os.listdir(v_data_dir) if os.path.isfile(os.path.join(v_data_dir, nam))])
truthLabelDict = dict()
truth = open(v_data_dir+'labels.txt','r')
k=0
for tLabel in truth:
    if not tLabel or tLabel.isspace():
        continue
    truthLabelDict[int(k+1)] = int(tLabel)
    k=k+1


predMat = dict()
n= 200
print("Initializing xy Array ")
#xy = np.zeros(shape = (n, 224, 224, 3))
labelIndexArr = np.zeros(shape = (n, 1))
par = 5;
i=0
accuracy = 0;
accuracy5 = 0;
recall = 0;
precision = 0;
for root, d, files in os.walk(v_data_dir):
    for name in files:
        print(i)
        imagePath = str(v_data_dir) +str(name)
        if not imagePath.endswith("JPEG"):
            continue
        img = image_utils.load_img(imagePath,target_size=(224, 224)) 
        temp = image_utils.img_to_array(img)
        temp = np.expand_dims(temp, axis=0)
        temp = preprocess_input(temp)
        preds = TopModel.predict(temp)
        P = decode_predictions(preds,par)
        (imagenetID, label, prob, index) = P[0][0]
        print("Predicted Label : " + str(label))
        print("ImageNet ID : " + str(imagenetID))
        print("Value Prob : "+ str(prob))
        print("Predicted Label : "+ str(index))
        if index in predMat :
            predMat[index] +=1
        else :
            predMat[index] = 1
        #print(preds)
        fName = str(name)
     
      
        allLabels = dict()
        al = 0
        for y in range(0,par):
            (a,b,c,d) = P[0][y]
            allLabels[al] = d
            al = al+1
        al =0
        target = open(v_data_dir+'CNNLabel_'+ str(fName[17:-5])  + '.txt', 'w')
        #target.write(str(imagenetID) + ", " + str(prob) + " , "+ str(label)+ " , " + str(truthLabelVal) +"\n")
        for indLabel in range(len(allLabels)) :
            if indLabel == len(allLabels)-1 :
                target.write(str(allLabels[al]))
            else :
                target.write(str(allLabels[al])+'\n')
            al = al+1
        i = i+1




        

