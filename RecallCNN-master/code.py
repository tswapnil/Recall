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
TopModel = getModel(-1)

target = open('Predictions'  + '.txt', 'w')

v_data_dir = "/home/swapnil/Downloads/partVal/"

#n = len([nam for nam in os.listdir(v_data_dir) if os.path.isfile(os.path.join(v_data_dir, nam))])
truthLabelDict = dict()
truth = open(v_data_dir+'labels.txt','r')
k=0
for tLabel in truth:
    if not tLabel or tLabel.isspace():
        continue
    truthLabelDict[int(k+1)] = tLabel
    k=k+1

#print (truthLabelDict)

n= 100
print("Initializing xy Array ")
xy = np.zeros(shape = (n, 224, 224, 3))
labelIndexArr = np.zeros(shape = (n, 1))

i=0
for root, d, files in os.walk(v_data_dir):
    for name in files:
        imagePath = str(v_data_dir) +str(name)
        if not imagePath.endswith("JPEG"):
            continue
        img = image_utils.load_img(imagePath,target_size=(224, 224)) 
        temp = image_utils.img_to_array(img)
        temp = np.expand_dims(temp, axis=0)
        temp = preprocess_input(temp)
        preds = TopModel.predict(temp)
        P = decode_predictions(preds)
        (imagenetID, label, prob) = P[0][0]
        print("Predicted Label : " + str(label))
        print("ImageNet ID : " + str(imagenetID))
        print("Value Prob : "+ str(prob))
        #print(preds)
        fName = str(name)
        labelIndexArr[i] = int(fName[17:-5])
        truthLabelVal = truthLabelDict[int(labelIndexArr[i])]
        print("Ground Label :" + str(truthLabelVal))
        target.write(str(imagenetID) + ", " + str(prob) + " , "+ str(label)+ " , " + str(truthLabelVal) +"\n")
        i = i+1
        

#TimeStart = datetime.datetime.now()
#print (TimeStart)
#output = TopModel.predict(xy, batch_size=32, verbose=1)
#TimeEnd = datetime.datetime.now()
#print (TimeEnd)
#print ("Inference Time =  " + str((TimeEnd - TimeStart).total_seconds() / 60))

#j=0
#maxLabel = np.zeros(shape= (n,1))
#for textLine in output:
    #maxLabel[j] = np.argmax(textLine)+1
    #print("Value of j = "+str(j))
    #print(int(labelIndexArr[j]))
    #truthLabelVal = truthLabelDict[int(labelIndexArr[j])]
    #truthLabelVal = 0
    #print(truthLabelVal)
    #target.write(str(labelIndexArr[j])+", " +str(maxLabel[j])+ str(truthLabelVal) +"\n")
    #j=j+1


#v_datagen = ImageDataGenerator()
 
#v_generator = v_datagen.flow_from_directory(
#        v_data_dir,
#        target_size=(224, 224),
#        batch_size=32,
#        class_mode='categorical',
#        shuffle = False
#         )
#output = TopModel.predict_generator(v_generator,val_samples=49999 )
#print (np.argmax(output,axis=1))

#TopModel.summary()
