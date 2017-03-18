import numpy as np
import datetime
import os
import shutil
from PIL import Image
import pickle


truth = open('C:\\study\\Second Quarter\\Recall\\newData\\labels.txt', 'r')

stats = open('stats.txt', 'w')
mat = dict()
for line in truth:
    if line in mat: 
        mat[line] +=1
    else :
        mat[line] = 1

#stats.write(str(mat))

#print (mat)
