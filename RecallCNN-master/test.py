
import numpy as np
import datetime
import os
import shutil
import PIL


v_data_dir = "/home/swapnil/291G/partData/"
truthLabelDict = dict()
truth = open(v_data_dir+'labels.txt','r')
k=0
for tLabel in truth:
    #print(tLabel)
    if not tLabel or tLabel.isspace():
        continue
    truthLabelDict[int(k+1)] = tLabel
    k=k+1
    
print(len(truthLabelDict))
