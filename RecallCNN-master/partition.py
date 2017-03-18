import numpy as np
import datetime
import os
import shutil
import subprocess

def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    return iter(p.stdout.readline, b'')

sdir = "/home/swapnil/Downloads/val/"
destDir = "/home/swapnil/Downloads/partVal/"

testFile = open('/home/swapnil/291G/TestOutput.txt', 'w')
labelOFile = open(str(destDir)+'labels.txt','w')
labelIFile = open('/home/swapnil/Downloads/devkit-1.0/data/ILSVRC2010_validation_ground_truth.txt','r')
#strCom = 'ls /home/swapnil/291G/imagenet/test'.split()
p1 =subprocess.Popen(["ls",sdir],stdout=subprocess.PIPE)
p2 = subprocess.Popen(["head","-100"],stdin=p1.stdout, stdout=subprocess.PIPE)
p1.stdout.close()

#(Out1,err)=p2.communicate()
#Out1 = subprocess.getoutput('ls /home/swapnil/291G/imagenet/test | head -100')

for fileName in iter(p2.stdout.readline,b''):
    name = str(fileName)
    name = name[2:-3]
    filePath = sdir + str(name)
    destPath = destDir + str(name)
    p3 = subprocess.Popen(["cp",filePath,destPath],stdout=subprocess.PIPE)
    p3.communicate()
    p3.stdout.close()
    testFile.write(str(name)+"\n")
count = 0
for line in labelIFile :
    if count==100 :
        break
    labelOFile.write(str(line))
    count+=1

    

