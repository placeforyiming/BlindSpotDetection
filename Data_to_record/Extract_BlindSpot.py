
import PreProcess
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.utils import shuffle

train_neg_path="./BlindSpot/Training_image_1/Training_image_1/neg/"
train_pos_path="./BlindSpot/Training_image_1/Training_image_1/pos_strict/"
train_vague_path="./BlindSpot/Training_image_2/Training_image_2/0001/"

Train_image=[]
Train_lable=[]
images=glob.glob(train_neg_path+"*.jpg")
for i in images:
    Train_image.append(plt.imread(i).flatten())
    Train_lable.append(0)
images=glob.glob(train_pos_path+"*.jpg")
for i in images:
    Train_image.append(plt.imread(i).flatten())
    Train_lable.append(1)



Vague_image=[]
Vague_label=[]
images=glob.glob(train_vague_path+"*.jpg")
for i in images:
    Vague_image.append(plt.imread(i).flatten())
    Vague_label.append(1)

X_train,x_validation_strict,Y_train,y_validation_strict=train_test_split(Train_image,Train_lable,test_size=0.1)

X_train_sup,x_validation_sup,Y_train_sup,y_validation_sup=train_test_split(Vague_image,Vague_label,test_size=0.1)


X_train=X_train+X_train_sup
x_validation=x_validation_strict+x_validation_sup
Y_train=Y_train+Y_train_sup
y_validation=y_validation_strict+y_validation_sup

X_train,Y_train=shuffle(X_train,Y_train,random_state=0)

x_validation,y_validation=shuffle(x_validation,y_validation,random_state=0)

x_validation_strict,y_validation_strict=shuffle(x_validation_strict,y_validation_strict,random_state=0)

Dict_train={'data':X_train,'labels':Y_train}
Dict_validation={'data':x_validation,'labels':y_validation}
Dict_validation_strict={'data':x_validation_strict,'labels':y_validation_strict}


PreProcess.write_record("./BlindSpotRecord/train.tfrecords",Dict_train)
PreProcess.write_record("./BlindSpotRecord/validation.tfrecords",Dict_validation)
PreProcess.write_record("./BlindSpotRecord/validation_strict.tfrecords",Dict_validation_strict)


Test_image_strict=[]
Test_label_strict=[]
Test_image=[]
Test_label=[]

test_neg_path="./BlindSpot/Testing_image_1/Testing_image_1/neg/"
test_pos_path="./BlindSpot/Testing_image_1/Testing_image_1/pos/"
test_pos_strict_path="./BlindSpot/Testing_image_1/Testing_image_1/pos_strict/"

images=glob.glob(test_neg_path+"*.jpg")
for i in images:
    Test_image.append(plt.imread(i).flatten())
    Test_label.append(0)
    Test_image_strict.append(plt.imread(i).flatten())
    Test_label_strict.append(0)
images=glob.glob(test_pos_path+"*.jpg")
for i in images:
    Test_image.append(plt.imread(i).flatten())
    Test_label.append(1)
images=glob.glob(test_pos_strict_path+"*.jpg")
for i in images:
    Test_image_strict.append(plt.imread(i).flatten())
    Test_label_strict.append(1)


Dict_test={'data':Test_image,'labels':Test_label}
Dict_test_strict={'data':Test_image_strict,'labels':Test_label_strict}

PreProcess.write_record("./BlindSpotRecord/test.tfrecords",Dict_test)
PreProcess.write_record("./BlindSpotRecord/test_strict.tfrecords",Dict_test_strict)



























