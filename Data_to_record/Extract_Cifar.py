import pickle
import PreProcess
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

Dict1=unpickle("/home/yiming/PycharmProjects/BlindSpotDetection/Data_to_record/cifar-10-batches-py/data_batch_1")
Dict2=unpickle("/home/yiming/PycharmProjects/BlindSpotDetection/Data_to_record/cifar-10-batches-py/data_batch_2")
Dict3=unpickle("/home/yiming/PycharmProjects/BlindSpotDetection/Data_to_record/cifar-10-batches-py/data_batch_3")
Dict4=unpickle("/home/yiming/PycharmProjects/BlindSpotDetection/Data_to_record/cifar-10-batches-py/data_batch_4")
Dict5=unpickle("/home/yiming/PycharmProjects/BlindSpotDetection/Data_to_record/cifar-10-batches-py/data_batch_5")
Dict6=unpickle("/home/yiming/PycharmProjects/BlindSpotDetection/Data_to_record/cifar-10-batches-py/test_batch")

Dict={'data':[],'labels':[]}

for i in range(len(Dict1[b'data'])):
    Dict['data'].append(np.transpose(np.reshape(Dict1[b'data'][i],(3,32,32)),(1,2,0)).flatten())
    Dict['labels'].append(Dict1[b'labels'][i])

for i in range(len(Dict2[b'data'])):
    Dict['data'].append(np.transpose(np.reshape(Dict2[b'data'][i],(3,32,32)),(1,2,0)).flatten())
    Dict['labels'].append(Dict2[b'labels'][i])

for i in range(len(Dict3[b'data'])):
    Dict['data'].append(np.transpose(np.reshape(Dict3[b'data'][i],(3,32,32)),(1,2,0)).flatten())
    Dict['labels'].append(Dict3[b'labels'][i])

for i in range(len(Dict4[b'data'])):
    Dict['data'].append(np.transpose(np.reshape(Dict4[b'data'][i],(3,32,32)),(1,2,0)).flatten())
    Dict['labels'].append(Dict4[b'labels'][i])

for i in range(len(Dict5[b'data'])):
    Dict['data'].append(np.transpose(np.reshape(Dict5[b'data'][i],(3,32,32)),(1,2,0)).flatten())
    Dict['labels'].append(Dict5[b'labels'][i])


X_train,x_validation,Y_train,y_validation=train_test_split(Dict['data'],Dict['labels'],test_size=0.1)


Dict_train={'data':X_train,'labels':Y_train}
Dict_validation={'data':x_validation,'labels':y_validation}


PreProcess.write_record("./CifarRecord/train.tfrecords",Dict_train)
PreProcess.write_record("./CifarRecord/validation.tfrecords",Dict_validation)


Dict={'data':[],'labels':[]}
for i in range(len(Dict6[b'data'])):
    Dict['data'].append(np.transpose(np.reshape(Dict6[b'data'][i],(3,32,32)),(1,2,0)).flatten())
    Dict['labels'].append(Dict6[b'labels'][i])
PreProcess.write_record("./CifarRecord/test.tfrecords",Dict)