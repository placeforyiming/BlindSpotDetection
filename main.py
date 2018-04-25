
from model_VGG import VGG
from model_Depthwise_Residual import Depthwise_Residual
from model_Depthwise_SENet import Depthwise_SENet
from model_Depthwise_SENet_residual import Depthwise_SENet_Residual


input_shape=(64,128,3)
dataset="BlindSpot"
batch_size=64
epoch=30
learning_rate=0.00001




if (0):
    print("Model Depthwise SENet Residual")
    Model=Depthwise_SENet_Residual(input_shape,dataset,batch_size,epoch,learning_rate)
    #Model.fit()
    Model.test_check()

if (0):
    print ("Model Depthwise Residual")
    Model = Depthwise_Residual(input_shape, dataset, batch_size, epoch, learning_rate)
    #Model.fit()
    Model.test_check()


if (0):
    print ("Model VGG:")
    Model=VGG(input_shape,dataset,batch_size,epoch,learning_rate)
    #Model.fit()
    Model.test_check()



if (0):
    print ("Model Depthwise_SENet")
    Model = Depthwise_SENet(input_shape, dataset, batch_size, epoch, learning_rate)
    #Model.fit()
    Model.test_check()



input_shape=(32,32,3)
dataset="Cifar"
batch_size=64
epoch=100
learning_rate=0.001




if (0):
    print("Model Depthwise SENet Residual")
    Model=Depthwise_SENet_Residual(input_shape,dataset,batch_size,epoch,learning_rate)
    #Model.fit()
    Model.test_check()

if (0):
    print ("Model Depthwise Residual")
    Model = Depthwise_Residual(input_shape, dataset, batch_size, epoch, learning_rate)
    #Model.fit()
    Model.test_check()


if (0):
    print ("Model VGG:")
    Model=VGG(input_shape,dataset,batch_size,epoch,learning_rate)
    #Model.fit()
    Model.test_check()



if (0):
    print ("Model Depthwise_SENet")
    Model = Depthwise_SENet(input_shape, dataset, batch_size, epoch, learning_rate)
    #Model.fit()
    Model.test_check()




