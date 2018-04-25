import tensorflow as tf
import numpy as np
from Basic_function import conv_layer,Max_pooling,Fully_connected,Relu,SqueezeExcitation,Avg_pooling,Dropout,Global_Average_Pooling

class SENet_Residual:

    def __init__(self,input_shape,dataset,batch_size,epoch,learning_rate):
        self.input_shape=input_shape
        self.x_input = input_shape[0]
        self.y_input = input_shape[1]
        self.channel = input_shape[2]
        self.dataset=dataset
        self.batch_size=batch_size
        self.epoch=epoch
        self.learning_rate=learning_rate
        self.checkpoint_path='./save/SENet_Residual'+dataset+'Epoch'+str(self.epoch)+'.ckpt'
        if dataset=="Cifar":
            self.output_num=10
            self.Train_data="./Data_to_record/CifarRecord/train.tfrecords"
            self.Test_data = "./Data_to_record/CifarRecord/test.tfrecords"
            self.Validation_data = "./Data_to_record/CifarRecord/validation.tfrecords"
        elif dataset=="BlindSpot":
            self.output_num=1
            self.datasource = "./"
        else:
            print ("There is no such dataset. It should be 'Cifar' or 'BlindSpot'.")
        tf.set_random_seed(1337)
        np.random.seed(1337)


    def Data_input(self,data_path,Epoch=1):
        feature = {'example': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=Epoch)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['example'], tf.float32)

        # Cast label data into int32
        label = tf.cast(features['label'], tf.int32)
        # Reshape image data into the original shape
        image = tf.reshape(image, self.input_shape)

        # Any preprocessing here ...

        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.shuffle_batch([image, label], batch_size=self.batch_size, capacity=100, num_threads=1,
                                                min_after_dequeue=10)
        return (images,labels)


    def model_preprocess(self):
        Mean_RGB = [0, 0, 0]
        with tf.Session() as sess:
            images, labels = self.Data_input(data_path=self.Train_data, Epoch=1)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            count=0.0
            for j in range(100000):
                try:
                    img, lbl = sess.run((images, labels))
                    lbl = np.reshape(lbl, (self.batch_size, 1))
                except:
                    break
                for m in range(self.batch_size):
                    assert np.shape(img[m])==(32,32,3)
                    count=count+1
                    Mean_RGB[0] = Mean_RGB[0] + np.mean(img[m][:,:,0])
                    Mean_RGB[1] = Mean_RGB[1] + np.mean(img[m][:, :, 1])
                    Mean_RGB[2] = Mean_RGB[2] + np.mean(img[m][:, :, 2])
            Mean_RGB[0]=Mean_RGB[0]/count
            Mean_RGB[1] = Mean_RGB[1] / count
            Mean_RGB[2] = Mean_RGB[2] / count
            coord.request_stop()
            # Wait for threads to stop
            coord.join(threads)
        return Mean_RGB


    def model_structure(self):
        self.img = tf.placeholder(tf.float32, shape=(None, self.x_input, self.y_input, self.channel))
        self.label = tf.placeholder(tf.int32, shape=(None, 1))
        self.keep_prob=tf.placeholder(tf.float32)
        if self.dataset=='Cifar':
            self.conv_1 = conv_layer(self.img, filter=64, kernel=[3, 3], stride=1, padding='SAME', layer_name="conv_1")
            self.Relu_1=Relu(self.conv_1)
            self.Pool_1 = Max_pooling(self.Relu_1, pool_size=[2, 2], stride=2, padding='VALID')
        if self.dataset=='BlindSpot':
            self.conv_1 = conv_layer(self.img, filter=64, kernel=[7, 7], stride=2, padding='SAME', layer_name="conv_1")
            self.Relu_1 = Relu(self.conv_1)
            self.Pool_1 = Max_pooling(self.Relu_1, pool_size=[2, 2], stride=2, padding='VALID')


        self.BottleNeck_1=conv_layer(self.Pool_1, filter=128, kernel=[1, 1], stride=1, padding='SAME', layer_name="bottle_1")
        self.conv_2 = conv_layer(self.BottleNeck_1, filter=128, kernel=[3, 3], stride=1, padding='SAME', layer_name="conv_2")
        self.SElayer_1 = SqueezeExcitation(self.conv_2, input_channel=128, R=8, batch_size=self.batch_size,se_name="SE_1")
        self.Residul_1 = self.BottleNeck_1 + self.SElayer_1
        self.Relu_2 = Relu(self.Residul_1)

        self.conv_3 = conv_layer(self.Relu_2, filter=128, kernel=[3, 3], stride=1, padding='SAME', layer_name="conv_3")
        self.SElayer_2=SqueezeExcitation(self.conv_3, input_channel=128, R=8, batch_size=self.batch_size,se_name="SE_2")
        self.Residul_2=self.conv_3+self.SElayer_2
        self.Relu_3 = Relu(self.Residul_2)

        self.BottleNeck_2 = conv_layer(self.Relu_3, filter=256, kernel=[1, 1], stride=1, padding='SAME',layer_name="bottle_2")
        self.conv_4 = conv_layer(self.BottleNeck_2, filter=256, kernel=[3, 3], stride=1, padding='SAME', layer_name="conv_4")
        self.SElayer_3 = SqueezeExcitation(self.conv_4, input_channel=256, R=8, batch_size=self.batch_size,se_name="SE_3")
        self.Residul_3 = self.BottleNeck_2 + self.SElayer_3
        self.Relu_4 = Relu(self.Residul_3)

        self.conv_5 = conv_layer(self.Relu_4, filter=256, kernel=[3, 3], stride=1, padding='SAME', layer_name="conv_5")
        self.SElayer_4 = SqueezeExcitation(self.conv_5, input_channel=256, R=8, batch_size=self.batch_size,se_name="SE_4")
        self.Residul_4 = self.conv_5 + self.SElayer_4
        self.Relu_5 = Relu(self.Residul_4)

        self.Pool_2 = Avg_pooling(self.Relu_5, pool_size=[2, 2], stride=2, padding='VALID')
        self.flatten = tf.layers.flatten(self.Pool_2)
        self.FC_0 = Fully_connected(self.flatten, out_num=1024, layer_name='fc_0')
        self.Relu_0 = Relu(self.FC_0)
        self.FC_1 = Fully_connected(self.Relu_0, out_num=128, layer_name='fc_1')
        self.Relu_6 = Relu(self.FC_1)
        self.Dropout_1 = Dropout(self.Relu_6, self.keep_prob)
        self.FC_2 = Fully_connected(self.Relu_6, out_num=32, layer_name='fc_2')
        self.Relu_7 = Relu(self.FC_2)
        #self.Dropout_2 = Dropout(self.Relu_7, self.keep_prob)
        self.FC_3 = Fully_connected(self.Relu_7, out_num=self.output_num, layer_name='fc_3')

        if self.output_num == 1:
            self.prob = tf.nn.sigmoid(self.FC_3, name="prob")
        else:
            self.prob = tf.nn.softmax(self.FC_3, name="prob")
        return self.prob

    def fit(self):
        tf.reset_default_graph()
        self.RGB_mean_value =self.model_preprocess()
        Output_Probability= self.model_structure()
        if self.output_num==1:
            cross_entropy=-tf.reduce_mean(tf.multiply(self.label,tf.log(Output_Probability+0.0001))+tf.multiply((1.0-Output_Probability),tf.log(1.0-self.prob+0.0001)))
        else:
            self.one_hot=tf.one_hot(indices=self.label,depth=self.output_num)
            self.one_hot=tf.squeeze(self.one_hot,[1])

            cross_entropy=-tf.reduce_mean(tf.reduce_mean(tf.multiply(tf.log(self.prob+0.0001),self.one_hot),axis=1),axis=0)

        Learning_rate=tf.placeholder(tf.float32,None)
        trainer=tf.train.AdamOptimizer(Learning_rate)
        gvs=trainer.compute_gradients(cross_entropy)
        train_step = trainer.apply_gradients(gvs)

        '''
        def ClipGradient(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad,-10,10)
        clip_gradient=[]
        for grad,var in gvs:
            clip_gradient.append((ClipGradient(grad),var))

        train_step=trainer.apply_gradients(clip_gradient)
        '''
        with tf.Session() as sess:
            train_accuracy = 0
            Train_Step = 0

            images, labels = self.Data_input(data_path=self.Train_data, Epoch=self.epoch)
            vali_images, vali_labels = self.Data_input(data_path=self.Validation_data, Epoch=100)

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            def Validation(step):
                validation_accuracy = 0
                validation_Step = 0
                for j in range(100):
                    try:
                        img, lbl = sess.run([vali_images, vali_labels])
                        img[:, :, :, 0] = img[:, :, :, 0] - self.RGB_mean_value[0]
                        img[:, :, :, 1] = img[:, :, :, 1] - self.RGB_mean_value[1]
                        img[:, :, :, 2] = img[:, :, :, 2] - self.RGB_mean_value[2]
                        lbl = np.reshape(lbl, (self.batch_size, 1))
                    except:
                        break
                    validation_Step = validation_Step + 1
                    Probability = Output_Probability.eval(
                        feed_dict={self.img: img, self.label: lbl, self.keep_prob: 1.0})
                    Probability = np.asarray(Probability)
                    if self.output_num == 1:
                        assert np.shape(Probability) == (self.batch_size, 1)
                        validation_accuracy = validation_accuracy + np.mean(
                            (Probability > 0.5) * lbl + (Probability <= 0.5) * (1 - lbl))
                    else:
                        assert np.shape(Probability) == (self.batch_size, self.output_num)
                        validation_accuracy = validation_accuracy + +np.mean(
                            np.reshape(np.argmax(Probability, axis=1), (self.batch_size, 1)) == lbl)
                print("Step %d validation accuracy is: %f" % (step, validation_accuracy / validation_Step))

            for j in range(10000000):
                try:
                    img, lbl = sess.run((images, labels))
                    img[:, :, :, 0] = img[:, :, :, 0] - self.RGB_mean_value[0]
                    img[:, :, :, 1] = img[:, :, :, 1] - self.RGB_mean_value[1]
                    img[:, :, :, 2] = img[:, :, :, 2] - self.RGB_mean_value[2]
                    lbl=np.reshape(lbl,(self.batch_size,1))
                except:
                    break
                Train_Step=Train_Step+1
                if j%100==0 and j>5000:
                    self.learning_rate=self.learning_rate*0.99
                train_step.run(feed_dict={self.img:img,self.label:lbl,Learning_rate:self.learning_rate,self.keep_prob:0.6})
                Probability=Output_Probability.eval(feed_dict={self.img:img,self.label:lbl,self.keep_prob:1.0})
                Probability=np.asarray(Probability)

                if self.output_num==1:
                    assert np.shape(Probability)==(self.batch_size,1)
                    train_accuracy=train_accuracy+np.mean((Probability>0.5)*lbl+(Probability<=0.5)*(1-lbl))
                else:
                    assert np.shape(Probability)==(self.batch_size,self.output_num)

                    train_accuracy=train_accuracy+np.mean(np.reshape(np.argmax(Probability,axis=1),(self.batch_size,1))==lbl)
                if j%100==0 and j>0:
                    print ("Step %d accuracy is : %f"%(int(j),train_accuracy/j))
                if j%5000==0 and j>1:
                    Validation(j)
            print ("Epoch %d training accuracy is: %f"%(self.epoch,train_accuracy/Train_Step))
            coord.request_stop()
            # Wait for threads to stop
            coord.join(threads)
            saver=tf.train.Saver()
            saver.save(sess,self.checkpoint_path)




    def test_check(self):
        tf.reset_default_graph()
        self.RGB_mean_value = self.model_preprocess()
        Output_Probability = self.model_structure()
        with tf.Session() as sess:
            test_accuracy = 0
            Test_Step = 0
            test_images, test_labels = self.Data_input(data_path=self.Test_data, Epoch=1)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            saver=tf.train.Saver()
            saver.restore(sess,self.checkpoint_path)
            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for j in range(1000000):
                try:
                    img, lbl = sess.run([test_images, test_labels])
                    img[:, :, :, 0] = img[:, :, :, 0] - self.RGB_mean_value[0]
                    img[:, :, :, 1] = img[:, :, :, 1] - self.RGB_mean_value[1]
                    img[:, :, :, 2] = img[:, :, :, 2] - self.RGB_mean_value[2]
                    lbl = np.reshape(lbl, (self.batch_size, 1))
                except:
                    break
                Test_Step = Test_Step + 1
                Probability = Output_Probability.eval(feed_dict={self.img: img, self.label: lbl,self.keep_prob:1.0})
                Probability = np.asarray(Probability)
                if self.output_num == 1:
                    assert np.shape(Probability) == (self.batch_size, 1)
                    test_accuracy = test_accuracy + np.mean(
                        (Probability > 0.5) * lbl + (Probability <= 0.5) * (1 - lbl))
                else:
                    assert np.shape(Probability) == (self.batch_size, self.output_num)
                    test_accuracy = test_accuracy + +np.mean(
                        np.reshape(np.argmax(Probability, axis=1), (self.batch_size, 1)) == lbl)
            print("Epoch %d test accuracy is: %f" % (self.epoch, test_accuracy / Test_Step))
            coord.request_stop()
            # Wait for threads to stop
            coord.join(threads)


    def predict(self):
        pass