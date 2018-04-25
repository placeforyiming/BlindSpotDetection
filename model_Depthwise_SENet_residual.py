import tensorflow as tf
import numpy as np
from Basic_function import Batch_Normalization,conv_layer,conv_layer_Sep,Max_pooling,Fully_connected,Relu,SqueezeExcitation,Avg_pooling,Dropout,Global_Average_Pooling

class Depthwise_SENet_Residual:

    def __init__(self,input_shape,dataset,batch_size,epoch,learning_rate):
        self.input_shape=input_shape
        self.x_input = input_shape[0]
        self.y_input = input_shape[1]
        self.channel = input_shape[2]
        self.dataset=dataset
        self.batch_size=batch_size
        self.epoch=epoch
        self.learning_rate=learning_rate
        self.checkpoint_path='./save/Depthwise_SENet_Residual'+dataset+'Epoch'+str(self.epoch)+'.ckpt'
        if dataset=="Cifar":
            self.output_num=10
            self.Train_data="./Data_to_record/CifarRecord/train.tfrecords"
            self.Test_data = "./Data_to_record/CifarRecord/test.tfrecords"
            self.Validation_data = "./Data_to_record/CifarRecord/validation.tfrecords"
        elif dataset=="BlindSpot":
            self.output_num = 1
            self.Train_data = "./Data_to_record/BlindSpotRecord/train.tfrecords"
            self.Test_data = "./Data_to_record/BlindSpotRecord/test.tfrecords"
            self.Validation_data = "./Data_to_record/BlindSpotRecord/validation.tfrecords"
            self.Test_strict_data = "./Data_to_record/BlindSpotRecord/test_strict.tfrecords"
            self.Validation_strict_data = "./Data_to_record/BlindSpotRecord/validation_strict.tfrecords"
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
                except:
                    break
                if j==0:
                    Train=img
                else:
                    Train=np.concatenate([Train,img],axis=0)
            Mean=np.mean(Train,axis=(0,1,2,3))
            std=np.std(Train,axis=(0,1,2,3))
            coord.request_stop()
            # Wait for threads to stop
            coord.join(threads)
        return (Mean,std)

    def model_structure(self, training):
        self.img = tf.placeholder(tf.float32, shape=(None, self.x_input, self.y_input, self.channel))
        self.label = tf.placeholder(tf.int32, shape=(None, 1))
        self.keep_prob=tf.placeholder(tf.float32)
        self.keep_prob_1 = tf.placeholder(tf.float32)
        if self.dataset=='Cifar':
            self.conv_1 = conv_layer(self.img, filter=64, kernel=[3, 3], stride=1, padding='SAME', layer_name="conv_1")
            self.BN_0 = Batch_Normalization(self.conv_1, training, scope="BN_0")
            self.Relu_1=Relu(self.BN_0)
            self.Pool_1 = Max_pooling(self.Relu_1, pool_size=[2, 2], stride=2, padding='VALID')
        if self.dataset=='BlindSpot':
            self.conv_1 = conv_layer(self.img, filter=64, kernel=[5, 5], stride=2, padding='SAME', layer_name="conv_1")
            self.Relu_1=Relu(self.conv_1)
            self.Pool_1 = Max_pooling(self.Relu_1, pool_size=[2, 2], stride=2, padding='VALID')

        self.BottleNeck_1 = conv_layer(self.Pool_1, filter=128, kernel=[1, 1], stride=1, padding='SAME',layer_name="bottle_1")
        self.BN_1 = Batch_Normalization(self.BottleNeck_1, training, scope="BN_1")
        self.Relu_plus_1 = Relu(self.BN_1)
        self.conv_2 = conv_layer_Sep(self.Relu_plus_1, filter=128, kernel=[3, 3], stride=1, padding='SAME', layer_name="conv_2")
        #self.BN_2 = Batch_Normalization(self.conv_2, training, scope="BN_2")
        self.SElayer_1 = SqueezeExcitation(self.conv_2, input_channel=128, R=16, batch_size=self.batch_size,trainable=training,se_name="SE_1")
        self.Residul_1 = self.BottleNeck_1 + self.SElayer_1
        self.Relu_2 = Relu(self.Residul_1)

        self.BottleNeck_1_1 = conv_layer(self.Relu_2, filter=128, kernel=[1, 1], stride=1, padding='SAME',layer_name="bottle_1_1")
        self.BN_1_1 = Batch_Normalization(self.BottleNeck_1_1, training, scope="BN_1_1")
        self.Relu_plus_1_1 = Relu(self.BN_1_1)
        self.conv_3 = conv_layer_Sep(self.Relu_plus_1_1, filter=128, kernel=[3, 3], stride=1, padding='SAME', layer_name="conv_3")
        #self.BN_3 = Batch_Normalization(self.conv_3, training, scope="BN_3")
        self.SElayer_2 = SqueezeExcitation(self.conv_3, input_channel=128, R=16,batch_size=self.batch_size,trainable=training,se_name="SE_2")
        self.Residul_2 = self.BottleNeck_1_1 + self.SElayer_2
        self.Relu_3 = Relu(self.Residul_2)
        self.Dropout_3 = Dropout(self.Relu_3, self.keep_prob_1)


        self.Pool_0 = Max_pooling(self.Dropout_3, pool_size=[2, 2], stride=2, padding='VALID')

        self.BottleNeck_3 = conv_layer(self.Pool_0, filter=256, kernel=[1, 1], stride=1, padding='SAME',layer_name="bottle_3")
        self.BN_4 = Batch_Normalization(self.BottleNeck_3, training, scope="BN_4")
        self.Relu_plus_2=Relu(self.BN_4)
        self.conv_4 = conv_layer_Sep(self.Relu_plus_2, filter=256, kernel=[3, 3], stride=1, padding='SAME', layer_name="conv_4")
        #self.BN_5 = Batch_Normalization(self.conv_4, training, scope="BN_5")
        self.SElayer_3 = SqueezeExcitation(self.conv_4, input_channel=256, R=16, batch_size=self.batch_size,trainable=training,se_name="SE_3")
        self.Residul_3 = self.BottleNeck_3 + self.SElayer_3
        self.Relu_4 = Relu(self.Residul_3)

        self.BottleNeck_3_3 = conv_layer(self.Relu_4, filter=256, kernel=[1, 1], stride=1, padding='SAME',layer_name="bottle_3_3")
        self.BN_3_3 = Batch_Normalization(self.BottleNeck_3_3, training, scope="BN_3_3")
        self.Relu_plus_2_2 = Relu(self.BN_3_3)
        self.conv_5 = conv_layer_Sep(self.Relu_plus_2_2, filter=256, kernel=[3, 3], stride=1, padding='SAME', layer_name="conv_5")
        #self.BN_6 = Batch_Normalization(self.conv_5, training, scope="BN_6")
        self.SElayer_4 = SqueezeExcitation(self.conv_5, input_channel=256, R=16, batch_size=self.batch_size,trainable=training,se_name="SE_4")
        self.Residul_4 = self.BottleNeck_3_3 + self.SElayer_4
        self.Relu_5 = Relu(self.Residul_4)
        self.Dropout_4 = Dropout(self.Relu_5, self.keep_prob_1)

        self.Pool_2 = Avg_pooling(self.Dropout_4, pool_size=[2, 2], stride=2, padding='VALID')
        self.flatten = tf.layers.flatten(self.Pool_2)
        self.FC_0 = Fully_connected(self.flatten, out_num=1024, layer_name='fc_0')
        self.Relu_0 = Relu(self.FC_0)
        self.Dropout_1 = Dropout(self.Relu_0, self.keep_prob)
        self.FC_1 = Fully_connected(self.Dropout_1, out_num=128, layer_name='fc_1')
        self.Relu_6 = Relu(self.FC_1)
        self.Dropout_2 = Dropout(self.Relu_6, self.keep_prob)

        if self.output_num == 1:
            self.FC_2 = Fully_connected(self.Dropout_2, out_num=32, layer_name='fc_2')
            self.Relu_7 = Relu(self.FC_2)
            self.FC_3 = Fully_connected(self.Relu_7, out_num=self.output_num, layer_name='fc_3')
            self.prob = tf.nn.sigmoid(self.FC_3, name="prob")
        else:
            self.FC_2 = Fully_connected(self.Dropout_2, out_num=self.output_num, layer_name='fc_2')
            self.prob = tf.nn.softmax(self.FC_2, name="prob")
        return self.prob

    def fit(self):
        tf.reset_default_graph()
        self.Mean, self.std = self.model_preprocess()
        Output_Probability = self.model_structure(training=True)
        if self.output_num==1:
            cross_entropy = -tf.reduce_mean(tf.multiply(tf.cast(self.label, tf.float32), tf.log(Output_Probability + 0.0001)) + tf.multiply((1.0 - tf.cast(self.label, tf.float32)), tf.log(1.0 - Output_Probability + 0.0001)))
        else:
            self.one_hot=tf.one_hot(indices=self.label,depth=self.output_num)
            self.one_hot=tf.squeeze(self.one_hot,[1])

            cross_entropy=-tf.reduce_mean(tf.reduce_mean(tf.multiply(tf.log(self.prob+0.0001),self.one_hot),axis=1),axis=0)

        l2_loss = tf.losses.get_regularization_loss()
        cross_entropy = cross_entropy + l2_loss

        Learning_rate=tf.placeholder(tf.float32,None)
        trainer=tf.train.AdamOptimizer(Learning_rate)
        gvs=trainer.compute_gradients(cross_entropy)
        #train_step = trainer.apply_gradients(gvs)


        def ClipGradient(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad,-1,1)
        clip_gradient=[]
        for grad,var in gvs:
            clip_gradient.append((ClipGradient(grad),var))

        train_step=trainer.apply_gradients(clip_gradient)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            train_accuracy = 0
            Train_Step = 0

            images, labels = self.Data_input(data_path=self.Train_data, Epoch=self.epoch)
            vali_images, vali_labels = self.Data_input(data_path=self.Validation_data, Epoch=100)

            if self.dataset=="BlindSpot":
                vali_images_strict, vali_labels_strict = self.Data_input(data_path=self.Validation_strict_data, Epoch=200)

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            if self.dataset == "BlindSpot":
                def Validation_strict(step):
                    validation_accuracy = 0
                    validation_Step = 0
                    for j in range(100):
                        try:
                            img, lbl = sess.run([vali_images_strict, vali_labels_strict])
                            img = (img - self.Mean) / (self.std + 0.000001)
                            lbl = np.reshape(lbl, (self.batch_size, 1))
                        except:
                            break
                        validation_Step = validation_Step + 1
                        Probability = Output_Probability.eval(
                            feed_dict={self.img: img, self.label: lbl, self.keep_prob: 1.0, self.keep_prob_1: 1.0})
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
                    return (validation_accuracy / validation_Step)

            def Validation(step):
                validation_accuracy = 0
                validation_Step = 0
                for j in range(100):
                    try:
                        img, lbl = sess.run([vali_images, vali_labels])
                        img = (img - self.Mean) / (self.std + 0.000001)
                        lbl = np.reshape(lbl, (self.batch_size, 1))
                    except:
                        break
                    validation_Step = validation_Step + 1
                    Probability = Output_Probability.eval(
                        feed_dict={self.img: img, self.label: lbl, self.keep_prob: 1.0,self.keep_prob_1:1.0})
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
                    img = (img - self.Mean) / (self.std + 0.000001)
                    lbl = np.reshape(lbl, (self.batch_size, 1))
                except:
                    break
                Train_Step=Train_Step+1
                if j%100==0 and j>5000:
                    self.learning_rate=self.learning_rate*0.99
                    self.learning_rate=np.maximum(self.learning_rate,0.00005)
                    if j>35000:
                        self.learning_rate=0.00001
                    if j>60000:
                        self.learning_rate=0.000001

                train_step.run(feed_dict={self.img:img,self.label:lbl,Learning_rate:self.learning_rate,self.keep_prob:0.5,self.keep_prob_1:0.7})
                Probability=Output_Probability.eval(feed_dict={self.img:img,self.label:lbl,self.keep_prob:1.0,self.keep_prob_1:1.0})
                Probability=np.asarray(Probability)

                if self.output_num==1:
                    assert np.shape(Probability)==(self.batch_size,1)
                    train_accuracy=train_accuracy+np.mean((Probability>0.5)*lbl+(Probability<=0.5)*(1-lbl))
                else:
                    assert np.shape(Probability)==(self.batch_size,self.output_num)

                    train_accuracy=train_accuracy+np.mean(np.reshape(np.argmax(Probability,axis=1),(self.batch_size,1))==lbl)
                if j%100==0 and j>0:
                    print ("Step %d accuracy is : %f"%(int(j),train_accuracy/j))

                thresh=0.0
                if j%5000==0 and j>1 and self.dataset=="Cifar":
                    Validation(j)
                if self.dataset=="BlindSpot" and j%500==0 and j>0:
                    Validation(j)
                    thresh=Validation_strict(j)
                    #if thresh>0.99:
                    #    break

            print ("Epoch %d training accuracy is: %f"%(self.epoch,train_accuracy/Train_Step))
            coord.request_stop()
            # Wait for threads to stop
            coord.join(threads)
            saver=tf.train.Saver()
            saver.save(sess,self.checkpoint_path)




    def test_check(self):
        tf.reset_default_graph()
        self.Mean, self.std = self.model_preprocess()
        Output_Probability = self.model_structure(training=False)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            test_accuracy = 0
            Test_Step = 0
            if self.dataset=="BlindSpot":
                test_images, test_labels = self.Data_input(data_path=self.Test_strict_data, Epoch=1)
            else:
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
                    img = (img - self.Mean) / (self.std + 0.000001)
                    lbl = np.reshape(lbl, (self.batch_size, 1))
                except:
                    break
                Test_Step = Test_Step + 1
                Probability = Output_Probability.eval(feed_dict={self.img: img, self.label: lbl,self.keep_prob:1.0,self.keep_prob_1:1.0})
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


    def Cal_pre(self):
        self.Mean, self.std = self.model_preprocess()
        self.Output_Probability = self.model_structure(training=False)
        sess=tf.Session()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        saver=tf.train.Saver()
        saver.restore(sess,self.checkpoint_path)
        return (self.Mean, self.std, sess)

    def predict(self,input_batch, Mean, td,sess):
        Probability = sess.run(self.Output_Probability,feed_dict={self.img: input_batch,self.keep_prob:1.0,self.keep_prob_1:1.0})
        Probability = np.asarray(Probability)
        return Probability

        
        

        