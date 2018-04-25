
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SHAPE=(64,128,3)
Epoch=1
batch_size=32
data_path = './BlindSpotRecord/test.tfrecords'  # address to save the hdf5 file


with tf.Session() as sess:
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
    image = tf.reshape(image,IMAGE_SHAPE)

    # Any preprocessing here ...


    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=100, num_threads=1,
                                            min_after_dequeue=10)


    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(5):
        try:
            img, lbl = sess.run([images, labels])
        except:
            break

        img = img.astype(np.uint8)

        plt.imshow(img[1])

        plt.show()
    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()