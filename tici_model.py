import numpy as np
import tensorflow as tf
from preprocess import append_csv_features, get_mips_data, TICI_MAP
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Embedding

class TICI_Model(tf.keras.Model):
    def __init__(self, vocab_size, num_classes):

        super(TICI_Model, self).__init__()

        # hyperparameters
        self.sample_shape = (64, 256, 256, 1)
        self.vocab_size = vocab_size
        self.batch_size = 16
        self.learning_rate = 0.002
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.num_classes = num_classes

        self.embedding_size = 64
        self.dropout_rate = 0.2
        self.kernel_size = (3, 9, 9)  # TODO Check order of dimensions, which one is depth
        self.strides = (1, 2, 2)
        self.pool_size = (4, 4, 4)
        self.num_filters_1 = 4
        self.num_filters_2 = 8

        self.dense1_size = 128
        self.dense2_size = 128

        # model for image
        self.conv3d_1 = Conv3D(self.num_filters_1, kernel_size=self.kernel_size, strides=self.strides,
                               activation='relu',
                               input_shape=self.sample_shape)
        self.maxpool_1 = MaxPooling3D(pool_size=self.pool_size, padding='same')
        self.batchnorm_1 = BatchNormalization(center=True, scale=True)
        self.dropout_1 = Dropout(self.dropout_rate)

        self.conv3d_2 = Conv3D(self.num_filters_2, kernel_size=self.kernel_size, strides=self.strides,
                               activation='relu')
        self.maxpool_2 = MaxPooling3D(pool_size=self.pool_size, padding='same')
        self.batchnorm_2 = BatchNormalization(center=True, scale=True)
        self.dropout_2 = Dropout(self.dropout_rate)

        # model for vessel and location
        self.embedding = Embedding(self.vocab_size, self.embedding_size)
        # feedforward layer combined with other features (e.g. age, gender)
        self.dense1 = Dense(self.dense1_size, activation='relu')
        self.dense2 = Dense(self.dense2_size, activation='relu')
        self.dense3 = Dense(self.num_classes, activation='softmax')  # convert to probs in last layer
        # #todo: remove sm if we want to get rid of classification of passes

    def call(self, inputs, is_train=False):
        """
        Runs a forward pass on an input batch of data.
        :param inputs: numpy array of data of length (batch_size)
        :return: logits - a matrix of shape (batch_size, num_classes)
        """

        # images = inputs[:,0]
        images = [row[0] for row in inputs]
        # if is_train:
        #     for i in range(len(images)):
        #         images[i] = tf.image.random_flip_left_right(images[i])

        images = tf.convert_to_tensor(images)
        vessel_text = tf.convert_to_tensor([row[1] for row in inputs])
        other_feats = np.array([row[2:] for row in inputs])
        # call image layer
        # print("starting conv")
        image_out = self.conv3d_1(images)
        image_out = self.batchnorm_1(image_out)
        image_out = self.maxpool_1(image_out)
        # print("finishing conv")
        image_out = self.dropout_1(image_out)
        image_out = self.conv3d_2(image_out)
        image_out = self.batchnorm_2(image_out)
        image_out = self.maxpool_2(image_out)
        # print(image_out.shape)
        # image_out = #(batch_size, 64, 32, 32, 1)

        text_out = self.embedding(vessel_text)
        image_out = tf.reshape(image_out, [len(inputs), -1])
        text_out = tf.reshape(text_out, [len(inputs), -1])

        conjoined = tf.concat([image_out, text_out, other_feats], axis=1)

        # flattened = self.flatten(conjoined)
        # pass thru feedforward
        # print("start feedforward")
        lin_out = self.dense1(conjoined)
        lin_out = self.dense2(lin_out)
        probs = self.dense3(lin_out)
        # print("finish call")

        return probs

    def loss(self, probs, labels):
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
        return loss

    def accuracy(self, probs, labels):
        # correct_predictions = tf.equal(tf.argmax(probs, 1), labels)
        # return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        num_correct = 0
        for i in range(len(labels)):
            if np.argmax(probs[i]) == labels[i]:
                num_correct += 1
        return num_correct / len(labels)

    def multi_accuracy(self, probs, labels):
        # correct_predictions = tf.equal(tf.argmax(probs, 1), labels)
        # return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        num_correct = 0
        for i in range(len(labels)):
            sorted = np.argsort(probs[i])
            print("top choice: " + str(sorted[len(sorted) - 1]) + ", second: " + str(sorted[len(sorted) - 2]))

            if sorted[len(sorted) - 1] == labels[i] or sorted[len(sorted) - 2] == labels[i]:
                num_correct += 1

        return num_correct / len(labels)
