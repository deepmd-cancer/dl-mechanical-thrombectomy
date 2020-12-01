#hyperparameters
import numpy as np
import tensorflow as tf
from preprocess import append_csv_features, get_mips_data, TICI_MAP
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Embedding
from tensorflow.compat.v1 import ConfigProto, InteractiveSession


class Model(tf.keras.Model):
    def __init__(self, vocab_size, num_classes):
        
        super(Model, self).__init__()
        
        # hyperparameters
        self.sample_shape = (64, 256, 256, 1)
        self.vocab_size = vocab_size

        self.num_classes = num_classes

        self.embedding_size = 64
        self.dropout_rate = 0.2
        self.kernel_size = (8, 8, 4) # TODO Check order of dimensions, which one is depth

        self.num_filters_1 = 32
        self.num_filters_2 = 64

        self.dense1_size = 128
        self.dense2_size = 128

        # model for image
        self.conv3d_1 = Conv3D(self.num_filters_1, kernel_size=self.kernel_size, activation='relu',
                         input_shape=self.sample_shape)
        self.maxpool_1 = MaxPooling3D(pool_size=(2, 2, 2))
        self.batchnorm_1 = BatchNormalization(center=True, scale=True)
        self.dropout_1 = Dropout(self.dropout_rate)

        self.conv3d_2 = Conv3D(self.num_filters_2, kernel_size=self.kernel_size, activation='relu')
        self.maxpool_2 = MaxPooling3D(pool_size=(2, 2, 2))
        self.batchnorm_2 = BatchNormalization(center=True, scale=True)

        
        # model for vessel and location
        self.embedding = Embedding(self.vocab_size, self.embedding_size)
        self.flatten = Flatten()
        # feedforward layer combined with other features (e.g. age, gender)
        self.dense1 = Dense(self.dense1_size, activation='relu')
        self.dense2 = Dense(self.dense2_size, activation='relu')
        self.dense3 = Dense(self.num_classes, activation='softmax') # convert to probs in last layer


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of data.
        :param inputs: numpy array of data of length (batch_size)
        :return: logits - a matrix of shape (batch_size, num_classes)
        """

        # images = inputs[:,0]
        images = tf.convert_to_tensor([row[0] for row in inputs])
        vessel_text = [row[1] for row in inputs]
        other_feats = [row[2:] for row in inputs]

        # call image layer
        image_out = self.conv3d_1(images)
        image_out = self.maxpool_1(image_out)
        image_out = self.batchnorm_1(image_out)
        image_out = self.dropout_1(image_out)
        image_out = self.conv3d_2(image_out)
        image_out = self.maxpool_2(image_out)
        image_out = self.batchnorm_2(image_out)
        print("kek")
        # call the vessel layer
        print(vessel_text)
        text_out = self.embedding(vessel_text)
        # concatenate into a single array
        conjoined = tf.concat([image_out, text_out, other_feats])
        flattened = self.flatten(conjoined)
        # pass thru feedforward
        lin_out = self.dense1(flattened)
        lin_out = self.dense2(lin_out)
        probs = self.dense3(lin_out)

        return probs

    def loss(self, logits, labels):
        pass

    def accuracy(self, logits, labels):
        pass

def train(model, train_inputs, train_labels):

    model.call(train_inputs[0:5])
    print("Done!")

    pass

def test(model, test_inputs, test_labels):
    pass

def main():

    num_tici_scores = len(TICI_MAP)
    max_passes = 10 # last class represents >=10 passes
    num_epochs = 1

    train_data, train_labels, test_data, test_labels, vocab_size = append_csv_features(get_mips_data())

    tici_model = Model(vocab_size, num_tici_scores)
    passes_model = Model(vocab_size, max_passes)

    for i in range(num_epochs):
        train(tici_model, train_data, train_labels)
        train(passes_model, train_data, train_labels)

    test(tici_model, test_data, test_labels)
    test(passes_model, test_data, test_labels)


if __name__ == '__main__':
    main()