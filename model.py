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
        self.batch_size = 16
        self.learning_rate = 0.002
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.num_classes = num_classes

        self.embedding_size = 64
        self.dropout_rate = 0.2
        self.kernel_size = (3, 9, 9) # TODO Check order of dimensions, which one is depth
        self.strides = (1, 2, 2)
        self.pool_size = (4, 4, 4)
        self.num_filters_1 = 4
        self.num_filters_2 = 8

        self.dense1_size = 128
        self.dense2_size = 128

        # model for image
        self.conv3d_1 = Conv3D(self.num_filters_1, kernel_size=self.kernel_size, strides=self.strides, activation='relu',
                         input_shape=self.sample_shape)
        self.maxpool_1 = MaxPooling3D(pool_size=self.pool_size, padding='same')
        self.batchnorm_1 = BatchNormalization(center=True, scale=True)
        self.dropout_1 = Dropout(self.dropout_rate)

        self.conv3d_2 = Conv3D(self.num_filters_2, kernel_size=self.kernel_size, strides=self.strides, activation='relu')
        self.maxpool_2 = MaxPooling3D(pool_size=self.pool_size, padding='same')
        self.batchnorm_2 = BatchNormalization(center=True, scale=True)
        self.dropout_2 = Dropout(self.dropout_rate)

        
        # model for vessel and location
        self.embedding = Embedding(self.vocab_size, self.embedding_size)
        # feedforward layer combined with other features (e.g. age, gender)
        self.dense1 = Dense(self.dense1_size, activation='relu')
        self.dense2 = Dense(self.dense2_size, activation='relu')
        self.dense3 = Dense(self.num_classes, activation='softmax') # convert to probs in last layer
        # #todo: remove sm if we want to get rid of classification of passes


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of data.
        :param inputs: numpy array of data of length (batch_size)
        :return: logits - a matrix of shape (batch_size, num_classes)
        """

        # images = inputs[:,0]
        images = tf.convert_to_tensor([row[0] for row in inputs])
        vessel_text = tf.convert_to_tensor([row[1] for row in inputs])
        other_feats = np.array([row[2:] for row in inputs])
        # call image layer
        #print("starting conv")
        image_out = self.conv3d_1(images)
        image_out = self.batchnorm_1(image_out)
        image_out = self.maxpool_1(image_out)
        #print("finishing conv")
        image_out = self.dropout_1(image_out)
        image_out = self.conv3d_2(image_out)
        image_out = self.batchnorm_2(image_out)
        image_out = self.maxpool_2(image_out)
        #print(image_out.shape)
        #image_out = #(batch_size, 64, 32, 32, 1)


        # print(vessel_text)
        # call the vessel layer
        #print("start embedding")
        text_out = self.embedding(vessel_text)
        #print("finish embedding")
        # (batch_size, sentence_size)
        # concatenate into a single array

        # flatten image output
        # (batch_size, x)

        image_out = tf.reshape(image_out, [len(inputs), -1])
        text_out = tf.reshape(text_out, [len(inputs), -1])

        conjoined = tf.concat([image_out, text_out, other_feats], axis=1)

        # flattened = self.flatten(conjoined)
        # pass thru feedforward
        #print("start feedforward")
        lin_out = self.dense1(conjoined)
        lin_out = self.dense2(lin_out)
        probs = self.dense3(lin_out)
        #print("finish call")

        return probs

    def loss(self, probs, labels):
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,probs))
        return loss

    def accuracy(self, probs, labels):
        # correct_predictions = tf.equal(tf.argmax(probs, 1), labels)
        # return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        num_correct = 0
        for i in range(len(labels)):
            if np.argmax(probs[i]) == labels[i]:
                num_correct += 1
        return num_correct / len(labels)

    def multi_accuracy(self, probs, labels, num):
        # correct_predictions = tf.equal(tf.argmax(probs, 1), labels)
        # return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        num_correct = 0
        for i in range(len(labels)):
            if np.argmax(probs[i]) == labels[i]:
                num_correct += 1
        return num_correct / len(labels)

def train(model, train_inputs, train_labels, predict_tici=False):
    nrows = train_labels.shape[0]
    nbatches = int(np.ceil(nrows/model.batch_size))
    accuracy = np.zeros(nbatches)
    if predict_tici:
        train_labels = np.array([row[1] for row in train_labels])
    else:
        train_labels = np.array([row[0] for row in train_labels])

    for batch in range(0,nbatches):
        # For every batch, compute then descend the gradients for the model's weights
        start_idx = batch * model.batch_size
        inputs = train_inputs[start_idx:min((start_idx + model.batch_size), nrows)]
        labels = train_labels[start_idx:min((start_idx + model.batch_size), nrows)]
        with tf.GradientTape() as tape:
            probabilities = model.call(inputs)
            loss = model.loss(probabilities, labels)
            accuracy[batch] = model.accuracy(probabilities, labels)
            # model.loss_list.append(loss.numpy())
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print("Done! Train accuracy: " + str(tf.reduce_mean(accuracy).numpy()))

def test(model, test_inputs, test_labels, predict_tici=False):
    nrows = test_labels.shape[0]
    nbatches = int(np.ceil(nrows / model.batch_size))
    accuracy = np.zeros(nbatches)
    if predict_tici:
        test_labels = np.array([row[1] for row in test_labels])
    else:
        test_labels = np.array([row[0] for row in test_labels])
    for batch in range(0, nbatches):
        start_idx = batch * model.batch_size
        inputs = test_inputs[start_idx:min((start_idx + model.batch_size), nrows)]
        labels = test_labels[start_idx:min((start_idx + model.batch_size), nrows)]
        probs = model.call(inputs)
        accuracy[batch] = model.accuracy(probs, labels)
    return tf.reduce_mean(accuracy)


def main():
    num_tici_scores = len(TICI_MAP)
    max_passes = 10 # last class represents >=10 passes
    num_epochs = 3

    train_data, train_labels, test_data, test_labels, vocab_size = append_csv_features(get_mips_data())

    tici_model = Model(vocab_size, num_tici_scores)
    passes_model = Model(vocab_size, max_passes)

    for i in range(num_epochs):
        print(i)
        train(tici_model, train_data, train_labels, predict_tici=True)
        train(passes_model, train_data, train_labels)
        # print("train acc:")
        # print("tici:" + str(test(tici_model, train_data, train_labels, predict_tici=True).numpy()))
        # print("passes:" + str(test(passes_model, train_data, train_labels).numpy()))

    print("TICI Score Test Accuracy:" + str(test(tici_model, test_data, test_labels, predict_tici=True).numpy()))
    print("Num Passes Test Accuracy:" + str(test(passes_model, test_data, test_labels).numpy()))


if __name__ == '__main__':
    main()