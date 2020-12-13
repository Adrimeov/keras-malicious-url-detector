import math
import keras
import numpy as np
from sklearn.model_selection import train_test_split


class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=100, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.X = X
        self.Y = Y
        self.shuffle = shuffle
        self.indexes = [i for i in range(len(self.X))]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        idxs = [i for i in range(index * self.batch_size, (index + 1) * self.batch_size)]

        # Find list of IDs
        # list_IDs_temp = [self.indexes[k] for k in idxs]

        # Generate data
        X_ = self.X[idxs]
        Y_ = self.Y[idxs]

        return X_, Y_

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == "__main__":
    X = np.load("features.npy")
    Y = np.load("labels.npy")
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=.2)
    generator = DataGenerator(Xtrain, Ytrain, 5)
    print(generator[2][0].shape)