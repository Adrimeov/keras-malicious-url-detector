import numpy as np
import pandas as pd

from keras_malicious_url_detector.library.utility.url_data_loader import load_url_data
from keras_malicious_url_detector.library.utility.text_model_extractor import extract_text_model


def extract_training_data(text_model, data):
    data_size = data.shape[0]
    X = np.zeros(shape=(data_size, text_model['max_url_seq_length']), dtype="uint8")
    Y = np.zeros(shape=(data_size, 2), dtype="uint8")
    for i in range(data_size):
        url = data['text'][i]
        label = data['label'][i]
        for idx, c in enumerate(url):
            X[i, idx] = text_model['char2idx'][c]
        Y[i, label] = 1

    return X, Y


if __name__ == "__main__":

    # data_dir = "data/train_augmented.csv"
    #
    # url_data = load_url_data(data_dir)
    # text_model = extract_text_model(url_data['text'])
    #
    # X, Y = extract_training_data(text_model, url_data)

    # np.save("data/train_augmented_encoded.npy", X)
    # np.save("data/train_augmented_encoded_labels.npy", Y)
    print(np.load("data/train_augmented_encoded_text_model.npy", allow_pickle=True).item())

