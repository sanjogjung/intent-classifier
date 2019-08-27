from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import sys
import preprocessor
import pickle
import numpy as np


class Train:
    def __init__(self, x_train, y_train):
        """
        initializing training data
        :param x_train:
        :param y_train:
        """
        self.x_train = x_train
        self.y_train = y_train
        self.clf1 = MultinomialNB()

    def train_model(self):
        """
        training the model
        saving the model using pickle

        """
        self.clf1.fit(self.x_train, self.y_train)
        pickle.dump(self.clf1, open('model1.pkl', 'wb'))
        print('Model trained and saved successfully')


if __name__ == "__main__":
    file_name = sys.argv[1:]
    df = pd.read_csv(file_name[0])
    prp1 = preprocessor.Preprocess()
    features = prp1.process_data(df)
    featarray = features.toarray()
    np.save('feat.npy', featarray)
    t1 = Train(features, df['intent'])
    pickle.dump(df['intent'], open('mylist.pkl', 'wb'))
    t1.train_model()



