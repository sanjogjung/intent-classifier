import numpy as np
import pickle
import preprocessor
import pandas as pd
import sys
from scipy import spatial


class Predict:
    def __init__(self, text):
        """
        taking the user input string
        loading trained feature numpy array
        loading the output for the numpy array
        loading the vectorizer saved during training

        :param text:
        """
        self.text = text
        self.train_vec = np.load('feat.npy')
        self.train_output = pickle.load(open('mylist.pkl', 'rb'))
        self.vec = pickle.load(open('vector.pkl', 'rb'))

    def process_text(self):
        """
        creating an instance of Preprocess class
        applying clean_data function on the text
        transforming the text to tfidf array

        """
        prp1 = preprocessor.Preprocess()
        processed_text = prp1.clean_data(self.text)
        self.vec1 = self.vec.transform(pd.Series(processed_text))

    def compute_cosine_similarity(self):
        """
        creating an empty list for storing the cosine values
        multiplying the input vector with every row of training vector
        appending the cosine value to the list
        taking the index of maximum value of the list
        using the index to find the attribute from the output_vector

        """
        cos_matrix = []
        for i in range(len(self.train_vec)):
            val = self.vec1 * self.train_vec[i]
            cos_matrix.append(val[0])
        out = np.argmax(cos_matrix)
        print(self.train_output[out])


if __name__ == '__main__':
    text = sys.argv[1:]
    text = ' '.join(text)
    p1 = Predict(text)
    p1.process_text()
    p1.compute_cosine_similarity()










