import numpy as np
import pickle
import preprocessor
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class Predict:
    def __init__(self, x_test, y_test):
        """
        initializing x_test and y test
        loading trained feature numpy array
        loading the output for the numpy array
        loading the vectorizer saved during training

        :param x_test, y_test:
        """
        self.x_test = x_test
        self.y_test = y_test
        self.train_vec = np.load('feat.npy')
        self.train_output = pickle.load(open('mylist.pkl', 'rb'))
        self.vec = pickle.load(open('vector.pkl', 'rb'))

    def process_text(self):
        """
        creating an instance of Preprocess class
        applying clean_data function on the text
        transforming the text to tfidf array
        converting the sparse matrix into numpy array

        """
        prp1 = preprocessor.Preprocess()
        processed_text = self.x_test.apply(prp1.clean_data)
        self.vec1 = self.vec.transform(processed_text)
        self.numpyvec = self.vec1.toarray()

    def compute_cosine_similarity(self):
        """
        creating an empty list for storing the cosine values indexes and cosine values
        multiplying the input vector with every row of training vector
        appending the cosine value to the list
        taking the index of maximum value of the list when cosine values list is equal to length of train vector
        emptying the cosine value list
        and iterating for every test_vector

        """
        self.cos_values_index= []
        cos_val = []
        for i in range(len(self.train_vec)):
            for j in range(len(self.numpyvec)):
                val = self.numpyvec[j] * self.train_vec[i]
                cos_val.append(val[0])
                if len(cos_val) == len(self.train_vec):
                    self.cos_values_index.append(np.argmax(cos_val))
                    cos_val = []

    def prediction(self):
        """
        creating an empty prediction list
        iterating through the indexes
        and taking the output using that index with output vector
        appending prediction the predictions list

        :return:
        """
        self.predictions = []
        for i in range(len(self.cos_values_index)):
            output = self.train_output[self.cos_values_index[i]]
            self.predictions.append(output)

    def metrics(self):
        """
        computing accuracy score and confusion matrix
        :return:
        """
        print('Accuracy score for cosine similarity metrics : \n')
        print(accuracy_score(self.y_test, self.predictions))
        print('Confusion Matrix : \n')
        print(confusion_matrix(self.y_test, self.predictions))


if __name__ == '__main__':
    print(sys.argv)
    filename = sys.argv[1:]
    df = pd.read_csv(filename[0])
    obj1 = Predict(df['sentence'], df['intent'])
    obj1.process_text()
    obj1.compute_cosine_similarity()
    obj1.prediction()
    obj1.metrics()
