from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
from preprocessor import Preprocess
import pandas as pd
import sys


class Test:
    def __init__(self, x_test, y_test):
        """
        initializing x_test and y_test
        :param x_test:
        :param y_test:

        """
        self.x_test = x_test
        self.y_test = y_test

    def process_text(self):

        """
        creating an instance of Preprocess class of preprocessor module
        applying the clean_data function to the x_test
        loading the vectorizer saved during training
        calling prediction function for making prediction

        """
        prpobj = Preprocess()
        refined_text = self.x_test.apply(prpobj.clean_data)
        tf1_old = pickle.load(open('vector.pkl', 'rb'))
        self.features = tf1_old.transform(refined_text)
        print(self.features)
        self.prediction()

    def metrics(self):
        """
        printing the different metrics for models
        confusion metrics,accuracy, classification report, jaccard simalarity

        """
        print('Accuracy achieved  for Multi nomial Naive Bayes :\n')
        print(accuracy_score(self.y_test, self.predictions))
        print('Confusion Matrix : \n')
        print(confusion_matrix(self.y_test, self.predictions))
        print('Classification report :\n')
        print(classification_report(self.y_test, self.predictions))


    def prediction(self):
        """
        for checking the accuracy we test our model with test data
        calling metrics function for metrics calculation

        """
        model = pickle.load(open('model1.pkl', 'rb'))
        self.predictions = model.predict(self.features)
        self.metrics()


if __name__ == '__main__':
    print(sys.argv)
    filename = sys.argv[1:]
    df = pd.read_csv(filename[0])
    obj1 = Test(df['sentence'], df['intent'])
    obj1.process_text()




