import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import string


class Jaccard:
    def load_data(self):
        """loading all the training and test data
        applying the process_data function to the training and testing input dataset
        saving them as train and test features

        """
        df1 = pd.read_csv('train.csv')
        df2 = pd.read_csv('test.csv')
        x_train = df1['sentence']
        self.y_train = df1['intent']
        self.x_test = df2['sentence']
        self.y_test= df2['intent']
        self.train_features = x_train.apply(self.process_data)
        self.separate_data()
        self.test_features = self.x_test.apply(self.process_data)

    def process_data(self, sentence):
        """

        removing punctuation from each sentence
        joining the characters to form a sentence again
        tokenizing the sentence
        lowering the tokens

        :param sentence:
        :return: lower_tokebs
        """
        remove_punctuation = [char for char in sentence if char not in string.punctuation]
        single_string = ''.join(remove_punctuation)
        tokens = single_string.split()
        lower_tokens = [token.lower() for token in tokens]
        return lower_tokens

    def separate_data(self):
        self.greeting = []
        self.order = []
        self.goodbye = []
        for i in range(len(self.train_features)):
            if self.y_train[i] == 'greeting':
                self.greeting.append(self.train_features[i][0])
            elif self.y_train[i] == 'order':
                self.order.append(self.train_features[i][0])
            else:
                self.goodbye.append(self.train_features[i][0])
        '''print('Good bye list :')
            print(self.goodbye)
            print('Order list')
            print(self.order)
            print('Greeting list')
            print(self.greeting)'''
        self.intents = [self.goodbye,self.order,self.greeting]
        self.output_intents = ['goodbye', 'order', 'greeting']

    def predict(self):

        """
        creating an empty list for prediction
        taking prediction with y_train using max_score index as an index
        appending prediction

        :return:
        """
        self.predictions = []
        scores =[]
        for i in range(len(self.test_features)):
            for j in range(len(self.intents)):
                score = len(set(self.test_features[i]) & set(self.intents[j]))/len(set(self.test_features[i]) | set(self.intents[j]))
                scores.append(score)
                if len(scores) == len(self.intents):
                    indx = np.argmax(scores)
                    prediction = self.output_intents[indx]
                    self.predictions.append(prediction)
                    scores = []
        print(self.predictions)

    def metrics(self):
        """
        calculating accuracy score  and confusion matrix
        :return:
        """
        print('Accuracy score for jaccard similarity metrics : \n')
        print(accuracy_score(self.y_test, self.predictions))
        print('Confusion Matrix : \n')
        print(confusion_matrix(self.y_test, self.predictions))


if __name__ == '__main__':
    j1 = Jaccard()
    j1.load_data()
    j1.separate_data()
    j1.predict()
    j1.metrics()
























