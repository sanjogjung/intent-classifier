## Intent Detector
#### The main motto of intent detector is to detect an intention of a person.

##### The program into divided into different modules

- trainer.py -for training the model(Naive Bayes)
- preprocessor.py - for preprocessing the text
- test.py - for testing our model with training dataset
- predict.py - for predicting the intent using terminal
- api.py - for predicting using postman
- jaccard_similarity.py - for testing with test data using jaccard similarity score
- cosine_similarity.py - for testing with test data using cosine similarity score
- predict_cosine_similarity - for predicting intent with cosine similarity score

### Requirements :
- Sklearn
    - TFIDF vectorizer
    - Multinomial Naive Bayes classifier
 - Pandas
 - Numpy
 - Pickle
 - Flask (for api )
    -request
 - sys
 - string


### How to run the program :
- for training the naive bayes model we invoke the terminal with following commmand
l
```
python trainer.py train.csv
```
train.csv is the data on which you want to train the mode

- for predicting using terminal :
````
python predict.py bring me some water
````
- for predicting using postman:

```
python api.py
```
then goto postman and make a post request and insert the sentence which intention you want to predict in JSON format.

> {"sentence": "bring me some water "}
Then our program will return us a JSON object as a response

>{"intent": "order"}

- for making prediction using cosine_similarity metrics

```
python cosine_similarity.py hello how are you
```
- we can test our model with different models 
```
python cosine_simalrity.py
```
```
python jaccard_similarity.py
```
```
python test.py test.csv
```
These 3 commands will test our models and give us some metrics like accuracy and confusion matrix


 
