import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    """
    reading the dataset
    splitting data set into training and testing dataframe
    saving both dataframe in csv format

    """
    df = pd.read_csv('data.csv')
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)


if __name__ == '__main__':
    main()



