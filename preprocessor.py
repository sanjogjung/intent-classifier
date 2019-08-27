import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class Preprocess:
    def process_data(self, df):
        sentences = df['sentence']
        self.cleaned_text = sentences.apply(self.clean_data)
        features = self.features_selection()
        return features

    def clean_data(self, sentence):
        remove_punctuation = [char for char in sentence if char not in string.punctuation]
        single_string = ''.join(remove_punctuation)
        tokens = single_string.split()
        lower_tokens = [token.lower() for token in tokens]
        cleaned_sentence = ' '.join(lower_tokens)
        return cleaned_sentence

    def features_selection(self):
        tvf = TfidfVectorizer(ngram_range=(1, 2))
        features = tvf.fit_transform(self.cleaned_text)
        pickle.dump(tvf, open('vector.pkl', "wb"))
        return features










