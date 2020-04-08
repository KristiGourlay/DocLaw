import pandas as pd
import numpy as np
import regex as re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.base import TransformerMixin


class Datacleaner(TransformerMixin):


    def fit(self, raw_text):
        return self

    def clean(self, raw_text):

        tokenizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()

        clean_text = BeautifulSoup(raw_text).get_text()
        lower_cases = clean_text.lower()
        tokens = tokenizer.tokenize(lower_cases)
        lemms = [lemmatizer.lemmatize(word) for word in tokens]
        words = [word for word in lemms if word not in stopwords.words('english')]
        final_words = ' '.join(words)

        return final_words

    def clean_col(self, col):

        clean_list = []
        n = 0
        for item in col:
            n += 1
            if n % 25 == 0:
                print(n)
            clean_list.append( self.clean(item) )

        return clean_list
