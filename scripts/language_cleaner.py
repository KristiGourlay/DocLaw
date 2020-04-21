import pandas as pd
import numpy as np
import regex as re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.base import TransformerMixin


class Datacleaner(TransformerMixin):

    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    def fit(self, raw_text):
        return self


    def clean(self, raw_text):

        clean_text = BeautifulSoup(raw_text, features='lxml').get_text()
        lower_cases = clean_text.lower()
        tokens = Datacleaner.tokenizer.tokenize(lower_cases)
        lemms = [Datacleaner.lemmatizer.lemmatize(word) for word in tokens]
        words = [word for word in lemms if word not in stopwords.words('english')]
        final_words = ' '.join(words)

        return final_words


    def clean_tokens(self, raw_text):

        words = self.clean(raw_text)

        return Datacleaner.tokenizer.tokenize(words)


    def clean_col(self, col):

        clean_list = []
        n = 0
        for item in col:
            n += 1
            if n % 25 == 0:
                print(n)
            clean_list.append( self.clean(item) )

        return clean_list



ct = Datacleaner()
ct.clean('The dog was running around the garden')
ct.clean_tokens('The dog was running around the garden')
