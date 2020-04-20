import pandas as pd
import numpy as np
import regex as re
import pickle
import flask
from flask import render_template, request
import json

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pyLDAvis.sklearn
import pyLDAvis

import gensim
from gensim.summarization import keywords, summarize, mz_keywords
from gensim.models import doc2vec, Doc2Vec

from language_cleaner import Datacleaner
from prediction_model import Law_Docs

#Load data
data = pd.read_csv('../data/final_legal_df.csv', index_col=0)

#Load models
lda_model = pickle.load(open('../models/lda_final_model', 'rb'))
d2v_model = Doc2Vec.load('../models/d2v_final_model')
cvec = pickle.load(open('../models/final_cvec_model', 'rb'))



app = flask.Flask(__name__)

@app.route('/home')
def home():
   with open("templates/home.html", 'r') as home:
       return home.read()


@app.route('/results', methods=['POST', 'GET'])
def results():
    if flask.request.method == 'POST':

        inputs = flask.request.form

        query = inputs['text']
        date = int(inputs['date'])

        user_query = Law_Docs(query)
        prediction = user_query.super_version()
        prediction = user_query.apply_filters(date=date)[:20]


        return render_template('results.html', results=prediction)



if __name__ == '__main__':
    '''Connects to the server'''


    HOST = '127.0.0.1'
    PORT = 5000
    debug = True

    app.run(HOST, PORT)
