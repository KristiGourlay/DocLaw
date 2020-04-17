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

from models.language_cleaner import Datacleaner

#Load data
data = pd.read_csv('data/final_legal_df.csv', index_col=0)

#Load models
lda_model = pickle.load(open('models/lda_final_model', 'rb'))
d2v_model = Doc2Vec.load('models/d2v_final_model')
cvec = pickle.load(open('models/final_cvec_model', 'rb'))



class Law_Docs():


    def __init__(self, query):
        dc = Datacleaner()
        self.query = dc.clean(query)

    def show_results(self, predictions):

        result = []
        n = 0
        for item in predictions:
            n += 1
            # print(data.summs[i])
            result.append(f"Most Similar Case # {n}: Case: {data.case_citation_name[item]}")
            result.append(f"Case Summary: {data.summs[item]}")


        return result


    def ldamodel(self):

        '''
        returns prediction of which topic label input belongs to
        '''

        count_query = cvec.transform([self.query])
        topic_likelihood = lda_model.transform(count_query)[0]
        topic = sorted(enumerate(topic_likelihood, 1), key=lambda x: x[1], reverse=True)[0][0]

        return topic

    def d2v_preds(self):

        '''
        returns most similar legal cases based on context
        '''

        inferred_vector = d2v_model.infer_vector(self.query)
        sim_vectors = d2v_model.docvecs.most_similar([inferred_vector], topn=100)
        preds = [n[0] for n in sim_vectors]

        return preds


    def super_version(self):

        '''
        finds all similar documents within the same topic cluster
        '''

        sim_docs = self.d2v_preds()

        topic = self.ldamodel()

        sim_docs_same_topic = [num for num in sim_docs if data['lda_preds'][num] == topic]

        return self.show_results(sim_docs_same_topic)
        # return sim_docs_same_topic

    def apply_filters(self, date=1841, court='no_court'):

        '''
        input: allows you to add date and court restrictions (Illinois Appellate Court, Illinois Supreme Court,	Illinois Circuit Court, or Illinois Court of Claims)
        '''

        sim_docs = self.d2v_preds()

        topic = self.ldamodel()

        return_docs = [num for num in sim_docs if data['lda_preds'][num] == topic]

        if date != 1841 and court != 'no_court':
            return self.show_results([num for num in return_docs if data['decision_date'][num] >= date and data['court_name'][num] == court])
        elif date > 1841 and court == 'no_court':
            return self.show_results([num for num in return_docs if data['decision_date'][num] >= date])
        elif date == 1841 and court != 'no_court':
            return self.show_results([num for num in return_docs if data['court_name'][num] == court])
        else:
            return self.show_results(return_docs)



# user_query = Law_Docs('tax realestate mortgage')
# user_query.ldamodel()
# user_query.d2v_preds()
# user_query.super_version()
# user_query.apply_filters(date=1960)



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


        ct = Datacleaner()
        clean_query = ct.clean(query)
        user_query = Law_Docs(clean_query)

        prediction = user_query.apply_filters(date=date)[:10]


        return render_template('results.html', results=prediction)



if __name__ == '__main__':
    '''Connects to the server'''


    HOST = '127.0.0.1'
    PORT = 5000
    debug = True

    app.run(HOST, PORT)
