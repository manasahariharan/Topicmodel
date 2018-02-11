# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 09:19:57 2017

@author: manasa
"""
#%%
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import pandas as pd
from gensim.models import ldamodel
from gensim import corpora
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import re

def clean(doc):
    punc_free = re.sub("[^a-zA-Z]"," ", doc)
    stop_free = " ".join([i for i in punc_free.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
    return normalized

#%%
data = pd.read_csv('./stocknews/RedditNews.csv')
if __name__ == '__main__': 
    news = list(data['News'])
    dates = data.Date.unique()
    datacomplete = []
    for i in range(0,len(dates)):
        datacomplete.append(' '.join(str(x) for x in news[i*25:(i*25 + 24)]))

#%%
    stop = set(stopwords.words('english')+ ['ba','bthe','b',"r", "n", "amp", "girl"]) 
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    
    data_clean = [clean(doc).split() for doc in datacomplete]   
    
#%%
    dictionary = corpora.Dictionary(data_clean)
    dictionary.filter_extremes(no_below = 100, no_above = 0.3)
    d_mat = [dictionary.doc2bow(doc) for doc in data_clean]
#%%
    Lda = ldamodel.LdaModel
    ldamod = Lda(d_mat, num_topics = 5, id2word = dictionary, passes = 10)
    ldamod.print_topics(10)
#%%

    plot = gensimvis.prepare(ldamod,d_mat, dictionary)
#%%
    pyLDAvis.show(plot)
#%%
    ldamod1 = Lda(d_mat, num_topics = 10, id2word = dictionary, passes = 15)
    ldamod1.print_topics(10)
#%%
    plot1 = gensimvis.prepare(ldamod1,d_mat, dictionary)
    pyLDAvis.show(plot1)
#%%
    ldamod2 = Lda(d_mat, num_topics = 7, id2word = dictionary, passes = 15)
    ldamod2.print_topics(10)
#%%
    plot2 = gensimvis.prepare(ldamod2,d_mat, dictionary)
    pyLDAvis.show(plot2)
#%%
    ldamod3= Lda(d_mat, num_topics = 7, id2word = dictionary, passes = 30)
    ldamod3.print_topics(15)
#%%
    plot3 = gensimvis.prepare(ldamod3,d_mat, dictionary)
    pyLDAvis.show(plot3)