#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk
from nltk import FreqDist 
nltk.download('stopwords') # run this one time


# In[7]:


import pandas as pd 
pd.set_option("display.max_colwidth", 200) 
import numpy as np 
import re 
import spacy 
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim 
from gensim import corpora 


# In[8]:


# libraries for visualization 
import pyLDAvis 
import pyLDAvis.gensim 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import sklearn


# In[10]:


import gzip
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)
 
def getDF(path): 
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# In[11]:


df = getDF('reviews_Automotive_5.json.gz')


# In[12]:


##getting a peek of the data
df.head()


# In[13]:


## data processing- lets define a function that would plot a bar grpah of 'n' most frequent words in the data
# function to plot most frequent terms

def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    
    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()),
                'count':list(fdist.values())})
    
    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count",n = terms)
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel = 'count')
    plt.show()


# In[14]:


freq_words(df['reviewText'])


# In[15]:


#remove unwanted characters, numbers and symbols
df['reviewText'] = df['reviewText'].str.replace("[^a-zA-Z#]", " ")


# In[16]:


from nltk.corpus import stopwords 
stop_words = stopwords.words('english')
# function to remove stopwords 
def remove_stopwords(rev):     
  rev_new = " ".join([i for i in rev if i not in stop_words])      
  return rev_new 
# remove short words (length < 3) 
df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join([w for 
                   w in x.split() if len(w)>2])) 
# remove stopwords from the text 
reviews = [remove_stopwords(r.split()) for r in df['reviewText']] 
# make entire text lowercase 
reviews = [r.lower() for r in reviews]


# In[17]:


freq_words(reviews, 35)


# In[21]:


#one time run
get_ipython().system('python -m spacy download en')


# In[32]:


nlp = spacy.load('en', disable=['parser', 'ner']) 

def lemmatization(texts, tags=['NOUN', 'ADJ']): 
       output = []        
       for sent in texts:              
             doc = nlp(" ".join(sent))                             
             output.append([token.lemma_ for token in doc if 
             token.pos_ in tags])        
       return output


# In[33]:


# tokenize the reviews and then lemmatize them
tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
print(tokenized_reviews[1])


# In[34]:


reviews_2 = lemmatization(tokenized_reviews)
print(reviews_2[1]) # print lemmatized review


# In[35]:


##de-tokenize the lemmatized reviews and plot the most common words.
reviews_3 = []
for i in range(len(reviews_2)):
  reviews_3.append(' '.join(reviews_2[i]))

df['reviews'] = reviews_3

freq_words(df['reviews'], 35)


# In[36]:


## BUILDING THE LDA MODEL
dictionary = corpora.Dictionary(reviews_2)


# In[ ]:


doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
# Creating the object for LDA model using gensim library 
LDA = gensim.models.ldamodel.LdaModel 
# Build LDA model 
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary,                                     
                num_topics=5, random_state=100, chunksize=1000,                                     
                passes=50)


# In[40]:


lda_model.print_topics()


# In[41]:


# Visualize the topics 
pyLDAvis.enable_notebook() 
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix,  
                              dictionary) 
vis

