#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import spacy

from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')


# In[121]:


df=pd.read_csv("C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr/Elon_musk.csv",encoding= 'unicode_escape')
df.drop(['Unnamed: 0'],inplace=True,axis=1)
df


# In[122]:


df.isnull().sum()


# In[123]:


list(df)


# In[124]:


df['Text'].value_counts()


# In[125]:


#preprocessing
df['Text']=df.Text.map(lambda x : x.lower())
df['Text']


# In[126]:


df_text=''.join(df)#joining the list in to one string/text
df_text


# In[127]:


from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
tweets_tokens=tknzr.tokenize(df_text)
print(tweets_tokens)


# In[128]:


# Again Joining the list into one string/text
tweets_tokens_text=' '.join(tweets_tokens)
tweets_tokens_text


# In[129]:


# Remove Punctuations 
no_punc_text=tweets_tokens_text.translate(str.maketrans('','',string.punctuation))
no_punc_text


# In[130]:


# remove https or url within text
import re
no_url_text=re.sub(r'http\S+', '', no_punc_text)
no_url_text


# In[131]:


from nltk.tokenize import word_tokenize
text_tokens=word_tokenize(no_url_text)
print(text_tokens)


# In[132]:


# Tokenization
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[133]:


# Tokens count
len(text_tokens)


# In[134]:


# Remove Stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')

sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# In[135]:


# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])


# In[136]:


# Stemming (Optional)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])


# In[137]:


# Lemmatization
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)


# In[138]:


lemmas=[token.lemma_ for token in doc]
print(lemmas)


# In[139]:


clean_tweets=' '.join(lemmas)
clean_tweets


# In[140]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)
print(cv.vocabulary_)


# In[141]:


print(cv.get_feature_names()[100:200])


# In[142]:


print(tweetscv.toarray()[100:200])


# In[143]:


print(tweetscv.toarray().shape)


# In[144]:


2. ###CountVectorizer with N-grams (Bigrams & Trigrams)
cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)
print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# In[145]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud

STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_tweets)
plot_cloud(wordcloud)


# In[146]:


#Named Entity Recognition (NER)
# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_sm')

one_block=clean_tweets
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[147]:


for token in doc_block[100:200]:
    print(token,token.pos_)


# In[148]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# In[149]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# In[150]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');


# In[151]:


#Emotion Mining - Sentiment Analysis
from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(df['Text']))
sentences


# In[152]:


sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df


# In[153]:


afin=pd.read_csv("C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr/afinn.csv",sep=',',encoding='Latin-1')
afin


# In[154]:


affinity_scores=afin.set_index('word')['value'].to_dict()
affinity_scores


# In[155]:


# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[156]:


# manual testing
calculate_sentiment(text='great')


# In[158]:


# how many words are there in a sentence?
sent_df['word_count']=sent_df['sentence'].str.split().apply(len)
sent_df['word_count']


# In[159]:


sent_df.sort_values(by='word_count')


# In[163]:


# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['word_count'])


# In[164]:


# Calculating sentiment value for each sentence
sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']


# In[165]:


sent_df.sort_values(by='sentiment_value')


# In[166]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[167]:


# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]


# In[168]:


# Adding index cloumn
sent_df['index']=range(0,len(sent_df))
sent_df


# In[169]:


# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])


# In[ ]:




