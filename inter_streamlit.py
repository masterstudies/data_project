from gensim.models import Word2Vec
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re 
from collections import defaultdict
import pandas as pd
import folium
from streamlit_folium import folium_static



#mean vectorizer
class MeanEmbeddingVectorizer(object):
    def __init__(self, model_cbow):
        self.model_cbow = model_cbow
        self.vector_size = model_cbow.wv.vector_size

    def fit(self):  
        return self

    def transform(self, docs): 
        doc_word_vector = self.doc_average_list(docs)
        return doc_word_vector

    def doc_average(self, doc):
        mean = []
        for word in doc:
            if word in self.model_cbow.wv.index_to_key:
                mean.append(self.model_cbow.wv.get_vector(word))

        if not mean: 
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def doc_average_list(self, docs):
        return np.vstack([self.doc_average(doc) for doc in docs])

#tf-idf vectorized
class TfidfEmbeddingVectorizer(object):
    def __init__(self, model_cbow):

        self.model_cbow = model_cbow
        self.word_idf_weight = None
        self.vector_size = model_cbow.wv.vector_size

    def fit(self, docs): 


        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        tfidf = TfidfVectorizer()
        tfidf.fit(text_docs)  
        # if a word was never seen it is given idf of the max of known idf value
        max_idf = max(tfidf.idf_)  
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        return self

    def transform(self, docs): 
        doc_word_vector = self.doc_average_list(docs)
        return doc_word_vector

    def doc_average(self, doc):


        mean = []
        for word in doc:
            if word in self.model_cbow.wv.index_to_key:
                mean.append(
                    self.model_cbow.wv.get_vector(word) * self.word_idf_weight[word]
                ) 

        if not mean:  
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean
    def doc_average_list(self, docs):
      return np.vstack([self.doc_average(doc) for doc in docs])

@st.cache(allow_output_mutation=True)
def load_data(check): 
    if check: 
        data = pd.read_excel('final_data.xlsx')
        embeddings = pd.read_pickle('embed.pickle')
        clean_words = pd.read_pickle('words.pickle')
        swords = pd.read_pickle('swords.pickle')
    return data, embeddings, clean_words, swords

# @st.cache(allow_output_mutation=True)
# def corpus_l(data):
#     return list(data)

@st.cache(allow_output_mutation=True)
def load_model(mpath): 
    return Word2Vec.load(mpath)



#load up some cleaning functions
def tokenization(text):
    tokens = re.split('\s+',text)
    return tokens

def remove_stopwords(text):
    output= [i for i in text if i not in swords]
    return output

def len_control(text):
  lemm_text = [word for word in text if len(word)>=3]
  return lemm_text

def sorter(text):
  sorted_list = sorted(text)
  return sorted_list


def program_parser2(data):
    for i in range(data.shape[0]):
        data.Introduction[i] = str(re.sub('[0-9]+',' ',re.sub(r'[^\w\s]',' ',re.sub('\\\\n', ' ' ,re.sub('&.*?;.*?;|&.*?;|._....',' ',str(data.Introduction[i]))))).lower().strip())
    data['msg_sorted_clean']= (data['Introduction']
                               .apply(lambda x: tokenization(x))
                               .apply(lambda x:remove_stopwords(x))
                               .apply(lambda x:len_control(x))
                               .apply(lambda x: sorter(x)))
    return data

def get_recommendations(N, scores, data_path = 'clean_text.xlsx'):
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    data = (pd.read_excel(data_path, index_col = 0)
           .drop(columns = ['msg_sorted_clean', 'Tuition'])
           .loc[top]
           .reset_index())
    return data

def get_recs(sentence, N=5, mean=False):
    '''Get top-N recommendations based on your input'''
    input = pd.DataFrame({'Introduction': [str(sentence)]})
    input = program_parser2(input)
    input_embedding = tfidf_vec_tr.transform([input['msg_sorted_clean'][0]])[0].reshape(1, -1)
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    recommendations = get_recommendations(N,scores)
    return recommendations


with st.sidebar:
    col1, col2 =st.columns([2.2,6])
    with col1:
        st.write("")
    with col2:
        st.write("")  #st.image('logo.png')
    page = st.radio('Страница', ['Приветствие','Поиск программ','Интересная статистика','Карта'])


# Page 1-Intro
if page=='Приветствие':
  #  st.markdown(dash, unsafe_allow_html = True)
    st.subheader('Приветствие')
   # st.markdown(hello, unsafe_allow_html = True)

if page=='Поиск программ':
    st.title("University recommender app")
    st.write(
    "A simple grad school recommender")
    form = st.form(key="my_form")
    sentence = form.text_input(label="Enter the text to indicate ur university preferences")
    submit = form.form_submit_button(label="Find Universities")
    data, doc_vec, clean_words, swords = load_data(True)
    corpus = list(clean_words)
    model = load_model('model_cbow.bin')
    model.init_sims(replace=True)
    tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
    tfidf_vec_tr.fit(corpus)
    if submit:
    # make prediction from the input text
        recs = get_recs(sentence)
 
    # Display results of the NLP task
        st.dataframe(recs)
if page == 'Интересная статистика':
    st.title('Здесь должна быть описательная статистика')
if page == 'Карта':
    my_map = folium.Map()
    folium.Marker(
    location=[51.5074,0.1278]
    ).add_to(my_map)
    folium_static(my_map)
    