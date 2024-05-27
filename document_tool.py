import pickle
import itertools
from math import log
import re
import collections
from typing import List

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec
from konlpy.tag import Okt
from wordcloud import WordCloud
from textrankr import TextRank

class _MyTokenizer:
    def __call__(self, text: str) -> List[str]:
        okt: Okt = Okt()
        tokens: List[str] = okt.phrases(text)
        return tokens

_mytokenizer: _MyTokenizer = _MyTokenizer()
_textrank: TextRank = TextRank(_mytokenizer)

okt = Okt()

def get_doc2vec_model(path):
    return Doc2Vec.load(path)

def get_logistic_model(path):
    with open(path, 'rb') as fr:
        return pickle.load(fr)

def get_category(document, doc2vec_model, classification_model):
    category = {
        0: 'news_r',
        1: 'briefing',
        2: 'his_cul',
        3: 'paper',
        4: 'minute',
        5: 'edit',
        6: 'public',
        7: 'speech',
        8: 'literature',
        9: 'narration'
    }
    temp = classification_model.predict([doc2vec_model.infer_vector(_tokenize(document))])
    return category[temp[0]]

def _cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def _tokenize(text: str) -> list:
    text = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9\s\(\)\{\}\[\]\.,\?!:;\'\"]', '', text)
    pos_tagged: list = okt.pos(text)
    tokenized = [word for word, tag in pos_tagged if tag not in ('Josa', 'Eomi', 'Punctuation')]

    return tokenized

def find_similar_documents(category: str, document: str, tokenized_data: list, doc2vec_model, n=10):
    rank: list = []
    
    document_vector = doc2vec_model.infer_vector(_tokenize(document))

    compare_document: list = list(filter(lambda x: x['category'] == category, tokenized_data))

    for compare in compare_document:
        
        rank.append((_cosine_similarity(document_vector, doc2vec_model.infer_vector(compare['tokenized_passage'])), compare['passage']))

    rank.sort(key=lambda x: x[0], reverse=True)

    return rank[: n]

def _make_nouns_list(document):
    '''
    This function requires Counter in collections module and chain in itertools.
    document parameter: list or str
    '''
    if type(document) == str:
        nouns_list = okt.nouns(document)

        return collections.Counter(nouns_list)
    
    else:
        nouns_list: list = []
        
        for sentence in document:
            nouns_list.append(okt.nouns(sentence))

        nouns_list = itertools.chain(*nouns_list)

        return collections.Counter(nouns_list)
        
def _TF_IDF(word: str, document, document_word):
    '''
    document parameter: list or str
    document_word: all document's word
    '''
    def TF(word: str, document):
        word_counter = _make_nouns_list(document)

        return word_counter[word] / sum(word_counter.values())
    
    def IDF(word: str, document_word):
        count = 0
        
        for doc in document_word:
            if word in doc:
                count +=1

        return log(len(document_word) / (1 + count))
    
    return TF(word, document) * IDF(word, document_word)
    
def get_TF_IDF_values(document, document_word):
    
    nouns_dict = _make_nouns_list(document)
    nouns_list: list = nouns_dict.keys()
    TF_IDF_list: list = []
    
    for word in nouns_list:
        
        nouns_dict[word] = _TF_IDF(word, document, document_word)
        
    nouns_dict = list(nouns_dict.items())
    nouns_dict.sort(key=lambda x: x[1], reverse=True)
    
    nouns_dict: dict = dict(nouns_dict)
    
    return nouns_dict

def make_wordcloud(word_frequency, filename='wordcloud'):

    wc = WordCloud(random_state = 123, font_path = 'AppleGothic', width = 400,
                   height = 400, background_color = 'white')
    
    img_wordcloud = wc.generate_from_frequencies(word_frequency)
    
    plt.figure(figsize = (10, 10)) # 크기 지정하기
    plt.axis('off') # 축 없애기
    plt.imshow(img_wordcloud) # 결과 보여주기
    plt.savefig(filename) # 파일 저장

def summarize(text, k=3):
    return _textrank.summarize(text, k, verbose=False)
