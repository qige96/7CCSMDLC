'''
Implementation of some algorithms in Data Mining,
a suppliment of KCL 7CCSMDM1 lecture notes

:author: Ricky Zhu
:email:  rickyzhu@foxmail.com
:license: MIT
'''
from __future__ import print_function
import numpy as np
import pandas as pd


def bc_distance():
    '''
    compute between cluster distance for clustering tasks
    '''
    pass

def wc_distance():
    '''
    compute within cluster distance for clustering tasks
    '''
    pass

def category_utility_metric():
    '''
    compute category utility metric for intremental clustering tree 
    '''
    pass

def gini(branch):
    '''
    compute Gini coeffitient of a branch of a tree split
    
    Parameters
    ----------
    branch: list
        number of all instances for each class
        
    Returns
    -------
    gini_coef: float
        Gini coeffitient
    
    '''
    pass

def entropy():
    '''
    compute entropy of a branch of a tree split
    '''
    pass
    
def info_gain():
    '''
    compute information gain of a branch of a tree split
    '''
    pass

def metrics_from_confusion_matrix(mat, class_index):
    '''
    compute precision, recall, f_value from confusion matrix.
                         predicted
              -----------------------
              |    |  a  |  b  |  c |
              -----------------------
              | a  | 88  |  10 |  2 | 
       actual | b  | 14  |  40 |  6 |
              | c  | 18  |  10 | 12 |
              ----------------------- 
        For class a, precision = 0.73, recall = 0.88, f_value = 0.4
        Overall, success rate = 0.7
        [from: lec 3 slide 97]
    
    Parameters
    -----------
    mat: ndarray/ matrix
        confusion matrix of an classification result
    class_index: int
        index of class name, start from 0
    
    Returns
    --------
    metrics: dict
        {
            precision: float,
            recall:    float,
            f_value:   float,
            success_rate: float
        }
    
    Examples
    ---------
        >>> matrix = np.matrix([[88,10,2], [14,40,6], [18,10,12]])
        >>> metrics_from_confusion_matrix(matrix, 0) # for class a
        {'precision': 0.73, 'recall': 0.88, 'f_value': 0.4, 'successs_rate': 0.7}
    '''
    pass

def cohen_kappa():
    '''
    compute Cohen's Kappa Coefficient for two classifiers
    '''
    pass

def bayesain_probability(likelihood, prior, probability):
    '''
    compute the posterior probability using bayes theorem
    
    Parameters
    -----------
    likelihood: float
        P(observation|hypothesis)
    prior: float
        P(hypothesis)
    probability: float
        P(observation)
        
    Returns
    --------
    posterior: float
        P(hypothesis|observation)
    '''
    pass

def moving_average():
    '''
    compute moving average of a time series
    '''
    pass

def exponential_smoothing():
    '''
    apply exponential smoothing to a time series
    '''
    pass

def perplexity():
    '''
    compute perplexity of a character string
    '''
    pass


# ======================================================
#                 Infromation Retrieval
#  more info - https://www.jianshu.com/p/75b1137c1b18
# ======================================================


def get_token_stream(tokenized_docs, docs_dict):
    """get (term-doc_id) stream
    """
    token_stream = []
    for doc_id in docs_dict:
        for term in tokenized_docs[doc_id]:
            token_stream.append((term, doc_id))
    return token_stream

def build_indices(tokenized_docs, docs_dict):
    """main function -- build invertex index
       assume that the documents set is small enough to be loaded into Memory
    """
    token_stream = get_token_stream(tokenized_docs, docs_dict)
    # print(token_stream)
    indices = {}

    for pair in token_stream:
        if pair[0] in indices:
            if pair[1] not in indices[pair[0]]:
                indices[pair[0]].append(pair[1])
        else:
            indices[pair[0]] = [pair[1]]
    return indices

def _tf(tokenized_doc):
    """calculate term frequency for each term in each document"""
    term_tf = {}
    for term in tokenized_doc:
        if term not in term_tf:
            term_tf[term]=1.0
        else:
            term_tf[term]+=1.0

    # print(term_tf)
    return term_tf

def _idf(indices, docs_num):
    """calculate inverse document frequency for every term"""
    term_df = {}
    for term in indices:
        # 一个term的df就是倒排索引中这个term的倒排记录表（对应文档列表）的长度 
        term_df.setdefault(term, len(indices[term]))
    
    term_idf = term_df
    for term in term_df:
        term_idf[term] = np.log10(docs_num /term_df[term])
    # print(term_idf)
    return term_idf

def tfidf(tokenized_docs, indices):
    """calcalate tfidf for each term in each document"""
    term_idf = _idf(indices, len(tokenized_docs))
        
    term_tfidf={}
    doc_id=0
    for tokenized_doc in tokenized_docs:
        term_tfidf[doc_id] = {}
        term_tf = _tf(tokenized_doc)
        
        doc_len=len(tokenized_doc)
        for term in tokenized_doc:
            tfidf = term_tf[term]/doc_len * term_idf[term]
            term_tfidf[doc_id][term] =tfidf
        doc_id+=1
    # print(term_tfidf)
    return term_tfidf

def build_terms_dictionary(tokenized_docs):
    """assign an ID for each term in the vocabulary"""
    vocabulary = set()
    for doc in tokenized_docs:
        for term in doc:
            vocabulary.add(term)
    vocabulary = list(vocabulary)
    dictionary = {}
    for i in range(len(vocabulary)):
        dictionary.setdefault(i, vocabulary[i])
    return dictionary

def vectorize_docs(docs_dict, terms_dict, tf_idf):
    """ transform documents to vectors
        using bag-of-words model and if-idf
    """
    docs_vectors = np.zeros([len(docs_dict), len(terms_dict)])

    for doc_id in docs_dict:
        for term_id in terms_dict:
            if terms_dict[term_id] in tf_idf[doc_id]:
                docs_vectors[doc_id][term_id] = tf_idf[doc_id][terms_dict[term_id]]
    return docs_vectors

def vectorize_query(tokenized_query, terms_dict):
    """ transform user query to vectors 
        using bag-of-words model and vector normalization
    """
    query_vector = np.zeros(len(terms_dict))
    for term_id in terms_dict:
        if terms_dict[term_id] in tokenized_query:
            query_vector[term_id] += 1
    return query_vector / np.linalg.norm(query_vector)

def cos_similarity(vector1, vector2):
    """compute cosine similarity of two vectors"""
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))) 

def compute_simmilarity(docs_vectors, query_vector, docs_dict):
    """compute all similarites between user query and all documents"""
    similarities = {}
    for doc_id in docs_dict:
        similarities[doc_id] = cos_similarity(docs_vectors[doc_id], query_vector)
    return similarities

# tokenized_docs = [
#     ['hello', 'world'],
#     ['hello', 'python'],
#     ['i', 'love', 'c', 'java', 'python', 'typescript', 'and', 'php'],
#     ['use', 'python', 'to', 'build', 'inverted', 'indices'],
#     ['you', 'and', 'me', 'are', 'in', 'one', 'world']
#                 ]
# tokenized_query = ["python", "indices"]
# docs_dict = {
#     0: "docs[0]",
#     1: "docs[1]",
#     2: "docs[2]",
#     3: "docs[3]",
#     4: "docs[4]"
# }
# indices = {'and': [2, 4], 'are': [4], 'build': [3], 'c': [2], 'hello': [0, 1], 'i': [2], 
#         'in': [4], 'indices': [3], 'inverted': [3], 'java': [2], 'love': [2], 'me': [4],
#         'one': [4], 'php': [2], 'python': [1, 2, 3], 'to': [3], 'typescript': [2], 'use'
#         : [3], 'world': [0, 4], 'you': [4]}
# tf_idf = tfidf(tokenized_docs, indices)
# terms_dict = build_terms_dictionary(tokenized_docs);
# docs_vectors = vectorize_docs(docs_dict, terms_dict, tf_idf)
# query_vector = vectorize_query(tokenized_query, terms_dict)
# print(compute_simmilarity(docs_vectors, query_vector, docs_dict))
