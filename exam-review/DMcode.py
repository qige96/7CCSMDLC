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
    compute precision, recall, f1_value from confusion matrix.
                         predicted
              -----------------------
              |    |  a  |  b  |  c |
              -----------------------
              | a  | 88  |  10 |  2 | 
       actual | b  | 14  |  40 |  6 |
              | c  | 18  |  10 | 12 |
              ----------------------- 
        For class a, precision = 0.73, recall = 0.88, f1 = 0.8
        Overall, success rate = 0.7
        [from: lec 3 slide 97]
    
    Parameters
    -----------
    mat: ndarray/matrix
        confusion matrix of an classification result
    class_index: int
        index of class name, start from 0
    
    Returns
    --------
    metrics: dict
        {
            precision: float,
            recall:    float,
            f1:   float,
            success_rate: float
        }
    
    Examples
    ---------
        >>> matrix = np.matrix([[88,10,2], [14,40,6], [18,10,12]])
        >>> metrics_from_confusion_matrix(matrix, 0) # for class a
        {'precision': 0.73, 'recall': 0.88, 'f1': 0.8, 'successs_rate': 0.7}
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
    return likelihood * prior / probability

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


def get_token_stream(tokenized_docs):
    """get (term-doc_id) stream

    Parameters
    ----------
    tokenized_docs: list
        A list of list of strings

    Returns
    -------
    toekn_stream: list
        list of tuple (term, doc_id)

    Examples
    --------
        >>> tokens = [['a', 'b'], ['a', 'c']]
        >>> get_token_stream(tokens)
        [('a', 0), ('b', 0), ('a', 1), ('b', 1)]
    """
    token_stream = []
    for doc_id, term in enumerate(tokenized_docs):
        token_stream.append((term, doc_id))
    return token_stream

def build_indices(tokenized_docs):
    """
    build inverted index
    
    Parameters
    ----------
    tokenized_docs: list
        A list of list of strings

    Returns
    -------
    indices: dict
        A dict of which the key is term and value is a list of document id

    Examples
    --------
        >>> tokens = [['a', 'b'], ['a', 'c']]
        >>> build_indices(tokens)
        {'a': [0, 1], 'b':[0], 'c':[1]}
    """
    token_stream = get_token_stream(tokenized_docs)
    indices = {}

    for pair in token_stream:
        if pair[0] in indices: # term 
            if pair[1] not in indices[pair[0]]: # doc_id 
                indices[pair[0]].append(pair[1]) 
        else:
            indices[pair[0]] = [pair[1]]
    return indices

def _tf(tokenized_doc):
    """
    calculate term frequency for each term in one document

    Parameters
    ----------
    tokenized_docs: list
        A list of string 

    Returns
    -------
    term_if: dict
        A dict of {term: frequency}

    Examples
    --------
        >>> doc = ['a', 'a', 'b']
        >>> _tf(t_doc)
        {'a': 2, 'b': 1}
    """
    return dict(textblob(' '.join(tokenized_doc)).word_counts)

def _idf(tokenized_docs, fomula=None):
    """
    calculate inverse document frequency for every term

    Parameters
    ----------
    tokenized_docs: list
        A list of list of string (documents)
    fomula: function
        The fomula to calculate inverse docuemnt frequency,
        by default use Russell & Norvig's fomula

    Returns
    -------
    term_idf: dict
        A dict of {term: idf}

    Examples
    --------
        >>> tokens = [['a', 'b'], ['a', 'c']]
        >>> _idf(tokens)
        {'a': , 'b':, 'c': }
    """
    terms = []
    for doc in tokenized_docs:
        for term in doc:
            if term not in terms:
                terms.append(term)

    def DF(term):
        df = 0
    	for doc in books.values():
        if term in tokenized_docs:
            df += 1
        return df 
    
    if fomula == None:
        def fomula(term):
            return np.log((len(tokenized_docs)-DF(term)+0.5)/(DF(term)+0.5))
    
    def IDF(term):
        return fomula(term)

    term_idf = {}
    for term in terms:
        term_idf[term] = IDF(term)

    return term_idf

def tfidf(tokenized_docs):
    """
    calcalate tfidf for each term in each document

    Parameters
    ----------
    tokenized_docs: list
        A list of list of string (documents)

    Returns
    -------
    term_tfidf: dict
        A dict of {doc_id: {term: tfidf}}

    Examples
    --------
        >>> tokens = [['a', 'b'], ['a', 'c']]
        >>> ifidf(tokens)
        {0:{'a': , 'b':}, 1:{'a':, 'c':  }}
    """
    term_idf = _idf(tokenized_docs)
        
    term_tfidf={}
    for doc_id, tokenized_doc in enumerate(tokenized_docs):
        term_tfidf[doc_id] = {}
        term_tf = _tf(tokenized_doc)
        
        for term in tokenized_doc:
            tfidf = term_tf[term] / len(tokenized_doc) * term_idf[term]
            term_tfidf[doc_id][term] = tfidf
    return term_tfidf



def cos_similarity(vector1, vector2):
    """compute cosine similarity of two vectors"""
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))) 

