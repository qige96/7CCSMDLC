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


def square_of_distance(C, b):
    C = np.array(C)
    b = np.array(b)
    return np.linalg.norm(C-b)**2 

def owc(X, y):
    '''
    compute overall within cluster distance
    
    :param X: n by m matrix - feature data
    :param y: n vecter - label data
    :return: floating number - within cluster distance
    '''
    y = np.array(y)
    wc = 0.0
    for i in range(len(set(y))):
        c = np.mean(X[y==i], axis=0)
        for x in X[y==i]:
            wc += square_of_distance(x, c)
    return wc

def obc(X, y):
    '''
    compute overall between cluster distance
    
    :param X: n by m matrix - feature data
    :param y: n vecter - label data
    :return: floating number - between cluster distance
    '''
    y = np.array(y)
    bc = 0.0
    centres = []
    for i in range(len(set(y))):
        centres.append(np.mean(X[y==i], axis=0))
    for i in range(len(centres)):
        for j in range(i+1, len(centres)):
            bc += square_of_distance(centres[i], centres[j])
    return bc

def overall_clustering_score(X, y):
    '''
    compute overall clustering score
    OC = BC / WC
    
    :param X: n by m matrix - feature data
    :param y: n vecter - label data
    :return: floating number - overall clustering score
    '''
    return obc(X,y) / owc(X,y)

def calinski_harabaz():
    '''
    compute Calinski-Harabaz Index for evaluating clustering tasks
    '''
    pass

def silhouette_coecient():
    '''
    compute Silhouette Coecient for evaluating clustering tasks
    '''
    pass

def category_utility_metric():
    '''
    compute category utility metric for intremental clustering tree 
    '''
    pass

def gini(branch:list):
    '''
    compute Gini coeffitient of a branch of a tree split
    
    Parameters
    ----------
    branch: list
        number of all instances for each class in a branch
        
    Returns
    -------
    gini_coef: float
        Gini coeffitient
    
    Examples
    --------
        >>> l_branch1 = [5, 5]; gini(l_branch1) # from lec 3 slide 34
        0.5
        >>> l_branch2 = [10, 8]; gini(l_branch2)
        0.5072
        >>> l_branch3 = [10, 0]; gini(l_branch3)
        1.0
    '''
    gini_coef = 0
    for datum in branch:
        gini_coef += (datum / sum(branch))**2
    return gini_coef

def entropy(branch:list):
    '''
    compute entropy of a branch of a tree split
    
    Parameters
    ----------
    branch: list
        number of all instances for each class in a branch
        
    Returns
    -------
    etp: float
        Entropy

    Examples
    --------
        >>> l_branch1 = [5, 5]; entropy(l_branch1) # from lec 3 slide 39
        1.0
        >>> l_branch2 = [10, 8]; entropy(l_branch2)
        0.9896
        >>> l_branch3 = [10, 0]; entropy(l_branch3)
        0
    '''
    etp = 0
    for dataum in branch:
        prob = dataum / sum(branch)
        etp += prob * np.log2(prob)
    return (-1)*etp
    
def info_gain(parent:list, children:list):
    '''
    compute information gain of a tree split

    Parameters
    ----------
    parent: list
        number of all instances for each class in a node (parent node)
    children: list (of lists)
        number of instances for each class in each child node

    Returns
    -------
    ig: float
        information gain of a split

    Examples
    --------
        >>> parent = [10, 10]
        >>> children1 = [[5, 5], [5, 5]]; children2 = [[10, 8], [0, 2]]
        >>> info_gain(parent, children1)
        0
        >>> info_gain(parent, children2)
        0.1
    '''
    return entropy(parent) - sum([(sum(i)/sum(parent))*entropy(i) for i in children])

def metrics_from_confusion_matrix(mat, cls_i:int):
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
    cls_i: int
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
        {'precision': 0.73, 'recall': 0.88, 'f1': 0.8, 'success_rate': 0.7}
    '''
    p = mat[cls_i, cls_i] / np.sum(mat[:, cls_i])
    r = mat[cls_i, cls_i] / np.sum(mat[cls_i, :])
    f = (2 * p * r) / (p + r)
    s = mat.diagonal().sum() / mat.sum()
    return {
            'precision':p,
            'recall': r,
            'f1': f,
            'success_rate': s
            }

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

