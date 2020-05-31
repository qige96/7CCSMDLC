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


# ======================================================
#                 Clustering Metrics
# ======================================================

def cluster_diameter(C, print_log=False):
    '''
    compute The largest distance between any two points in a cluster

    Parameters
    ----------
    C: ndarray
        A cluster of points (vectors)
    print_log: bool
        whether to print intermediate results

    Returns
    -------
    dist: float
        cluster diameter

    Examples
    --------
        >>> cluster = np.array([[0,0], [1,1], [1,2]])
        >>> cluster_diameter(cluster)
        2.23606797749979
    '''
    from scipy.spatial.distance import cdist
    dists = cdist(C, C)
    idx = np.unravel_index(dists.argmax(), dists.shape)
    if print_log:
        print('distances:\n', dists)
        print('indices of two points in this cluster:\n', idx[0], idx[1])
        print('The two points:\n', C[idx[0]], C[idx[1]])
    return dists.max()

def single_link(C1, C2, print_log=False):
    '''
    The minimum distance between any two points in the two different clusters
    
    Parameters
    ----------
    C1, C2: ndarray
        Two clusters of points (vectors)
    print_log: bool
        whether to print intermediate results

    Returns
    -------
    dist: float
        single linkage

    Examples
    --------
        >>> cluster1 = np.array([[0,0], [1,1], [1,2]])
        >>> cluster2 = np.array([[2,2], [2,3], [3,4]])
        >>> single_link(cluster1, cluster2)
        1.0
    '''
    from scipy.spatial.distance import cdist
    dists = cdist(C1, C2)
    idx = np.unravel_index(dists.argmin(), dists.shape)

    if print_log:
        print('distances:\n',dists)
        print('indices of two samples in C1 and C2:\n', idx[0], idx[1])
        print('The two points:\n', C1[idx[0]], C2[idx[1]])
    return dists.min()

def complete_link(C1, C2, print_log=False):
    '''
    The maximum distance between any two points in the two different clusters
    
    Parameters
    ----------
    C1, C2: ndarray
        Two clusters of points (vectors)
    print_log: bool
        whether to print intermediate results

    Returns
    -------
    dist: float
        complete linkage

    Examples
    --------
        >>> cluster1 = np.array([[0,0], [1,1], [1,2]])
        >>> cluster2 = np.array([[2,2], [2,3], [3,4]])
        >>> complete_link(cluster1, cluster2)
        5.0
    '''
    from scipy.spatial.distance import cdist
    dists = cdist(C1, C2)
    idx = np.unravel_index(dists.argmax(), dists.shape)

    if print_log:
        print('distances:\n', dists)
        print('indices of two samples in C1 and C2:\n', idx[0], idx[1])
        print('The two points:\n', C1[idx[0]], C2[idx[1]])
    return dists.max()

def average_link(C1, C2,print_log=False):
    '''
    The average distance of every paire of points in the two different clusters
    
    Parameters
    ----------
    C1, C2: ndarray
        Two clusters of points (vectors)
    print_log: bool
        whether to print intermediate results

    Returns
    -------
    dist: float
        average linkage

    Examples
    --------
        >>> cluster1 = np.array([[0,0], [1,1], [1,2]])
        >>> cluster2 = np.array([[2,2], [2,3], [3,4]])
        >>> average_link(cluster1, cluster2)
        2.6591613225184823
    '''
    from scipy.spatial.distance import cdist
    dists = cdist(C1, C2)

    if print_log:
        print('distances:\n', dists)

    return dists.mean()

def centroid_link(C1, C2, print_log=False):
    '''
    The distance between centroids in the two different clusters
    
    Parameters
    ----------
    C1, C2: ndarray
        Two clusters of points (vectors)
    print_log: bool
        whether to print intermediate results

    Returns
    -------
    dist: float
        average linkage

    Examples
    --------
        >>> cluster1 = np.array([[0,0], [1,1], [1,2]])
        >>> cluster2 = np.array([[2,2], [2,3], [3,4]])
        >>> centroid_link(cluster1, cluster2)
        2.6034165586355518
    '''
    c1 = C1.mean(axis=0)
    c2 = C2.mean(axis=0)
    if print_log:
        print('centroids of two clusters:\n', c1, '\n', c2)
    return np.linalg.norm(c1-c2)

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

def calinski_harabaz(X, y):
    '''
    compute Calinski-Harabaz Index for evaluating clustering tasks
    simply call the corresponding function in `sklearn` package

    :param X: n by m matrix - feature data
    :param y: n vecter - label data
    :return: floating number - Calinski-Harabaz Index
    '''
    # n, K = len(X), len(set(y))
    # return (obc(X, y)/owc(X, y)) * ((n-K) / (k-1))
    from sklearn.metrics import calinski_harabasz_score
    return calinski_harabasz_score(X, y)

def silhouette_coeffcient(X, y):
    '''
    compute the average Silhouette Coeffcient of all samples,
    simply call the function in `sklearn` package
    
    :param X: n by m matrix - feature data
    :param y: n vecter - label data
    :return: floating number - mean silhouette coef of all samples
    '''
    from sklearn.metrics import silhouette_score
    return silhouette_score(X, y)


# ======================================================
#                 Tree Split Metrics
# ======================================================

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
    for datum in branch:
        if datum == 0:
            continue
        prob = datum / sum(branch)
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
        0.108
    '''
    return entropy(parent) - sum([(sum(i)/sum(parent))*entropy(i) for i in children])


# ======================================================
#               Classification Metrics
# ======================================================

def confmat_metrics(mat):
    '''
    compute success rate, error rate, true positive rate, false positive rate,
    given a binary classification confusion matrix.
                        predicted
                -------------------
                |     |  Y  |  N  |
                -------------------
                |  Y  | TP  | FN  |
          actual|  N  | FP  | TN  |
                -------------------

    Parameters
    ----------
    mat: ndarray/matrix
        confusion matrix of an BINARY classification result

    Returns
    -------
    metrics: dict
        {
            'success_rate': float,
            'error_rate': float,
            'TP_rate': float,
            'FP_rate': float,
        }

    Examples
    --------
        >>> matrix = np.array([[1, 3], [4, 2]])
        >>> confmat_metrics(matrix)
        {'success_rate': 0.3,
         'error_rate': 0.7,
         'TP_rate': 0.25,
         'FP_rate': 0.6666666666666666}
    '''
    s = (mat[0,0]+mat[1,1]) / mat.sum()
    e = (mat[1,0]+mat[0,1]) / mat.sum()
    tp = mat[0,0] / (mat[0,0] + mat[0,1])
    fp = mat[1,0] / (mat[1,0] + mat[1,1])
    return {
            'success_rate': s,
            'error_rate': e,
            'TP_rate': tp,
            'FP_rate': fp
            }

def metrics_from_general_confusion_matrix(mat, cls_i:int):
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
        >>> metrics_from_general_confusion_matrix(matrix, 0) # for class a
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

def cohen_kappa(mat):
    '''
    compute Cohen's Kappa Coefficient for two classifiers
    suppose that the results of two classifiers C and C' are as below
                       C'
            -----------------------
            |    |    A   |   B   |
            -----------------------
            | A  | aa(20) | bb(5) |
          C | B  | ba(10) | ab(15)|
            -----------------------
        n = aa+bb+ab+ba = 50
        Pr(a) = (aa+bb) / n = 0.7
        Pr(e) = (aa+ba)/n * (aa+ab)/n + (bb+ab)/n * (bb+ba)/n = 0.5
        ck_coef = (Pr(a) - Pr(e)) / (1 - Pr(e)) = 0.4
        [see https://en.wikipedia.org/wiki/Cohen%27s_kappa#Simple_example]

    Parameters
    ----------
    mat: matrix/ndarray
        result matrix of two classifiers

    Returns
    -------
    ck_coef: float
        Cohen Kappa Coefficient of these two classifiers

    Examples
    --------
        >>> matrix = np.matrix([[20, 5], [10, 15]])
        >>> cohen_kappa(matrix)
        0.4
    '''
    n = mat.sum()
    Pr_a = mat[0].sum()
    Pr_e = (mat[0,0]+mat[1,0])/n * (mat[0,0]+mat[1,1])/n \
            + (mat[0,1]+mat[1,1])/n * (mat[0,1]+mat[1,0])/n
    ck_coef = (Pr_a - Pr_e) / (1 - Pr_e)
    return ck_coef

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


# ======================================================
#                    Time Series
# ======================================================

def moving_average(series, window_size):
    '''
    apply moving average to a time series
    '''
    s = pd.Series(series)
    return s.rolling(window=window_size).mean().to_list()

def exponential_smoothing(series, alpha):
    '''
    apply exponential smoothing to a time series
    '''
    # estimated = [series[0]] 
    # for t in range(1, len(series)):
    #     estimated.append(alpha*series[t] + (1-alpha)*series[t-1])
    # return estimated
    s = pd.Series(series)
    return s.ewm(alpha=alpha).mean().to_list()


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
    require `textblob` package
        
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
    import textblob
    return dict(textblob.TextBlob(' '.join(tokenized_doc)).word_counts)

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
        {'a': -1.6, 'b': 0, 'c': 0}
    """
    terms = []
    for doc in tokenized_docs:
        for term in doc:
            if term not in terms:
                terms.append(term)

    def DF(term):
        df = 0
        for doc in tokenized_docs:
            if term in doc:
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
        {0:{'a': -0.805, 'b': 0}, 1:{'a': -0.805, 'c': 0}}
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

