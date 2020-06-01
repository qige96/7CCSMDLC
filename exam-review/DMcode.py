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
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(C, C)
    idx = np.unravel_index(dists.argmax(), dists.shape)
    if print_log:
        print('distances:\n', dists)
        print('indices of two points in this cluster:\n', idx[0], idx[1])

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
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(C1, C2)
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
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(C1, C2)
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
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(C1, C2)

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

    Examples
    --------
        from Exam 2019 Q 3
        >>> X = np.array([[4, 9],[6, 4],
        ...               [2, 6],[5, 4],
        ...               [9, 3],[7, 3],[6, 1],[4, 2],[9, 8],[8, 8],[2, 5]])
        >>> y = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2])
        >>> owc(X, y) # manually work out as 110.15
        110.14285714285714
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
    
    Examples
    --------
        from Exam 2019 Q 3
        >>> X = np.array([[4, 9],[6, 4],
        ...               [2, 6],[5, 4],
        ...               [9, 3],[7, 3],[6, 1],[4, 2],[9, 8],[8, 8],[2, 5]])
        >>> y = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2])
        >>> obc(X,y) # manually work out as 20.2
        20.530612244897963
    '''
    y = np.array(y)
    bc = 0.0
    centres = []
    for i in set(y):
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
        
        result different from slides because of not rounded intermediate (prob)
        >>> l_branch2 = [10, 8]; entropy(l_branch2) 
        0.9910760598382222
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
        0.0
        >>> info_gain(parent, children2)
        0.10803154614559995
    '''
    return entropy(parent) - sum([(sum(i)/sum(parent))*entropy(i) for i in children])


# ======================================================
#               Classification Metrics
# ======================================================

def build_confmat(predicted, actual, return_type='matrix'):
    '''
    construct a confusion matrix based on predicted and actual results
                        predicted
                -------------------
                |     |  Y  |  N  |
                -------------------
                |  Y  | TP  | FN  |
          actual|  N  | FP  | TN  |
                -------------------
    Call the function from `sklearn` package and reorganise the data so 
    as to fit with the format (as above) in lecture slides.

    Parameters
    ----------
    predicted: list/1darray
        a list of 0 and 1, predicted result for each test samples
    actual: list/1darray
        a list of 0 and 1, actual label for each test samples
    return_type: str
        determine which form of confusion matrix to return, could be
        'matrix', 'DataFrame', or '2d-array'

    Returns
    -------
    mat: matrix/2darray/pd.DataFrame
        confusion matrix for this calssification result

    Example
    -------
        >>> actual = [0, 1, 0, 1]; predicted = [1, 1, 1, 0]
        >>> build_confmat(predicted, actual)
        matrix([[1, 1],
                [2, 0]], dtype=int64)
        >>> build_confmat(predicted, actual, 'DataFrame')
           Y  N
        Y  1  1
        N  2  0
    '''
    y_true = np.array(actual)
    y_pred = np.array(predicted)
    from sklearn.metrics import confusion_matrix
    confmat = confusion_matrix(y_true, y_pred, labels=[1, 0])
    if return_type == 'matrix':
        return np.matrix(confmat)
    elif return_type == 'DataFrame':
        return pd.DataFrame(confmat, columns=['Y','N'], index=['Y','N'])
    # return 2d-array 
    return confmat
    

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
        >>> matrix = np.array([[1, 3], [4, 2]]) # exam 2017 Q 22
        >>> confmat_metrics(matrix)
        {'success_rate': 0.3, 'error_rate': 0.7, 'TP_rate': 0.25, 'FP_rate': 0.6666666666666666}
    '''
    s = (mat[0,0]+mat[1,1]) / mat.sum()
    e = (mat[1,0]+mat[0,1]) / mat.sum()
    tp = mat[0,0] / (mat[0,0] + mat[0,1])
    fp = mat[1,0] / (mat[1,0] + mat[1,1])
    tn = mat[1,1] / (mat[1,0] + mat[1,1])
    fn = mat[0,1] / (mat[0,0] + mat[0,1])
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
        {'precision': 0.7333333333333333, 'recall': 0.88, 'f1': 0.8, 'success_rate': 0.7}
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
            | A  | aa(20) | ab(5) |
          C | B  | ba(10) | bb(15)|
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
    Pr_a = (mat[0, 0] + mat[1, 1]) / n
    Pr_e = (mat[0,0]+mat[1,0])/n * (mat[0,0]+mat[0,1])/n \
            + (mat[1,1]+mat[0,1])/n * (mat[1,1]+mat[1,0])/n
    ck_coef = (Pr_a - Pr_e) / (1 - Pr_e)
    return ck_coef


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

def clear_symbols(text):
    """remove symbols like commas, semi-commas

    :param text: str, or list of str
    :return: str, or list of str, without non alphanumeric simbols
    """
    simbols = re.compile("[\s+\.\!\/_,$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）：]+")
    if type(text) is str:   
        processed_text = re.sub(simbols, ' ', text)
        return processed_text
    elif type(text) is list:
        return [re.sub(simbols, ' ', item) for item in text]
    else:
        raise TypeError("This function only accept str or list as argument")

def lowercase(text):
    """turn all the characters to be lowercase
    
    :param text: str, or list of str
    :return: str, or list of str, with all characters lowercased
    """
    if type(text) is str:
        return text.lower()
    elif type(text) is list:
        return [item.lower() for item in text]
    else:
        raise TypeError("This function only accept str or list as argument")

def tokenize(docs):
    '''tokenize documents, simply split strings according to spaces

    :param docs: list of str
    :return: list of list of str
    '''
    token_stream = []
    for doc in docs:
        token_stream.append(doc.split())
    return token_stream

def preprocess(docs):
    """clear symbols, lowercase, tokenize, get clean tokenized docs
    
    :param docs: list of str
    :return: list of list of str
    """
    normalized_docs = lowercase(clear_symbols(docs))
    tokenized_docs = tokenize(normalized_docs)
    return tokenized_docs


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
        [('a', 0), ('b', 0), ('a', 1), ('c', 1)]
    """
    token_stream = []
    for doc_id, doc in enumerate(tokenized_docs):
        for term in doc:
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
        {'a': [0, 1], 'b': [0], 'c': [1]}
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

def build_term_doc_mat(tokenized_docs):
    '''
    construct a term-document matrix based on given corpus

    Parameters
    ----------
    tokenized_docs: list
        A list of list of strings

    Returns
    -------
    mat: pd.DataFrame
        a term-docuemnt matrix where rows are docs and columns are terms

    Examples
    --------
        >>> t_docs = [['hello', 'world'], ['hello', 'data', 'mining']]
        >>> build_term_doc_mat(t_docs)
           hello  world  data  mining
        0      1      1     0       0
        1      1      0     1       1
    '''
    terms = []
    for doc in tokenized_docs:
        for term in doc:
            if term not in terms:
                terms.append(term)
    mat = np.zeros([len(tokenized_docs), len(terms)], dtype=int)
    for doc_id, doc in enumerate(tokenized_docs):
        for term_id, term in enumerate(terms):
            if term in doc:
                mat[doc_id, term_id] += 1
    return pd.DataFrame(mat, columns=terms)

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
        >>> t_doc = ['a', 'a', 'b']
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
        >>> tfidf(tokens)
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


# ======================================================
#                 Association Rules
# ======================================================

def one_item_sets(txs):
    '''
    construct one-item set from a batch of transactions
    
    Parameters
    ----------
    txs: list
        list of transactions that is a list of items

    Returns
    -------
    one_set: list
        list of one-item set

    Examples
    --------
        from Lecure 6 Slide 12
        >>> txs = [['o','s'], ['m','o','w'], ['o','d'], ['o','d','s'], ['w','s']]
        >>> one_item_sets(txs)
        ['o', 's', 'm', 'w', 'd']
    '''
    one_set = []
    for tx in txs:
        for item in tx:
            if item not in one_set:
                one_set.append(item)
    return one_set

def two_item_sets(one_sets, txs):
    '''
    construct two-item set from one-item set
    
    Parameters
    ----------
    one_sets: list
        one-item sets
    txs: list
        list of transactions that is a list of items

    Returns
    -------
    two_sets: list
        list of two-item sets

    Examples
    --------
        from Lecure 6 Slide 13
        >>> txs = [['o','s'], ['m','o','w'], ['o','d'], ['o','d','s'], ['w','s']]
        >>> one_sets = ['o', 's', 'm', 'w', 'd']
        >>> two_item_sets(one_sets, txs)
        [('o', 's'), ('o', 's'), ('o', 'm'), ('o', 'w'), ('o', 'd'), ('o', 'd'), ('s', 'w'), ('s', 'd'), ('m', 'w')]
    '''
    two_sets = []
    from itertools import combinations_with_replacement
    for tup in combinations_with_replacement(one_sets, 2):
        if len(set(tup)) == 2:
            for tx in txs:
                if (tup[0] in tx) and (tup[1] in tx):
                    two_sets.append(tup)
    return two_sets

def three_item_sets(two_sets, txs):
    '''
    construct three-item set from one-item set
    
    Parameters
    ----------
    two_sets: list
        two-item sets
    txs: list
        list of transactions that is a list of items

    Returns
    -------
    three_sets: list
        list of three-item sets

    Examples
    --------
        from Lecure 6 Slide 13
        >>> txs = [['o','s'], ['m','o','w'], ['o','d'], ['o','d','s'], ['w','s']]
        >>> two_sets = [('o','s'), ('o','d')]
        >>> three_item_sets(two_sets, txs)
        [('o', 's', 'd')]
    '''
    three_sets = []
    items = []
    for i in two_sets:
        for k in i:
            if k not in items:
                items.append(k)

    from itertools import combinations_with_replacement
    for tup in combinations_with_replacement(items, 3):
        if len(set(tup)) == 3:
            for tx in txs:
                if (tup[0] in tx) and (tup[1] in tx) and (tup[2] in tx):
                    three_sets.append(tup)
    return three_sets

def coverage(item_set, txs):
    '''
    count the number of instances covered by the rule

    Parameters
    ----------
    item_set: list
        item set with any number of items
    txs: list
        list of transactions that is a list of items

    Returns
    -------
    count: int
        coverage of a rule given a set of transaactions

    Examples
    --------
        from Lecure 6 Slide 12
        >>> txs = [['o','s'], ['m','o','w'], ['o','d'], ['o','d','s'], ['w','s']]
        >>> set1 = ('o',); coverage(set1, txs)
        4
        >>> set2 = ('o','s'); coverage(set2, txs)
        2
    '''
    count = 0
    for tx in txs:
        if set(tx).issuperset(item_set):
            count += 1
    return count

def support(item_set, txs):
    '''
    calculate the proportion of instances covered by the rule
    
    Parameters
    ----------
    item_set: list
        item set with any number of items
    txs: list
        list of transactions that is a list of items

    Returns
    -------
    proportion: float
        proportion of a rule gain supported given a set of transaactions
    '''
    return coverage(item_set, txs) / len(txs)

def confidence(rule, txs):
    '''
    calculate the proportion of instances that the rule predicts 
    correctly over all instances. Also called accuracy.

    Parameters
    ----------
    rule: tuple
        tuple of tuples, where tup[0] is antecedent and tup[1] is consequent
    txs: list
        list of transactions that is a list of items

    Returns
    -------
    confid: float
        confidence of the given rule

    Examples
    --------
        from Lecure 6 Slide 16
        >>> txs = [['o','s'], ['m','o','w'], ['o','d'], ['o','d','s'], ['w','s']]
        >>> rule1 = (('o',), ('s',)); confidence(rule1, txs)
        0.5
        >>> rule2 = (('s',), ('o',)); confidence(rule2, txs)
        0.6666666666666666
        >>> rule3 = (('o',), ('d',)); confidence(rule3, txs)
        0.5
        >>> rule4 = (('d',), ('o',)); confidence(rule4, txs)
        1.0
    '''
    return coverage(rule[0]+rule[1], txs) / coverage(rule[0], txs)

# ======================================================
#                    Miscellaneous
# ======================================================

def normalize_numeric(arr):
    '''
    perform standardised transformation to a numeric variable by
    firstly subtracting mean, and then divided by std

    Parameters
    ----------
    arr: list/1darray
        array like object

    Returns
    -------
    nor_arr: 1darray
        normalised array

    Examples
    --------
        >>> normalize_numeric([1, 2, 3, 4, 5, 6])
        array([-1.46385011, -0.87831007, -0.29277002,  0.29277002,  0.87831007,
                1.46385011])
    '''
    arr = np.array(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std

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

def naive_bayes(X, y, x, use_laplace=False, print_log=False):
    '''
    compute probability for each event in y useing naive bayes method

    Parameters
    ----------
    X: matrix/2darray
        training data
    y: 1darray
        labels for each rwo in traiing data
    use_laplace: bool, default False
        whether to use Laplace estimation

    Returns
    -------
    probs: dict
        probability for each event (label) in y

    Examples
    --------
        From Lecture 3 Slide 67
        >>> X = np.array([['sunny', 'hot', 'high', False],
        ...        ['sunny', 'hot', 'high', True],
        ...        ['overcast', 'hot', 'high', False],
        ...        ['rainy', 'mild', 'high', False],
        ...        ['rainy', 'cool', 'normal', False],
        ...        ['rainy', 'cool', 'normal', True],
        ...        ['overcast', 'cool', 'normal', True],
        ...        ['sunny', 'mild', 'high', False],
        ...        ['sunny', 'cool', 'normal', False],
        ...        ['rainy', 'mild', 'normal', False],
        ...        ['sunny', 'mild', 'normal', True],
        ...        ['overcast', 'mild', 'high', True],
        ...        ['overcast', 'hot', 'normal', False],
        ...        ['rainy', 'mild', 'high', True]], dtype=object)
        >>> y = np.array(['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes',
        ...         'yes', 'yes', 'yes', 'no'], dtype=object)
        >>> x = np.array(['sunny', 'cool', 'high', True], dtype=object)
        >>> naive_bayes(X, y, x, print_log=False)
        'no'
    '''
    probs = {}
    cls1, cls2 = set(y)

    # compute probability for each attributes in x
    for cls in (cls1, cls2):
        probs[cls] = []
        dat = X[y==cls]
        for attr in x:
            count = 0
            for row in dat:
                if attr in row:
                    count += 1
            if use_laplace:
                probs[cls].append((count+1)/(len(dat)+len(x)))
            else:
                probs[cls].append((count)/(len(dat)))
    
    Pr_cls1 = len(X[y==cls1]) / len(X)
    Pr_cls2 = len(X[y==cls2]) / len(X)
    
    # compute Pr(cls1) / Pr(cls2)
    from functools import reduce
    odd = (reduce(lambda x,y:x * y, probs[cls1]) * Pr_cls1) / \
            (reduce(lambda x,y:x * y,probs[cls2]) * Pr_cls2)
    if odd < 1.0:
        return cls2
    if print_log:
        print('cls1:', cls1, '    cls2:', cls2)
        print('Pr(%s):'%cls1, Pr_cls1, '    Pr(%s):'%cls2, Pr_cls2)
        print('Probability of each attribute in x for each label:\n', probs)
    return cls1

if __name__ == '__main__':
    import doctest
    doctest.testmod()


