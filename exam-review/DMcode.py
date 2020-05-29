'''
Implementation of some algorithms in PNN,
a suppliment of KCL PNN lecture notes

@ author: Ricky Zhu
@ email:  rickyzhu@foxmail.com
'''
from __future__ import print_function
import numpy as np
import pandas as pd


def bc_distance():
  '''
  compute between cluster distance for clustering tasks
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
        >>> metrics_from_confusion_matrix(matrix, 0)
        {'precision': 0.73, 'recall': 0.88, 'f_value': 0.4, 'successs_rate': 0.7}
    '''
    pass

def bayesain_probability(likelihood, prior, probability):
    '''
    compute the posterior probability using bayes theorem
    
    Parameters
    -----------
    likelihood: float
        P(evidence|hypothesis)
    prior: float
        P(hypothesis)
    probability: float
        P(evidence)
        
    Returns
    --------
    posterior: float
        P(hypothesis|evidence)
    '''
    pass
