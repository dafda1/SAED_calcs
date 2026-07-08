# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:01:41 2026

@author: dafda1
"""

import numpy as np

from AuxFuncs import sanitise_matrix

degrees = 180/np.pi

def metric_dotprod (vec1, vec2, metric):
    return np.matmul(vec1.T, np.matmul(metric, vec2))

def build_row_matrix (list_of_rows):
    rowmat = np.empty((len(list_of_rows), list_of_rows[0].size),
                      dtype = float)
    
    for i, row in enumerate(list_of_rows):
        rowmat[i] = row
    
    return rowmat

def build_col_matrix (list_of_cols):
    return build_row_matrix(list_of_cols).T

def index_crossprod (i, j):
    if i not in (0, 1, 2) or j not in (0, 1, 2):
        raise ValueError("Input indices not valid. Must be 0, 1, 2.")
    elif i == j:
        raise ValueError("Input indices not valid. Must not be equal.")
    else:
        return 3 - (i + j)

def generate_metric_tensor (apar, bpar = None, cpar = None,
                            alpha = 90, beta = 90, gamma = 90,
                            sanitise_output = True):
    
    if bpar is None:
        bpar = apar
    
    if cpar is None:
        cpar = apar
    
    alpha_rad = alpha/degrees
    beta_rad = beta/degrees
    gamma_rad = gamma/degrees
    
    lengths = np.array((apar, bpar, cpar))
    angles = np.array((alpha_rad, beta_rad, gamma_rad))
    
    metric = np.zeros((3, 3), dtype = float)
    for i in np.arange(3):
        metric[i, i] = lengths[i]**2
        for j in np.arange(i + 1, 3):
            val = lengths[i]*lengths[j]*\
                           np.cos(angles[index_crossprod(i, j)])
                           
            metric[i, j] = val
            metric[j, i] = val
            
    if sanitise_output:
        metric = sanitise_matrix(metric)
    
    return metric

def metric_norm (vector, metric):
    return np.sqrt(metric_dotprod(vec1 = vector,
                                  vec2 = vector,
                                  metric = metric))

def standard_root_cell (metric_tensor):
    evals, evecs = np.linalg.eigh(metric_tensor)
    
    eval_matrix = np.sqrt(np.diag(evals))
    
    return np.matmul(evecs, np.matmul(eval_matrix, evecs.T))

def compute_pi_matrix (cell_matrix):
    return 2*np.pi*np.linalg.inv(cell_matrix.T)