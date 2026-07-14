# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:50:45 2026

@author: dafda1
"""

import numpy as np

def draw_circle (axis, center, radius, resolution = 51,
                 fmt = "-", color = "green"):
    tvec = np.linspace(0, 2*np.pi, resolution)
    
    xvec = center[0] + radius*np.cos(tvec)
    yvec = center[1] + radius*np.sin(tvec)
    
    axis.plot(xvec, yvec, fmt, color = color)
    
    return axis

def Miller_bar (index):
    text = "%i" % np.abs(index)
    
    if index < 0:
        return r"\bar{" + text + r"}"
    else:
        return text

def translate_reflections (hkl_array, space = "reciprocal"):
    "integer Miller indices / real space indices only"
    
    if space == "reciprocal":
        text = r"$("
        end_text = r")$"
    elif space == "direct":
        text = r"$["
        end_text = r"]$"
    else:
        raise ValueError(f"Input '{space}' unrecognised. "+\
                         "'space' must be either 'reciprocal' or 'direct'.")
        
    for m in tuple(hkl_array):
        text += Miller_bar(m) + r"\,"
        
    text = text[:-2] + end_text
    
    return text

def sanitise_matrix (matrix, tolerance = 1e-10):
    return np.where(np.abs(matrix) > tolerance, matrix, 0)

def invert_matrix (square_matrix, sanitise_output = True):
    shape = square_matrix.shape
    if (len(shape) != 2) or (shape[0] != shape[1]):
        raise ValueError(f"Array not a square matrix, shape is f{shape}")
        
    n = shape[0]
        
    if n > 3: 
        return np.linalg.inv(square_matrix)
    
    elif n == 1:
        return 1.0/square_matrix
    
    else:
        Vol = np.linalg.det(square_matrix)
        
        if n == 2:
            adjugate = np.array((square_matrix[1, 1], -square_matrix[1, 0]),
                                (-square_matrix[0, 1], square_matrix[0, 0]))
        
        elif n == 3:
            adjugate = np.empty(shape, dtype = float)
            for i in np.arange(3):
                adjugate[:, i] = np.cross(square_matrix[(i + 1) % 3],
                                          square_matrix[(i + 2) % 3])
        
        else:
            return np.linalg.inv(square_matrix)
        
        output = adjugate/Vol
        
        if sanitise_output:
            output = sanitise_matrix(output)
            
        return output