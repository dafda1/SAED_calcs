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
        text = r"$("
        end_text = r")$"
    else:
        raise ValueError(f"Input '{space}' unrecognised. "+\
                         "'space' must be either 'reciprocal' or 'direct'.")
        
    for m in tuple(hkl_array):
        text += Miller_bar(m) + r"\,"
        
    text = text[:-2] + end_text
    
    return text