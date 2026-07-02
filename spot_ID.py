# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:28:56 2026

@author: dafda1
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage import measure

from AuxFuncs import draw_circle

#%% auxilliary functions

def correct_contours (contours_list):
    "corrects to xy instead of yx"
    new_contours = []
    for contour in contours_list:
        tmp = np.empty_like(contour)
        tmp[:, 0] = contour[:, 1]
        tmp[:, 1] = contour[:, 0]
        new_contours.append(tmp)
    
    return new_contours

def calc_center (contour):
    return np.sum(contour, axis = 0)/contour.shape[0]

def calc_ave_radius (contour, center):
    displacements = contour - center
    return np.sqrt(np.sum(displacements**2)/contour.shape[0])

def get_center_and_radius (contour):
    center = calc_center(contour)
    return center, calc_ave_radius(contour, center)

#%% usable functions

def identify_spots (frame, sigma_blur = 5, isovalue = 200):
    "returns list of spots (in array form) in decreasing radii: px, py, radius"
    
    frame_blur = gaussian_filter(frame, sigma = sigma_blur)
    
    contours = correct_contours(measure.find_contours(frame_blur, isovalue))
    
    spots = np.empty((len(contours), 3), dtype = float)
    
    for i, contour in enumerate(contours):
        center, radius = get_center_and_radius(contour)
        spots[i, :2] = center
        spots[i, 2] = radius

    order = np.argsort(spots[:, 2])
    spots = spots[np.flip(order)]
    
    return spots

def show_spots (frame, spots,
                figsize = (6.4, 6.4),
                spots_color = "green",
                text_color = "white",
                colormap = "viridis"):
    
    fig, ax = plt.subplots(figsize = figsize,
                           layout = "constrained")

    ax.imshow(frame, cmap = colormap)

    for i, spot in enumerate(spots):
        ax = draw_circle(ax, spot[:2], spot[2], color = spots_color)
        
        ax.text(spot[0], spot[1], i,
                ha = "left", va = "bottom",
                color = text_color)
        
    ax.plot(spots[:, 0], spots[:, 1], "x", color = spots_color)
    
    return fig, ax