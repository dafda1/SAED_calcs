# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:55:01 2026

@author: dafda1
"""

import numpy as np
import matplotlib.pyplot as plt

import CellCalcs as cc

from AuxFuncs import draw_circle, invert_matrix
from orientation import orient_from_zone_axis

#%% auxilliary functions

def check_input_for_radius (Q_radius, metric_tensor, hkl):
    return_Qrad = False
    
    if Q_radius is None:
        if (metric_tensor) is None or (hkl is None):
            raise ValueError("If Q_radius not given, must provide "+\
                             "metric tensor and example Miller indices")
        
        metric_recip = (4*np.pi**2)*invert_matrix(metric_tensor)
        Q_radius = cc.metric_norm(vector = hkl, metric = metric_recip)
        
        return_Qrad = True
        
    return Q_radius, return_Qrad

def calc_circle (spots):
    if spots.shape[0] != 3:
        raise ValueError("Must use 3 spots")
        
    a, b = tuple(spots[0, :2])
    c, d = tuple(spots[1, :2])
    x0, y0 = tuple(spots[2, :2])
    
    alpha = ((x0-a)*(x0-c) + (y0-b)*(y0-d)) / ((b-d)*x0 - (a-c)*y0 + a*d - b*c)
    
    xc = ((a+c) + alpha*(b-d)) / 2
    yc = ((b+d) - alpha*(a-c)) / 2
    r2 = alpha * (a*d - b*c) - a*c - b*d + xc**2 + yc**2
    
    return np.array((xc, yc, np.sqrt(r2)))

def calc_line (spots):
    if spots.shape[0] != 2:
        raise ValueError("Must use 2 spots")
    
    displacement = spots[0, :2] - spots[1, :2]
    distance = np.sqrt(np.sum(displacement**2))
    
    return np.array((spots[0, 0], spots[0, 1], distance))

#%% usable functions

def circle_calibration (spots, spotID_1, spotID_2, spotID_3,
                        Q_radius = None,
                        metric_tensor = None, hkl = None):
    
    Q_radius, return_Qrad =\
        check_input_for_radius(Q_radius, metric_tensor, hkl)
    
    spots_usable = spots[np.array((spotID_1, spotID_2, spotID_3))]
    
    pxc, pyc, prad = calc_circle(spots_usable)
    
    origin = np.array((pxc, pyc))
    difC = prad*1.0/Q_radius
    
    output = [origin, difC]
    if return_Qrad:
        output += [Q_radius,]
    
    return tuple(output)

def show_circle_cal (frame, origin, difC, Q_radius,
                     figsize = (6.4, 6.4),
                     circle_color = "green",
                     colormap = "viridis"):

    fig, ax = plt.subplots(figsize = figsize,
                           layout = "constrained")

    ax.imshow(frame, cmap = colormap)

    ax = draw_circle(ax, center = origin, radius = difC*Q_radius,
                     resolution = 101, color = circle_color)

    ax.plot(origin[0], origin[1], "x", color = circle_color)

    return fig, ax

def line_calibration (spots, spotID,
                      Q_radius = None,
                      metric_tensor = None, hkl = None):
    
    Q_radius, return_Qrad =\
        check_input_for_radius(Q_radius, metric_tensor, hkl)
        
    spots_usable = spots[np.array((0, spotID))]
    
    pxc, pyc, pdis = calc_line(spots_usable)
    
    origin = np.array((pxc, pyc))
    difC = pdis*1.0/Q_radius
    
    output = [origin, difC]
    if return_Qrad:
        output += [Q_radius,]
    
    return tuple(output)

def show_line_cal (frame, origin, difC, Q_radius,
                   figsize = (6.4, 6.4),
                   circle_color = "green",
                   colormap = "viridis"):

    fig, ax = plt.subplots(figsize = figsize,
                           layout = "constrained")

    ax.imshow(frame, cmap = colormap)

    ax = draw_circle(ax, center = origin, radius = difC*Q_radius,
                     resolution = 101, color = circle_color)

    ax.plot(origin[0], origin[1], "x", color = circle_color)
    
    return fig, ax

def map_pixels_to_Qspace (px, py, origin, difC):
    Qx = (px - origin[0])*1.0/difC
    Qy = -(py - origin[1])*1.0/difC
    
    return Qx, Qy

def map_Qspace_to_pixels (Qx, Qy, origin, difC):
    px = origin[0] + difC*Qx
    py = origin[1] - difC*Qy
    
    return px, py

def rotate_from_spot (spots, spotID, zone_axis,
                      metric_tensor, hkl,
                      origin, difC,
                      check_determinant = True):
    
    cell, pimat = orient_from_zone_axis(metric_tensor, zone_axis)
    
    Qvec_pre = np.matmul(pimat.T, hkl)

    Qmat_pre = (1.0/np.matmul(Qvec_pre, Qvec_pre))*\
               np.array([[ Qvec_pre[0], Qvec_pre[1]],
                         [-Qvec_pre[1], Qvec_pre[0]]])

    Qvec_pos = np.array(map_pixels_to_Qspace(px = spots[spotID, 0],
                                             py = spots[spotID, 1],
                                             origin = origin,
                                             difC = difC))

    cos, sin = tuple(np.matmul(Qmat_pre, Qvec_pos))
    
    if check_determinant:
        Rot_det = cos**2 + sin**2
        if np.abs(Rot_det - 1) > 1e-10:
            raise ValueError(f"Rotation determinant is {Rot_det}, should be 1.")

    Rot_correction = np.array([[cos, -sin, 0],
                               [sin,  cos, 0],
                               [  0,    0, 1]])

    pimat_new = np.matmul(pimat, Rot_correction.T)
    cell_new = np.matmul(cell, Rot_correction.T)
    
    return cell_new, pimat_new

def calc_angle_from_spots (spots, spotID1, spotID2, spotID3,
                           unit_degrees = True):
    
    disp1 = spots[spotID1, :2] - spots[spotID2, :2]
    disp2 = spots[spotID3, :2] - spots[spotID2, :2]
    
    norm = 1.0/np.sqrt(np.sum(disp1**2)*np.sum(disp2**2))
    
    mult = (1, cc.degrees)[int(unit_degrees)]
    
    return np.arccos(np.dot(disp1, disp2)*norm)*mult