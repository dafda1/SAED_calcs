# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:14:49 2026

@author: dafda1
"""

import numpy as np
import CellCalcs as cc

from AuxFuncs import sanitise_matrix, invert_matrix

def orient_from_zone_axis (metric_tensor, zone_axis,
                           sanitise_output = True):
    
    cell = cc.standard_root_cell(metric_tensor)
    pimat = cc.compute_pi_matrix(cell)
    
    Qvec_z = np.matmul(cell.T, zone_axis)
    Qvec_z_norm = np.sqrt(np.matmul(Qvec_z, Qvec_z))

    if zone_axis[0] == 0:
        Qvec_x = pimat[:, 0]
    elif zone_axis[1] == 0:
        Qvec_x = pimat[:, 1]
    else:
        Qvec_x = pimat[:, 2]
        
    Qvec_x_norm = np.sqrt(np.matmul(Qvec_x, Qvec_x))

    Qvec_y = np.cross(Qvec_z, Qvec_x)
    Qvec_y_norm = np.sqrt(np.matmul(Qvec_y, Qvec_y))

    Qmat = np.empty((3, 3), dtype = float)
    Qmat[:, 0] = Qvec_x
    Qmat[:, 1] = Qvec_y
    Qmat[:, 2] = Qvec_z

    Qmat_D = np.diag(np.array((Qvec_x_norm, Qvec_y_norm, Qvec_z_norm)))

    orientation = np.matmul(Qmat_D, invert_matrix(Qmat))

    pimat_new = np.matmul(pimat, orientation.T)
    cell_new = np.matmul(cell, orientation.T)
    
    if sanitise_output:
        pimat_new = sanitise_matrix(pimat_new)
        cell_new = sanitise_matrix(cell_new)
    
    return cell_new, pimat_new

def shuffle_reflections (reflections_array, shuffle_parameter):
    new_reflections = np.empty_like(reflections_array)
    
    for i in np.arange(3):
        new_reflections[:, i] = reflections_array[:, (i - shuffle_parameter) % 3]
    
    return new_reflections

def generate_ZOLZ_reflections_from_zone_axis (zone_axis, miller_lim = 10,
                                              metric_reciprocal = None,
                                              Qmax = None):
    
    #check input
    Qmax_given = Qmax is not None
    metric_given = metric_reciprocal is not None
    if np.logical_xor(Qmax_given, metric_given):
        raise ValueError("If limiting reflections by max Q, both " +\
                         "Qmax and metric_reciprocal must be given")
    
    iterator_all = np.arange(-miller_lim, miller_lim + 1)
    
    use_all = np.where(zone_axis == 0, True, False)
    count_zeros = np.count_nonzero(use_all)
    
    reflections = []
    
    if count_zeros == 3:
        raise ValueError("Invalid zone axis")
    
    elif count_zeros == 2:
        shuffle = np.argmin(use_all) #non-zero preferred axis direction
        
        #use default as [u 0 0] zone axis, then shuffle indices at the end
        # to match correct zone axis
        for k in iterator_all:
            for l in iterator_all:
                reflections.append(np.array((0, k, l)))
    
    elif count_zeros == 1:
        shuffle = np.argmax(use_all) #zero preferred axis direction
        
        #use default as [0 v w] zone axis, then shuffle indices at the end
        # to match correct zone axis
        v, w = zone_axis[(shuffle + 1) % 3], zone_axis[(shuffle + 2) % 3]
        lcm = np.lcm(v, w)
        
        for h in iterator_all:
            for n in iterator_all:
                reflections.append(np.array((h, n*lcm//v, -n*lcm//w)))
                
    else: #all non-zero
        shuffle = np.argmin(np.abs(zone_axis))
        
        u = zone_axis[shuffle]
        v, w = zone_axis[(shuffle + 1) % 3], zone_axis[(shuffle + 2) % 3]
        lcm = np.lcm.reduce(zone_axis)
        
        a, b, c = tuple([lcm // z for z in (u, v, w)])
        
        for n in iterator_all:
            k = b*n
            for m in iterator_all:
                l = c*m
                h = -a*(n + m)
                
                reflections.append(np.array((h, k, l)))
    
    reflections = np.array(reflections)
    reflections = shuffle_reflections(reflections, shuffle)
    
    if Qmax_given:
        new_reflections = []
        for hkl in reflections:
            Qval = cc.metric_norm(hkl, metric_reciprocal)
            if Qval < Qmax:
                new_reflections.append(hkl)
        
        reflections = np.array(new_reflections)
        
    return reflections
                
        
def test_generate_ZOLZ_reflections_from_zone_axis (reflections, zone_axis,
                                                   tolerance = 1e-9):
    for hkl in reflections:
        test = np.dot(hkl, zone_axis)
        
        if np.abs(test) > tolerance:
            print(hkl)
            return False
        
    return True