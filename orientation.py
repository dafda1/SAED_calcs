# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:14:49 2026

@author: dafda1
"""

import numpy as np
import MyPyRSP.CellCalcs as cc

def sanitise_matrix (matrix, tolerance = 1e-10):
    return np.where(np.abs(matrix) > tolerance, matrix, 0)

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

    orientation = np.matmul(Qmat_D, np.linalg.inv(Qmat))

    pimat_new = np.matmul(pimat, orientation.T)
    cell_new = np.matmul(cell, orientation.T)
    
    if sanitise_output:
        pimat_new = sanitise_matrix(pimat_new)
        cell_new = sanitise_matrix(cell_new)
    
    return cell_new, pimat_new