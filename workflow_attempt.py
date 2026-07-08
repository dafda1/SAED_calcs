# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:30:28 2026

@author: dafda1
"""

import numpy as np
import matplotlib.pyplot as plt

import spot_ID
import Calibration as cal
import CellCalcs as cc

from AuxFuncs import translate_reflections

from os.path import join

#%% import image(s) from tiff file(s) and define metric

#define path to tiff files (not available on Github repo)
tiff_path = r"C:\Users\dafda1\OneDrive - University of St Andrews\Measurements\In-house\E-Micro\TEM\2026.06.17_Sr4Ru2O9-c-axis\tiff"

fnames = {"ZA_001": join(tiff_path, "0009 Ceta.tif"),
          "ZA_unk": join(tiff_path, "0024 Ceta.tif"),
          "ZA_001_out": join(tiff_path, "0008 Ceta.tif")}

frames = {name: plt.imread(fname)[:, :, 0] for name, fname in fnames.items()}

del fnames

metric = cc.generate_metric_tensor(apar = 9.642,
                                   cpar = 8.104,
                                   gamma = 120)

metric_recip = (4*np.pi**2)*np.linalg.inv(metric)

#%% for a frame, identify spots to be used for calibration

frame = frames["ZA_001"]

spots = spot_ID.identify_spots(frame)

fig, ax = spot_ID.show_spots(frame, spots)

#%% calibrate diffractometer constant - circle

# 45, 66, 35
# 14, 15, 21

origin, difC, Qrad = cal.circle_calibration(spots, 14, 15, 21,
                                            metric_tensor = metric,
                                            hkl = np.array((6, 0, 0)))

#show figure to confirm
fig, ax = cal.show_circle_cal(frame, origin, difC, Qrad)

#%% calibrate diffractometer constant - line radius

origin, difC, Qrad = cal.line_calibration(spots, 14,
                                          metric_tensor = metric,
                                          hkl = np.array((6, 0, 0)))

#show figure to confirm
fig, ax = cal.show_line_cal(frame, origin, difC, Qrad)

#%% find orientation from zone axis and spot

cell, pimat = cal.rotate_from_spot(spots, spotID = 14,
                                   zone_axis = np.array((0, 0, 1)),
                                   metric_tensor = metric,
                                   hkl = np.array((6, 0, 0)),
                                   origin = origin, difC = difC)

#%% for GM - show grid

qxlims, qylims = cal.map_pixels_to_Qspace(px = np.array((0, frame.shape[1])),
                                          py = np.array((frame.shape[0], 0)),
                                          origin = origin, difC = difC)

reflections = []
l = 0 #[001] zone axis
mlim = 15
iterator = np.arange(-mlim, mlim)

for h in iterator:
    for k in iterator:
        hkl = np.array((h, k, l))
        
        qx, qy = tuple(np.matmul(pimat.T, hkl)[:2])
        
        if np.logical_and(qx > qxlims[0], qx < qxlims[1]) and\
           np.logical_and(qy > qylims[0], qy < qylims[1]):
               reflections.append(hkl)

reflections = np.array(reflections)

Qvecs = np.matmul(reflections, pimat)

Px, Py = cal.map_Qspace_to_pixels(Qx = Qvecs[:, 0],
                                  Qy = Qvecs[:, 1],
                                  origin = origin,
                                  difC = difC)

fig, ax = plt.subplots(figsize = (6.4, 6.4),
                       layout = "constrained")

maps = ax.imshow(frame)

ax.scatter(Px, Py, s = 150, marker = "o",
           facecolors = "none", edgecolors = "green")

for hkl, px, py in zip(reflections, Px, Py):
    ax.text(px, py, translate_reflections(hkl),
            ha = "left", va = "bottom", color = "w")

Qvals = np.arange(-5, 6, 1)

Pxticks, Pyticks = cal.map_Qspace_to_pixels(Qvals, Qvals,
                                            origin = origin,
                                            difC = difC)

ax.set_xticks(Pxticks, labels = Qvals)
ax.set_yticks(Pyticks, labels = Qvals)

ax.set_xlim(0, frame.shape[1])
ax.set_ylim(frame.shape[0], 0)

ax.set_xlabel(r"$Q_x$ ($\AA^{-1}$)")
ax.set_ylabel(r"$Q_y$ ($\AA^{-1}$)")