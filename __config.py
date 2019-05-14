#!/usr/bin python
# -*- coding: utf-8 -*-
#*******************************************************************************
# @Author: Anne Philipp (University of Vienna)
#
# @Date: Sun May 5 2019
#
# @License:
#    (C) Copyright 2019.
#    Anne Philipp
#
#    This work is licensed under the Creative Commons Attribution 4.0
#    International License. To view a copy of this license, visit
#    http://creativecommons.org/licenses/by/4.0/ or send a letter to
#    Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#*******************************************************************************

import os
import sys
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# BASIC PRORGAM INFORMATION

__VERSION__ = '1.0'
__DATE__ = '2019-05-05'
__UPDATED__ = '2019-05-14'

PPROGRAM_NAME = os.path.basename(sys.argv[0])
PROGRAM_VERSION = "v" + __VERSION__
PROGRAM_BUILD_DATE = "%s" % __UPDATED__

PROGRAM_VERSION_STRING = '%%prog %s (%s)' % (PROGRAM_VERSION,
                                             PROGRAM_BUILD_DATE)
PROGRAM_LONGDESC = 'Plot ECMWF data fields!'
PROGRAM_LICENSE = "Created by Anne Philipp 2019"

# Messages for print information output
ERR =  "  [ERROR]"
INFO = "   [INFO]"
WRN =  "[WARNING]"


# ------------------------------------------------------------------------------
# DEBUG SELECTION

VERBOSE = True
SHOW_PLOT = True


# ------------------------------------------------------------------------------
# GLOBAL DEFINITIONS

# file with assignment of level number to height in meter
PATH_HEIGHTS = 'data/heights.txt'

# Available fields in ECMWF FLEXPART input files
# Append additional field types if necessary
FIELD_TYPES_2D = ['sp', 'lsp', 'cp', 'sshf', 'ewss', 'nsss', 'ssr',
                  'sd', 'msl', 'tcc', '10u', '10v', '2t', '2d', 'z',
                  'lsm', 'lcc', 'mcc', 'hcc', 'skt', 'stl1', 'swvl1',
                  'sdor', 'cvl', 'cvh', 'fsr']

FIELD_TYPES_3D = ['u', 'v', 't', 'q', 'qc', 'etadot']


# ------------------------------------------------------------------------------
# PLOT SELECTIONS

FIG_X = 10  # size of figure in x
FIG_Y = 8  # size of figure in y
FONT_SIZE = 14 # size of font for plot title, axis and unit label
TIGHT_LAYOUT = True  # reduces space around if True like a tight bounding box

BASEMAP_RESOLUTION = 'i'
# resolution of map:
#'c' (crude - fast),
#'l' (low),
#'i' (intermediate),
#'h' (high),
#'f' (full - slow)

BASEMAP_AREA_THR = 1000  # basemap area threshold

PROJECTION = 'cyl'
# cyl  = cylindrical
# merc = mercator

C_MAP = plt.cm.jet
# use matplotlib colormaps,
# list is avalibale at http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps

ALPHA = 0.7
# transparency of 2d plots, 0 - 1, where 1 means no transparency at all

DATE_FORMAT_OUT = '%a, %d %b %Y %HZ'
# format of date displayed in title of frames

MAJOR_MER_STEP = 20.0  # step of meridians for mother domain in degrees
MAJOR_PAR_STEP = 10.0  # step of parallel for mother domain in degrees

MINOR_MER_STEP = 10.0  # step of meridians for nested domain in degrees
MINOR_PAR_STEP = 5.0  # step of parallel for nested domain in degrees

PLOT_TYPE = 1
# Plot method type
# 1 = imshow
# 2 = pcolormesh
# 3 = contourf
# 4 = cross section

# log scale assumes, that 1 step in levels is 1 order of magnitude
LOG_FLAG = True  # logarithmic scale, set false for linear scale

# number of orders to show in log scale
NUMBER_OF_ORDERS = 6
# maximum order of log levels, use -999 for real data maximum order
LOG_LEVELS_MAX = -999

# number of levels for linear scale, equidistant
NUMBER_OF_LEVELS = 10
# minimum of linear levels, use -999 for real data minimum
LIN_LEVELS_MIN = -999
# maximum of linear levels, use -999 for real data maximum
LIN_LEVELS_MAX = -999

# if the wind vector plot for param_name = ff is selected:
  # * the scaling of the arrows can be adjusted (inches) with
ARROW_SCALE = 4
  # * the maximum wind speed for all plots can be defined
MAX_WIND_SPEED = 35.


PNGS_FLAG = True  # plots are stored as PNG
PDFS_FLAG = False  # plots are stored as PDF




