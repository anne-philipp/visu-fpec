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

#*******************************************************************************
# MODULES
#*******************************************************************************
from __future__ import print_function

import os
import sys
import time
import inspect
from mods import reading
import __config as cf


PATH_LOCAL_PYTHON = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
# add path to pythonpath
if PATH_LOCAL_PYTHON not in sys.path:
    sys.path.append(PATH_LOCAL_PYTHON)

#*******************************************************************************
# FUNCTIONS
#*******************************************************************************

def main(argv=None):
    """

    Parameters
    ----------

    Return
    ------

    """


    filelist = []

    # store arguments from script call
    if argv is None:
        argv = sys.argv[1:]

    # read in input and output paths
    arguments = reading.get_parameters(argv)
    ipath = arguments['input_path']
    opath = arguments['output_path']
    fpattern = arguments['file_pattern']

    filelist = reading.get_inputfiles(ipath, fpattern)
    filelist = reading.create_sorted_filelist(filelist, cf.VERBOSE)
    datelist = reading.get_datelist(filelist, cf.VERBOSE)
    if cf.VERBOSE:
        print('filelist: ', filelist)
        print('datelist: ', datelist)

    info = reading.get_informations(ipath, filelist[0], cf.VERBOSE)
    domain = reading.get_final_domain(arguments, info, cf.VERBOSE)

    if arguments['param_name'] in cf.FIELD_TYPES_2D:
        reading.read_2D_data(arguments, filelist, datelist, domain,
                             info, cf.VERBOSE)
    elif arguments['param_name'] in cf.FIELD_TYPES_3D:
        matchLevToZ = reading.get_heights()
        reading.read_3D_data(arguments, filelist, datelist, domain,
                             info, matchLevToZ, cf.VERBOSE)
    elif arguments['param_name'] == 'ff':
        matchLevToZ = reading.get_heights()
        reading.read_ff_data(arguments, filelist, datelist, domain,
                             info, matchLevToZ, cf.VERBOSE)
    else:
        print(cf.ERR + ' Field type is not available!')
        print(cf.WRN + ' Did you mis-spell the field name or')
        print(cf.WRN + ' if not mis-spelled it might be that the ')
        print(cf.WRN + ' field name is not yet stored in the variable ')
        print(cf.WRN + ' "FIELD_TYPE_xD". Please add in "__config.py".')


if __name__ == "__main__":
    # --- create tree structure and prepare runs
    start_time_int=time.time()
    main()
    print(" .... End of file --- %s seconds ---" % (time.time() - start_time_int))
