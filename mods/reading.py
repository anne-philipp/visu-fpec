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
import fnmatch
import inspect
from datetime import datetime, timedelta
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
import xarray as xr
from eccodes import *

import plotting
import __config as cf

#*******************************************************************************
# FUNCTIONS
#*******************************************************************************

def get_inputfiles(ipath, matchingstring, verb=False):
    '''Get input filenames from input path matching the string or
    regular expression and return it.

    Parameters
    ----------
    ipath: string
        Path to the files.

    matchingstring: string
        A string with or without regular expression.

    verb : logical
        Decides if additional information is printed.

    Return
    ------
    filelist: list of strings
        A list of all files matching the pattern of matchingstring.
    '''

    # read input files and store them in a list

    filelist = []

    print('Search for input files in ' + ipath)
    print('Filemask is ' + matchingstring)

    filelist = [k for k in os.listdir(ipath)
                if fnmatch.fnmatch(k, matchingstring)]

    print(str(len(filelist)) + ' files found!')

    if verb: print('The input files are: ', filelist)

    print('\n')

    return filelist


def get_heights():
    '''Reads a file which contains the assignment of ECMWF height levels to
    heights in meter.

    Parameters
    ----------

    Return
    ------
    matchLevToZ : dict-like
        Assignment of height levels to height in meter.

    '''

    # read level heights in meter for title
    with open(cf.PATH_HEIGHTS, 'r') as f:
        fdata = f.read().split('\n')

    matchLevToZ = {}
    for l in fdata:
        line = l.split()
        if line:
            matchLevToZ[int(line[0])] = line[6]

    return matchLevToZ


def get_frames_to_plot(path_frames):
    '''Reads ...

    Parameters
    ----------

    Return
    ------
    frames : list of

    '''
 #   receptors = {}
 #
 #   with open(path_frames) as f:
 #       s = f.readlines()
 #
 #       for i,line in enumerate(s):
 #           spl = line.split()
 #           spl[:2] = map(float, spl[:2]) #mapping to float values
 #           if len(spl) < 3:
 #               spl += [""] #in no station name is entered
 #           receptors[i] = spl
    print('FRAMES TO BE READ')
    return


def get_datelist(filelist, verb=False):
    '''Gets a list of all dates/times from the filelist.

    Parameters
    ----------
    filelist : list of str
        List of filenames.

    verb : logical
        Decides if additional information is printed.

    Return
    ------
    datelist : list of datetime
       A list of all dates to be read.
    '''

    datelist = []
    for file in filelist:
        datelist.append(datetime.strptime(file[-8:], '%y%m%d%H'))

    return datelist


def none_or_str(value):
    '''Converts the input string into pythons None - type if the string
    contains string "None".

    Parameters
    ----------
    value : str
        String to be checked for the "None" word.

    Return
    ------
    None or value:
        Return depends on the content of the input value. If it was "None",
        then the python type None is returned. Otherwise the string itself.
    '''
    if value == 'None':
        return None
    return value


def none_or_int(value):
    '''Converts the input string into pythons None-type if the string
    contains string "None". Otherwise it is converted to an integer value.

    Parameters
    ----------
    value : str
        String to be checked for the "None" word.

    Return
    ------
    None or int(value):
        Return depends on the content of the input value. If it was "None",
        then the python type None is returned. Otherwise the string is
        converted into an integer value.
    '''
    if value == 'None':
        return None
    return int(value)


def get_parameters(parlist):
    '''Decomposes the command line arguments and assigns them to variables.
    Apply default values for non mentioned arguments.

    Parameters
    ----------
    argv : Namespace
        Contains all command line arguments.

    Return
    ------
    args : dict-like
        Contains the commandline arguments from script/program call.
    '''

    parser = ArgumentParser(version=cf.PROGRAM_VERSION_STRING,
                            epilog=cf.PROGRAM_LONGDESC,
                            description=cf.PROGRAM_LICENSE,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    # control parameters that override control file values
    parser.add_argument("-i", "--input_path", dest="input_path",
                        type=none_or_str, default=None,
                        help="Path to ECMWF files.")
    parser.add_argument("-o", "--output_path", dest="output_path",
                        type=none_or_str, default=None,
                        help="Path where the plots are stored.")
    parser.add_argument("-f", "--file_pattern", dest="file_pattern",
                        type=none_or_str, default=None,
                        help="Matching string for input files.")
    parser.add_argument("-p", "--param", dest="param_name",
                        type=none_or_str, default=None,
                        help="Parameter name of ECMWF field.")
    parser.add_argument("-l", "--levels", dest="levels",
                        type=int, nargs=2, default=None,
                        help="Single level or level list. \
                        E.g. z0 z1 (z0=z1 to select just one level)")
    parser.add_argument("-m", "--max_lat_lon", dest="max_lat_lon",
                        action="store_true", help="Takes maximum available \
                        lat-lon domain (overrides -d)")
    parser.add_argument("-d", "--domain", dest="domain",
                        type=float, nargs=4, default=None,
                        help="Coordinates of the domain: \
                        ll_lon, ll_lat, ur_lon, ur_lat")

    parser.add_argument("-u", "--unitlabel", type=none_or_str, dest="unitlabel",
                        help="String containing units of visualized quantity")
    parser.add_argument("-x", "--title", type=none_or_str, dest="title",
                        help="Title of images")
    parser.add_argument("-z", "--projection", type=none_or_str, dest="projection",
                        help="Map projection \
                        [cylindrical:cyl (default), Mercator:merc]")

    args = parser.parse_args()

    return vars(args)


def create_sorted_filelist(filelist, verb=False):
    '''Gets a filelist and returns a sorted filelist.

    Parameters
    ----------
    filelist : list of str
        List of filenames.

    verb : logical
        Decides if additional information is printed.

    Return
    ------
    filelist : list of str
        List of filenames.
    '''

    from collections import Counter

    # check if times are unique
    if [i for i,c in Counter(filelist).items() if c>1] != []:
        print('ERROR: At least one file is a duplicate!')

    # to be sure that the time order is correct,
    # sort the datelist again
    filelist.sort()

    return filelist


def get_informations(ipath, filename, verb=False):
    '''Retrieves some basic information about dataset from first file in
    filelist. Informations are:
    Ni, Nj, latitudeOfFirstGridPointInDegrees,
    longitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees,
    longitudeOfLastGridPointInDegrees, jDirectionIncrementInDegrees,
    iDirectionIncrementInDegrees, missingValue

    Parameters
    ----------
    ipath : str
        Path to the files from filelist.

    filename : str
        The name of an individual file of the filelist.

    verb : logical
        Decides if additional information is printed.

    Return
    ------
    info : dict-like
        Contains some basic information of dataset.
        Ni, Nj, latitudeOfFirstGridPointInDegrees,
        longitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees,
        longitudeOfLastGridPointInDegrees, jDirectionIncrementInDegrees,
        iDirectionIncrementInDegrees, missingValue
    '''
    data = {}

    # --- open file ---
    print("Opening file for getting information data --- %s" % filename)

    keys = [
        'Ni',
        'Nj',
        'latitudeOfFirstGridPointInDegrees',
        'longitudeOfFirstGridPointInDegrees',
        'latitudeOfLastGridPointInDegrees',
        'longitudeOfLastGridPointInDegrees',
        'jDirectionIncrementInDegrees',
        'iDirectionIncrementInDegrees',
        'missingValue',
    ]

    # just get the key information
    with open(ipath+filename) as f:
        # load first message from file
        gid = codes_grib_new_from_file(f)
        if verb:
            print('\nInformations are: ')
        for key in keys:
            # Get the value of the key in a grib message.
            data[key] = codes_get(gid, key)
            if verb:
                print("%s = %s" % (key, data[key]))

    # get information about the numbers of levels
    # in this case just from param 131 which should be there all the time
    level = set()
    with open(ipath+filename) as f:
        while True:
            # load first message from file
            gid = codes_grib_new_from_file(f)
            if not gid: break
            # information needed from grib message
            paramId = codes_get(gid,'paramId')
            if paramId == 131:
                level.add(codes_get(gid,'level'))

            # Free the memory for the message referred as gribid.
            codes_release(gid)

    levels = list(level)
    levels.sort()
    data['levels'] = levels
    if verb: print("%s = %s" % ('levels', data['levels']))
    if verb: print('\n')

    return data


def get_final_domain(args, info, verb=False):
    '''Defines the final domain boundaries.

    Parameters
    ----------
    args : dict-like
        Contains the parameters of the command line call.

    info : dict-like
        Contains some basic information of dataset.
        Ni, Nj, latitudeOfFirstGridPointInDegrees,
        longitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees,
        longitudeOfLastGridPointInDegrees, jDirectionIncrementInDegrees,
        iDirectionIncrementInDegrees, missingValue

    verb : logical
        Decides if additional information is printed.

    Return
    ------
    domain : list of float
        Contains the lower left and upper right corners of latitude and
        longitude. E.g. llx, lly, urx, ury

    '''

    if not args['domain']:
        llx = info['longitudeOfFirstGridPointInDegrees']
        lly = info['latitudeOfLastGridPointInDegrees']

        urx = info['longitudeOfLastGridPointInDegrees']
        ury = info['latitudeOfFirstGridPointInDegrees']
    else:
        llx = args['domain'][0]
        lly = args['domain'][1]
        urx = args['domain'][2]
        ury = args['domain'][3]

        if info['longitudeOfLastGridPointInDegrees'] > 180. and \
           llx < 0.:
            llx = llx + 360.

        if info['longitudeOfLastGridPointInDegrees'] > 180. and \
           urx < 0.:
            urx = urx + 360.

        # test for cross section plot
        # minimum one horizontal coordinate must be constant
        if cf.PLOT_TYPE == 4:
            if not (llx == urx or lly == ury):
                print(cf.ERR + ' For a cross section plot at least one!')
                print(cf.ERR + ' horizontal coordinate must be constant!')
                print(cf.ERR + ' EXIT PROGRAM WITH ERROR!')
                sys.exit()

        if llx < info['longitudeOfFirstGridPointInDegrees'] or \
           urx > info['longitudeOfLastGridPointInDegrees'] or \
           lly < info['latitudeOfLastGridPointInDegrees'] or \
           ury > info['latitudeOfFirstGridPointInDegrees']:
            print(cf.ERR + ' Domain selection is out of range!')
            print(cf.ERR + ' EXIT PROGRAM WITH ERROR!')
            sys.exit()

    return llx, lly, urx, ury


def decide_on_datafield(args, info, datelist, verb=False):
    '''Declarate the data fields for reading EC data depending on the size
    of field.

    Parameters
    ----------
    args: dict - like
        Contains the parameters of the command line call.

    info: dict - like
        Contains some basic information of dataset.
        Ni, Nj, latitudeOfFirstGridPointInDegrees,
        longitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees,
        longitudeOfLastGridPointInDegrees, jDirectionIncrementInDegrees,
        iDirectionIncrementInDegrees, missingValue

    datelist : list of datetime
       A list of all dates to be read.

    verb : logical
        Decides if additional information is printed.

    Return
    ------
    data : DataArray
        Field to store the data for plotting. Size depends on the
        parameter(2D vs 3D) and level list.

    u : DataArray
        Specific extra field for the u wind component if the wind speed
        and direction should be plotted. It is needed for calculation.

    v : DataArray
        Specific extra field for the v wind component if the wind speed
        and direction should be plotted. It is needed for calculation.

    levRange : list of int
        List of levels which should be plotted.
    '''

    if not args['levels']:
        args['levels'] = info['levels']

    levMin = args['levels'][0]
    levMax = args['levels'][-1]
    levRange = range(levMin, levMax + 1, 1)
    if verb:
        print('level range: ', levRange)

    llx = info['longitudeOfFirstGridPointInDegrees']
    lly = info['latitudeOfLastGridPointInDegrees']
    urx = info['longitudeOfLastGridPointInDegrees']
    ury = info['latitudeOfFirstGridPointInDegrees']

    numxgrid = info['Ni']
    numygrid = info['Nj']
    dxout = info['iDirectionIncrementInDegrees']
    dyout = info['jDirectionIncrementInDegrees']

    lons = np.linspace(llx, llx + (numxgrid - 1) * dxout, numxgrid)
    lats = np.linspace(lly, lly + (numygrid - 1) * dyout, numygrid)

    u = 0
    v = 0

    if args['param_name'] in cf.FIELD_TYPES_3D:
        data = xr.DataArray(np.zeros((info['Nj'], info['Ni'],
                                      len(levRange), len(datelist)),
                                     dtype=np.float64),
                          dims=('lats', 'lons', 'lev', 'dt'),
                          coords={'lats': lats,
                                  'lons': lons,
                                  'lev': levRange,
                                  'dt': datelist})
    elif args['param_name'] == 'ff':
        data = xr.DataArray(np.zeros((info['Nj'], info['Ni'],
                                      len(levRange), len(datelist)),
                                     dtype=np.float64),
                          dims=('lats', 'lons', 'lev', 'dt'),
                          coords={'lats': lats,
                                  'lons': lons,
                                  'lev': levRange,
                                  'dt': datelist})
        u = xr.DataArray(np.zeros((info['Nj'], info['Ni'],
                                   len(levRange), len(datelist)),
                                  dtype=np.float64),
                          dims=('lats', 'lons', 'lev', 'dt'),
                          coords={'lats': lats,
                                  'lons': lons,
                                  'lev': levRange,
                                  'dt': datelist})
        v = xr.DataArray(np.zeros((info['Nj'], info['Ni'],
                                   len(levRange), len(datelist)),
                                  dtype=np.float64),
                          dims=('lats', 'lons', 'lev', 'dt'),
                          coords={'lats': lats,
                                  'lons': lons,
                                  'lev': levRange,
                                  'dt': datelist})
    elif args['param_name'] in cf.FIELD_TYPES_2D:
        data = xr.DataArray(np.zeros((info['Nj'], info['Ni'],
                                      len(datelist)), dtype=np.float64),
                          dims=('lats', 'lons', 'dt'),
                          coords={'lats': lats,
                                  'lons': lons,
                                  'dt': datelist})
    else:
        pass

    return data, u, v, levRange


def read_ff_data(args, filelist, datelist, domain, info,
                 matchLevToZ, verb=False):
    '''Reads u and v fields and calculates the horizontal wind speed.
    Plotting is done for each level over the selected domain.

    Parameters
    ----------
    args : dict-like
        Contains the parameters of the command line call.

    filelist: list of strings
        A list of all files matching the pattern of matchingstring.

    datelist : list of datetime
       A list of all dates to be read.

    domain : list of float
        Contains the lower left and upper right corners of latitude and
        longitude. E.g. llx, lly, urx, ury

    info : dict-like
        Contains some basic information of dataset.
        Ni, Nj, latitudeOfFirstGridPointInDegrees,
        longitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees,
        longitudeOfLastGridPointInDegrees, jDirectionIncrementInDegrees,
        iDirectionIncrementInDegrees, missingValue

    matchLevToZ : dict-like
        Assignment of height levels to height in meter.

    verb : logical
        Decides if additional information is printed.

    Return
    ------

    '''

    ff, u, v, levRange = decide_on_datafield(args, info, datelist, verb)

    for file in filelist:

        f = open(args['input_path'] + file)
        if verb:
            print("Opening file for reading data --- %s" % file)

        # iterate through all grib messages
        while True:
            # Load in memory a grib message from a file.
            gid = codes_grib_new_from_file(f)
            if not gid:
                        break

            # Get the value of a key in a grib message.
            paramId = codes_get(gid, "paramId")
            param = codes_get(gid, "shortName")

            if param == 'u' or param == 'v':
                stepRange = codes_get(gid,"stepRange")
                dataDate = codes_get(gid,"dataDate")
                dataTime = codes_get(gid,"dataTime")
                level = codes_get(gid,"level")

                # create timestamp
                time_hrs = "{0:0>4}".format(int(dataTime) + int(stepRange))
                dtime = datetime.strptime(str(dataDate) + str(time_hrs),
                                          "%Y%m%d%H%M")

                if paramId == 131 and level in levRange:
                    u.loc[:, :, level, dtime] = \
                        codes_get_values(gid).reshape(info['Nj'],
                                                      info['Ni'])
                elif paramId == 132 and level in levRange:
                    v.loc[:, :, level, dtime] = \
                        codes_get_values(gid).reshape(info['Nj'],
                                                      info['Ni'])
                else:
                    pass

            # Free the memory for the message referred as gid.
            codes_release(gid)

        # --- close file ---
        f.close()

    # calculate wind vectors
    ff = np.sqrt(u * u + v * v)

    for l in levRange:
        for t in datelist:
            timetitle = t.strftime(cf.DATE_FORMAT_OUT)
            timestr = t.strftime('%Y%m%d%H')
            plotting.plot_ff_field(args,
                                   u.loc[:, :, l, t],
                                   v.loc[:, :, l, t],
                                   ff.loc[:, :, l, t],
                                   domain, timetitle, timestr, l,
                                   info, matchLevToZ)
    return


def read_3D_data(args, filelist, datelist, domain, info,
                 matchLevToZ, verb=False):
    '''Reads the data field of the selected param and selected height levels.
    Plotting is done for each level over the selected domain.

    Parameters
    ----------
    args : dict-like
        Contains the parameters of the command line call.

    filelist: list of strings
        A list of all files matching the pattern of matchingstring.

    datelist : list of datetime
       A list of all dates to be read.

    domain : list of float
        Contains the lower left and upper right corners of latitude and
        longitude. E.g. llx, lly, urx, ury

    info : dict-like
        Contains some basic information of dataset.
        Ni, Nj, latitudeOfFirstGridPointInDegrees,
        longitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees,
        longitudeOfLastGridPointInDegrees, jDirectionIncrementInDegrees,
        iDirectionIncrementInDegrees, missingValue

    matchLevToZ : dict-like
        Assignment of height levels to height in meter.

    verb : logical
        Decides if additional information is printed.

    Return
    ------

    '''

    data, _, _, levRange = decide_on_datafield(args, info, datelist, verb)

    for file in filelist:

        f = open(args['input_path']+file)
        print( "Opening file for reading data --- %s" % file)

        # iterate through all grib messages
        while True:
            # Load in memory a grib message from a file.
            gid = codes_grib_new_from_file(f)
            if not gid: break

            # Get the value of a key in a grib message.
            paramId = codes_get(gid, "paramId")
            param = codes_get(gid, "shortName")

            if args['param_name'] == param:
                stepRange = codes_get(gid,"stepRange")
                dataDate = codes_get(gid,"dataDate")
                dataTime = codes_get(gid,"dataTime")
                level = codes_get(gid,"level")

                # create timestamp
                time_hrs = "{0:0>4}".format(int(dataTime) + int(stepRange))
                dtime = datetime.strptime(str(dataDate) + str(time_hrs),
                                          "%Y%m%d%H%M")

                if level in levRange:
                    data.loc[:, :, level, dtime] = \
                        codes_get_values(gid).reshape(info['Nj'], info['Ni'])

            # Free the memory for the message referred as gid.
            codes_release(gid)

        # --- close file ---
        f.close()

# vertical integration    eg QUICKLOOK
#    if z0 < z1:  # we want integral value over more vertical levels, get integrated column [kq,Bq]/m2
#        data_aux = numpy.zeros(data0.shape[:2])
#        heights = header['outheight']
#        for zindex in range(z0, z1+1):
#            data_aux += data0[:,:,zindex]*float(heights[zindex])

    # first get overall min and max vals
    min_val = data.min(dim=['lev', 'lats', 'lons', 'dt'])
    max_val = data.max(dim=['lev', 'lats', 'lons', 'dt'])

    # plot the data
    if cf.PLOT_TYPE in [1, 2, 3]:
        # plot 2D BASEMAP
        for l in levRange:
            for t in datelist:
                timetitle = t.strftime(cf.DATE_FORMAT_OUT)
                timestr = t.strftime('%Y%m%d%H')
                plotting.plot_3D_field(args, data.loc[:, :, l, t], domain,
                                       timetitle, timestr, l,
                                       info, matchLevToZ,
                                       min_val, max_val)
    elif cf.PLOT_TYPE == 4:
        # plot cross section
        for t in datelist:
            timetitle = t.strftime(cf.DATE_FORMAT_OUT)
            timestr = t.strftime('%Y%m%d%H')
            # domain = [llx, lly, urx, ury]
            plotting.plot_cross_section(args,
                                        data.loc[domain[1]:domain[3],
                                                 domain[0]:domain[2],
                                                 :, t],
                                        domain, timetitle, timestr,
                                        info, matchLevToZ,
                                        min_val, max_val)

    return


def read_2D_data(args, filelist, datelist, domain, info, verb=False):
    '''Reads the data field of the selected parameter.
    Plotting is done for the selected domain.

    Parameters
    ----------
    args : dict-like
        Contains the parameters of the command line call.

    filelist: list of strings
        A list of all files matching the pattern of matchingstring.

    datelist : list of datetime
       A list of all dates to be read.

    domain : list of float
        Contains the lower left and upper right corners of latitude and
        longitude. E.g. llx, lly, urx, ury

    info : dict-like
        Contains some basic information of dataset.
        Ni, Nj, latitudeOfFirstGridPointInDegrees,
        longitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees,
        longitudeOfLastGridPointInDegrees, jDirectionIncrementInDegrees,
        iDirectionIncrementInDegrees, missingValue

    verb : logical
        Decides if additional information is printed.

    Return
    ------

    '''

    data, _, _, _ = decide_on_datafield(args, info, datelist, verb)

    for file in filelist:

        f = open(args['input_path']+file)
        if verb:
            print( "Opening file for reading data --- %s" % file)

        # iterate through all grib messages
        while True:
            # Load in memory a grib message from a file.
            gid = codes_grib_new_from_file(f)
            if not gid: break

            # Get the value of a key in a grib message.
            paramId = codes_get(gid, "paramId")
            param = codes_get(gid, "shortName")

            if args['param_name'] == param:
                stepRange = codes_get(gid,"stepRange")
                dataDate = codes_get(gid,"dataDate")
                dataTime = codes_get(gid,"dataTime")

                time_hrs = "{0:0>4}".format(int(dataTime)+int(stepRange))
                dtime = datetime.strptime(str(dataDate) + str(time_hrs),
                                          "%Y%m%d%H%M")

                data.loc[:, :, dtime] = \
                    codes_get_values(gid).reshape(info['Nj'], info['Ni'])

            # Free the memory for the message referred as gid.
            codes_release(gid)

        # --- close file ---
        f.close()

    # first get overall min and max vals
    min_val = data.min(dim=['lats', 'lons', 'dt'])
    max_val = data.max(dim=['lats', 'lons', 'dt'])

    for t in datelist:
        timetitle = t.strftime(cf.DATE_FORMAT_OUT)
        timestr = t.strftime('%Y%m%d%H')

        plotting.plot_2D_field(args, data.loc[:, :, t], domain,
                               timetitle, timestr,
                               info, min_val, max_val)

    return
