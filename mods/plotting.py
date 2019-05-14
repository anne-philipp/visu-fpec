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
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm, Normalize

import __config as cf


font = {'family': 'monospace', 'size': cf.FONT_SIZE}
matplotlib.rcParams['xtick.major.pad'] = '20'

matplotlib.rc('font', **font)


#*******************************************************************************
# FUNCTIONS
#*******************************************************************************


def plot_ff_field(args, u, v, ff, domain, time_title, timestr,
                  lev, info, matchLevToZ):
    '''Plots a vector field with matplotlibs quiver.

    Parameters
    ----------
    args : dict-like
        Contains the parameters of the command line call.

    u : DataArray
        2D u-wind field

    v : DataArray
        2D v-wind field

    ff : DataArray
        2D wind direction and speed information

    domain : list of float
        Contains the lower left and upper right corners of latitude and
        longitude. E.g. llx, lly, urx, ury

    time_title : str
        Time information as string for the plot title.

    timestr : str
        Time information as string for plot axes.

    lev : int
        Height level of data field from ECMWF.

    info : dict-like
        Contains some basic information of dataset.
        Ni, Nj, latitudeOfFirstGridPointInDegrees,
        longitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees,
        longitudeOfLastGridPointInDegrees, jDirectionIncrementInDegrees,
        iDirectionIncrementInDegrees, missingValue

    matchLevToZ : dict-like
        Assignment of height levels to height in meter.

    Return
    ------

    '''
    opath = args['output_path']

    llx, lly, urx, ury = domain

    dxout = info['iDirectionIncrementInDegrees']
    dyout = info['jDirectionIncrementInDegrees']
    numxgrid = (urx - llx) / dxout + 1
    numygrid = (ury - lly) / dyout + 1

    lons0 = np.linspace(llx, llx + (numxgrid - 1) * dxout, numxgrid)
    lats0 = np.linspace(lly, lly + (numygrid - 1) * dyout, numygrid)

    fig = plt.figure(figsize=(cf.FIG_X, cf.FIG_Y))
    fig.set_facecolor('white')

    m = Basemap(projection=cf.PROJECTION,
                llcrnrlat=lly, urcrnrlat=ury,
                llcrnrlon=llx, urcrnrlon=urx,
                resolution=cf.BASEMAP_RESOLUTION,
                area_thresh=cf.BASEMAP_AREA_THR)

    lons, lats = np.meshgrid(lons0, lats0)

    # now we create a mesh for plotting data
    x, y = m(lons, lats)

    # select correct domain of full data field
    u = np.flipud(u.loc[lly:ury, llx:urx])
    v = np.flipud(v.loc[lly:ury, llx:urx])
    ff = np.flipud(ff.loc[lly:ury, llx:urx])

    yy = np.arange(0, len(lats0), cf.ARROW_SCALE)
    xx = np.arange(0, len(lons0), cf.ARROW_SCALE)

    points = np.meshgrid(yy, xx)

    Q = m.quiver(x[points], y[points], u[points], v[points], ff[points],
                 cmap=cf.C_MAP, latlon=True, scale=cf.MAX_WIND_SPEED,
                 scale_units='inches')

    cb=m.colorbar(Q, location="right")
    plt.clim(0, cf.MAX_WIND_SPEED)

    if not args['unitlabel']:
        cb.set_label('wind speed in $(\mathrm{m s^{-2}})$', size=cf.FONT_SIZE)
    else:
        cbar.set_label(args['unitlabel'], size=cf.FONT_SIZE)

    if not args['title']:
        stitle = time_title + '\nwind vector in ' + \
            str(matchLevToZ[lev]) + ' m'
    else:
        stitle = time_title + '\n' + args['title'] + \
            ' in ' + str(matchLevToZ[lev]) + ' m'
    plt.title(stitle, size=cf.FONT_SIZE)

    thickline = np.arange(lly, ury + 1, cf.MAJOR_PAR_STEP)
    thinline = np.arange(lly, ury + 1, cf.MINOR_PAR_STEP)
    m.drawparallels(thickline, color='gray', dashes=[1, 1],
                    linewidth=0.5, labels=[1, 0, 0, 0], xoffset=1.)
    m.drawparallels(np.setdiff1d(thinline, thickline), color='lightgray',
                    dashes=[1, 1], linewidth=0.5, labels=[0, 0, 0, 0])

    thickline = np.arange(llx, urx + 1, cf.MAJOR_MER_STEP)
    thinline = np.arange(llx, urx + 1, cf.MINOR_MER_STEP)
    m.drawmeridians(thickline, color='gray', dashes=[1, 1],
                    linewidth=0.5, labels=[0, 0, 0, 1], yoffset=1.)
    m.drawmeridians(np.setdiff1d(thinline, thickline), color='lightgray',
                    dashes=[1, 1], linewidth=0.5, labels=[0, 0, 0, 0])

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    file_name = opath + '/plot_' + str(args['param_name']) + '_' + \
        str(timestr) + '_' + str('{:02d}'.format(lev))

    if cf.PDFS_FLAG:
        if cf.TIGHT_LAYOUT:
            plt.savefig(file_name + '.pdf', bbox_inches='tight', pad_inches=0,
                        facecolor=fig.get_facecolor(), edgecolor='none',
                        format='pdf')
        else:
            plt.savefig(file_name + '.pdf', edgecolor='none',
                        facecolor=fig.get_facecolor(), format='pdf')

    if cf.PNGS_FLAG:
        if cf.TIGHT_LAYOUT:
            plt.savefig(file_name + '.png', bbox_inches='tight', pad_inches=0,
                        facecolor=fig.get_facecolor(), edgecolor='none',
                        format='png')
        else:
            plt.savefig(file_name + '.png', edgecolor='none',
                        facecolor=fig.get_facecolor(), format='png')


    if cf.SHOW_PLOT:
        plt.show()

    fig.clf()
    plt.close(fig)

    return


def plot_2D_field(args, data, domain, time_title, timestr, info,
                  min_val, max_val):
    '''Plots a 2D field with matplotlibs imshow, pcolormesh or contourf.

    Parameters
    ----------
    args : dict-like
        Contains the parameters of the command line call.

    data : DataArray
        2D  data field of an ECMWF parameter

    domain : list of float
        Contains the lower left and upper right corners of latitude and
        longitude. E.g. llx, lly, urx, ury

    time_title : str
        Time information as string for the plot title.

    timestr : str
        Time information as string for plot axes.

    info : dict-like
        Contains some basic information of dataset.
        Ni, Nj, latitudeOfFirstGridPointInDegrees,
        longitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees,
        longitudeOfLastGridPointInDegrees, jDirectionIncrementInDegrees,
        iDirectionIncrementInDegrees, missingValue

    min_val : float
        Overall minimum value of complete read data.

    max_val : float
        Overall maximum value of complete read data.

    Return
    ------

    '''

    opath = args['output_path']

    llx, lly, urx, ury = domain

    dxout = info['iDirectionIncrementInDegrees']
    dyout = info['jDirectionIncrementInDegrees']
    numxgrid = (urx - llx) / dxout + 1
    numygrid = (ury - lly) / dyout + 1

    lons0 = np.linspace(llx, llx + (numxgrid - 1) * dxout, numxgrid)
    lats0 = np.linspace(lly, lly + (numygrid - 1) * dyout, numygrid)

    fig = plt.figure(figsize=(cf.FIG_X, cf.FIG_Y))
    fig.set_facecolor('white')

    m = Basemap(projection=cf.PROJECTION,
                llcrnrlat=lly, urcrnrlat=ury,
                llcrnrlon=llx, urcrnrlon=urx,
                resolution=cf.BASEMAP_RESOLUTION,
                area_thresh=cf.BASEMAP_AREA_THR)

    lons, lats = np.meshgrid(lons0, lats0)

    # now we create a mesh for plotting data
    x, y = m(lons, lats)

    data = data.loc[lly:ury, llx:urx]
    data = np.ma.array(data, mask=data == 0.)
    cf.C_MAP.set_under(color='white')

    if cf.LOG_FLAG:  # log scale
        if cf.LOG_LEVELS_MAX == - 999:
            max_val_log = math.ceil(math.log10(max_val))

        levels = np.logspace(max_val_log - cf.NUMBER_OF_ORDERS,
                             max_val_log, num=cf.NUMBER_OF_ORDERS + 1,
                             base=10.0)
        norm = LogNorm(vmin=levels[0], vmax=levels[-1])
    else:  # linear scale
        if cf.LIN_LEVELS_MAX == -999:
            max_val_lin = max_val
        if cf.LIN_LEVELS_MIN == -999:
            min_val_lin = min_val

        levels = np.linspace(min_val_lin, max_val_lin, cf.NUMBER_OF_LEVELS)
        norm = Normalize(vmin=levels[0], vmax=levels[-1])

    if cf.PLOT_TYPE == 1:  # imshow
        Q = m.imshow(data,
                     origin='upper',
                     alpha=cf.ALPHA,
                     cmap=cf.C_MAP,
                     norm=norm  # Normalize(vmin=levels[0], vmax=levels[-1])
                     )
    elif cf.PLOT_TYPE == 2:  # pcolormesh
        Q = m.pcolormesh(x, y, np.flipud(data),
                         cmap=cf.C_MAP,
                         alpha=cf.ALPHA,
                         norm=norm  # Normalize(vmin=levels[0], vmax=levels[-1])
                         )
    elif cf.PLOT_TYPE == 3:  # contourf
        Q = m.contourf(x, y, np.flipud(data),
                       latlon=False,
                       cmap=cf.C_MAP,
                       alpha=cf.ALPHA,
                       levels=levels
                       )

    cb = m.colorbar(Q, location="right")

    if not args['unitlabel']:
        cb.set_label(str(args['param_name']), size=cf.FONT_SIZE)
    else:
        cbar.set_label(args['unitlabel'], size=cf.FONT_SIZE)

    if not args['title']:
        stitle = time_title
    else:
        stitle = time_title + ' ' + args['title']
    plt.title(stitle, size=cf.FONT_SIZE)

    thickline = np.arange(lly, ury + 1, cf.MAJOR_PAR_STEP)
    thinline = np.arange(lly, ury + 1, cf.MINOR_PAR_STEP)
    m.drawparallels(thickline, color='gray', dashes=[1, 1],
                    linewidth=0.5, labels=[1, 0, 0, 0], xoffset=1.)
    m.drawparallels(np.setdiff1d(thinline, thickline), color='lightgray',
                    dashes=[1, 1], linewidth=0.5, labels=[0, 0, 0, 0])

    thickline = np.arange(llx, urx + 1, cf.MAJOR_MER_STEP)
    thinline = np.arange(llx, urx + 1, cf.MINOR_MER_STEP)
    m.drawmeridians(thickline, color='gray', dashes=[1, 1],
                    linewidth=0.5, labels=[0, 0, 0, 1], yoffset=1.)
    m.drawmeridians(np.setdiff1d(thinline, thickline), color='lightgray',
                    dashes=[1, 1], linewidth=0.5, labels=[0, 0, 0, 0])

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    # just like in QUICKLOOK

    # in config
    # TEXT_OFFSET = 0.3  # text offset for labeling POIs in degrees, applied in both lan and lot
    # POI_MARKER = 'ro' # marker and color of POI following matplotlib conventions
    #POI_MARKER_SIZE = 8 # size of POI MARKER


    # plotting of receptors:
#    for receptor in receptors:
#        font = {'family': 'serif',
#                'color':  'darkred',
#                'weight': 'bold',
#                'size': 14,
#                }
#        #r_lon = receptor[0]
#        #r_lat = receptor[1]
#        r_lon, r_lat = m(receptor[0], receptor[1])
#        m.plot(r_lon, r_lat, cf.POI_MARKER, markersize=cf.POI_MARKER_SIZE)
#        plt.text(r_lon+cf.TEXT_OFFSET, r_lat+cf.TEXT_OFFSET, receptor[2], fontdict=font)




    file_name=opath + '/plot_' + str(args['param_name']) + '_' + \
        str(timestr)

    if cf.PDFS_FLAG:
        if cf.TIGHT_LAYOUT:
            plt.savefig(file_name + '.pdf', bbox_inches='tight', pad_inches=0,
                        facecolor=fig.get_facecolor(), edgecolor='none',
                        format='pdf')
        else:
            plt.savefig(file_name + '.pdf', edgecolor='none',
                        facecolor=fig.get_facecolor(), format='pdf')

    if cf.PNGS_FLAG:
        if cf.TIGHT_LAYOUT:
            plt.savefig(file_name + '.png', bbox_inches='tight', pad_inches=0,
                        facecolor=fig.get_facecolor(), edgecolor='none',
                        format='png')
        else:
            plt.savefig(file_name + '.png', edgecolor='none',
                        facecolor=fig.get_facecolor(), format='png')


    if cf.SHOW_PLOT:
        plt.show()

    fig.clf()
    plt.close(fig)

    return



def plot_3D_field(args, data, domain, time_title, timestr,
                  lev, info, matchLevToZ, min_val, max_val):
    '''Plots a 2D field with matplotlibs quiver.

    Parameters
    ----------
    args : dict-like
        Contains the parameters of the command line call.

    data : DataArray
        2D  data field of an ECMWF parameter in a specific height level.

    domain : list of float
        Contains the lower left and upper right corners of latitude and
        longitude. E.g. llx, lly, urx, ury

    time_title : str
        Time information as string for the plot title.

    timestr : str
        Time information as string for plot axes.

    lev : int
        Height level of data field from ECMWF.

    info : dict-like
        Contains some basic information of dataset.
        Ni, Nj, latitudeOfFirstGridPointInDegrees,
        longitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees,
        longitudeOfLastGridPointInDegrees, jDirectionIncrementInDegrees,
        iDirectionIncrementInDegrees, missingValue

    matchLevToZ : dict-like
        Assignment of height levels to height in meter.

    min_val : float
        Overall minimum value of complete read data.

    max_val : float
        Overall maximum value of complete read data.

    Return
    ------

    '''

    opath = args['output_path']

    llx, lly, urx, ury = domain

    dxout = info['iDirectionIncrementInDegrees']
    dyout = info['jDirectionIncrementInDegrees']
    numxgrid = (urx - llx) / dxout + 1
    numygrid = (ury - lly) / dyout + 1

    lons0 = np.linspace(llx, llx + (numxgrid - 1) * dxout, numxgrid)
    lats0 = np.linspace(lly, lly + (numygrid - 1) * dyout, numygrid)

    fig = plt.figure(figsize=(cf.FIG_X, cf.FIG_Y))
    fig.set_facecolor('white')

    m = Basemap(projection=cf.PROJECTION,
                llcrnrlat=lly, urcrnrlat=ury,
                llcrnrlon=llx, urcrnrlon=urx,
                resolution=cf.BASEMAP_RESOLUTION,
                area_thresh=cf.BASEMAP_AREA_THR)

    lons, lats = np.meshgrid(lons0, lats0)

    # now we create a mesh for plotting data
    x, y = m(lons, lats)

    data = data.loc[lly:ury, llx:urx]

    if cf.LOG_FLAG:  # log scale
        if cf.LOG_LEVELS_MAX == - 999:
            max_val_log = math.ceil(math.log10(max_val))

        levels = np.logspace(max_val_log - cf.NUMBER_OF_ORDERS,
                             max_val_log, cf.NUMBER_OF_ORDERS + 1)
    else:  # linear scale
        if cf.LIN_LEVELS_MAX == -999:
            max_val_lin = max_val
        if cf.LIN_LEVELS_MIN == -999:
            min_val_lin = min_val

        levels = np.linspace(min_val_lin, max_val_lin, cf.NUMBER_OF_LEVELS)

    if cf.PLOT_TYPE == 1:  # imshow
        Q = m.imshow(data,
                     origin='upper',
                     alpha=cf.ALPHA,
                     cmap=cf.C_MAP,
                     norm=Normalize(vmin=levels[0], vmax=levels[-1])
                     )
    elif cf.PLOT_TYPE == 2:  # pcolormesh
        Q = m.pcolormesh(x, y, np.flipud(data),
                         cmap=cf.C_MAP,
                         alpha=cf.ALPHA,
                         norm=Normalize(vmin=levels[0], vmax=levels[-1])
                         )
    elif cf.PLOT_TYPE == 3:  # contourf
        Q = m.contourf(x, y, np.flipud(data),
                       latlon=False,
                       cmap=cf.C_MAP,
                       alpha=cf.ALPHA,
                       levels=levels
                       )

    cb = m.colorbar(Q, location="right")

    if not args['unitlabel']:
        cb.set_label(args['param_name'], size=cf.FONT_SIZE)
    else:
        cbar.set_label(args['unitlabel'], size=cf.FONT_SIZE)

    if not args['title']:
        stitle = time_title + ' in ' + str(matchLevToZ[lev]) + ' m'
    else:
        stitle = time_title + '\n' + args['title'] + \
            ' in ' + str(matchLevToZ[lev]) + ' m'
    plt.title(stitle, size=cf.FONT_SIZE)

    thickline = np.arange(lly, ury + 1, cf.MAJOR_PAR_STEP)
    thinline = np.arange(lly, ury + 1, cf.MINOR_PAR_STEP)
    m.drawparallels(thickline, color='gray', dashes=[1, 1],
                    linewidth=0.5, labels=[1, 0, 0, 0], xoffset=1.)
    m.drawparallels(np.setdiff1d(thinline, thickline), color='lightgray',
                    dashes=[1, 1], linewidth=0.5, labels=[0, 0, 0, 0])

    thickline = np.arange(llx, urx + 1, cf.MAJOR_MER_STEP)
    thinline = np.arange(llx, urx + 1, cf.MINOR_MER_STEP)
    m.drawmeridians(thickline, color='gray', dashes=[1, 1],
                    linewidth=0.5, labels=[0, 0, 0, 1], yoffset=1.)
    m.drawmeridians(np.setdiff1d(thinline, thickline), color='lightgray',
                    dashes=[1, 1], linewidth=0.5, labels=[0, 0, 0, 0])

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    file_name = opath + '/plot_' + str(args['param_name']) + '_' + \
        str(timestr) + '_' + str('{:02d}'.format(lev))

    if cf.PDFS_FLAG:
        if cf.TIGHT_LAYOUT:
            plt.savefig(file_name + '.pdf', bbox_inches='tight', pad_inches=0,
                        facecolor=fig.get_facecolor(), edgecolor='none',
                        format='pdf')
        else:
            plt.savefig(file_name + '.pdf', edgecolor='none',
                        facecolor=fig.get_facecolor(), format='pdf')

    if cf.PNGS_FLAG:
        if cf.TIGHT_LAYOUT:
            plt.savefig(file_name + '.png', bbox_inches='tight', pad_inches=0,
                        facecolor=fig.get_facecolor(), edgecolor='none',
                        format='png')
        else:
            plt.savefig(file_name + '.png', edgecolor='none',
                        facecolor=fig.get_facecolor(), format='png')


    if cf.SHOW_PLOT:
        plt.show()

    fig.clf()
    plt.close(fig)

    return


def plot_cross_section(args, data, domain, time_title,
                       timestr, info, matchLevToZ, min_val, max_val):
    '''Plots a cross section in height vs horizontal of a specific parameter.

    Parameters
    ----------
    args : dict-like
        Contains the parameters of the command line call.

    data : DataArray
        2D  data field of an ECMWF parameter in a specific height level.

    domain : list of float
        Contains the lower left and upper right corners of latitude and
        longitude. E.g. llx, lly, urx, ury

    time_title : str
        Time information as string for the plot title.

    timestr : str
        Time information as string for plot axes.

    info : dict-like
        Contains some basic information of dataset.
        Ni, Nj, latitudeOfFirstGridPointInDegrees,
        longitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees,
        longitudeOfLastGridPointInDegrees, jDirectionIncrementInDegrees,
        iDirectionIncrementInDegrees, missingValue

    matchLevToZ : dict-like
        Assignment of height levels to height in meter.

    min_val : float
        Overall minimum value of complete read data.

    max_val : float
        Overall maximum value of complete read data.

    Return
    ------

    '''
    opath = args['output_path']

    llx, lly, urx, ury = domain
    dx = info['iDirectionIncrementInDegrees']
    nx = (urx - llx) / dx + 1

    # for plotting we have to correct the longitude
    # from 0 - 360 to -180 to 180
    if llx > 180.:
        llx = llx - 360.
    if urx > 180.:
        urx = urx - 360.

    newlons = np.linspace(llx, llx + (nx - 1) * dx, nx)
    data.coords['lons'] = newlons

    # test if it will be a cross section on a latitude or longitude
    if llx == urx:  # constant longitude
        lon = data.coords['lons'].values[0]
        lat = False
    elif lly == ury:  # constant latitude
        lat = data.coords['lats'].values[0]
        lon = False

    # for plotting get the actual height in m
    # instead of level number at the axis
    heights = []
    for l in data.coords['lev'].values:
        heights.append(float(matchLevToZ[l]))
    data.coords['lev'].values = heights

    # do plotting
    fig = plt.figure(figsize=(cf.FIG_X, cf.FIG_Y))
    fig.set_facecolor('white')

    if lon:
        data[:, 0, :].T.plot.imshow(#cmap=cf.C_MAP,
                                    vmin=min_val, vmax=max_val,
                                    cbar_kwargs={'label': args['param_name']})
        plt.xlabel('Latitude', size=cf.FONT_SIZE)
        plt.title('Vertical cross section at lon=' + str(lon) +
                  '\n' + time_title, size=cf.FONT_SIZE)
        plt.xlim(data.coords['lats'].values[0], data.coords['lats'].values[-1])
        plt.xticks(np.arange(data.coords['lats'].values[0],
                             data.coords['lats'].values[-1] + 1,
                             step=cf.MAJOR_PAR_STEP))
        file_name = opath + '/cross_lat_' + str(args['param_name']) + '_' + \
            str(timestr) + '_' + str(lon)
    elif lat:
        data[0, :, :].T.plot.imshow(#cmap=cf.C_MAP,
                                    vmin=min_val, vmax=max_val,
                                    cbar_kwargs={'label': args['param_name']})
        plt.xlabel('Longitude', size=cf.FONT_SIZE)
        plt.title('Vertical cross section at lat=' + str(lat) +
                  '\n' + time_title, size=cf.FONT_SIZE)
        plt.xlim(data.coords['lons'].values[0], data.coords['lons'].values[-1])
        plt.xticks(np.arange(data.coords['lons'].values[0],
                             data.coords['lons'].values[-1] + 1,
                             step=cf.MAJOR_MER_STEP))

        file_name = opath + '/cross_lon_' + str(args['param_name']) + '_' + \
            str(timestr) + '_' + str(lat)
    else:
        pass


    plt.ylabel('Height (m)')

    if cf.PDFS_FLAG:
        if cf.TIGHT_LAYOUT:
            plt.savefig(file_name + '.pdf', bbox_inches='tight', pad_inches=0,
                        facecolor=fig.get_facecolor(), edgecolor='none',
                        format='pdf')
        else:
            plt.savefig(file_name + '.pdf', edgecolor='none',
                        facecolor=fig.get_facecolor(), format='pdf')

    if cf.PNGS_FLAG:
        if cf.TIGHT_LAYOUT:
            plt.savefig(file_name + '.png', bbox_inches='tight', pad_inches=0,
                        facecolor=fig.get_facecolor(), edgecolor='none',
                        format='png')
        else:
            plt.savefig(file_name + '.png', edgecolor='none',
                        facecolor=fig.get_facecolor(), format='png')


    if cf.SHOW_PLOT:
        plt.show()

    fig.clf()
    plt.close(fig)

    return
