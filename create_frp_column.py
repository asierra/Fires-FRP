#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rasterio
import numpy as np
from scipy import stats
import csv

def compute_pixel_size( lattd, satz, ifov=2.0 ):
    """
    Get pixel size

    ABI scans the Earth via two orthogonal scan mirrors:
    one east-west (EW) and one north-south (NS).
    The EW mirror scans the Earth at 1.4° (optical) per second;
    a single EW scan is called a swath. The  NS  mirror  is
    then  stepped  to  a  new  location  to  begin  another  EW  swath.
    EW IFOV(μrad): 51.5 (ch07)
    NS IFOV(μrad): 47.7 (ch07)
    https://www.goes-r.gov/downloads/resources/documents/GOES-RSeriesDataBook.pdf
    1 km ~= 26 urad
    """
    #long0 = -75.0
    lat0 = 0.0
    h = 35786.0 # km
    R = 6371.0 # km

    difLat = math.radians(math.fabs( lat0 - lattd ))

    if satz <= 0.5:
        satz = 1.0
    av = math.sin(math.radians(180 - satz)) * (R / (R + h))
    ac = satz - math.degrees(math.asin(av))
    d = R * math.sin(math.radians(ac)) / math.sin(av)

    pixsz_y = d * ifov / (h * math.cos(difLat))
    pixsz_x = pixsz_y * math.cos(difLat) / math.sin(math.radians(90 - satz))

    res_y = 2.0 * math.atan( (math.degrees(pixsz_y) / (2.0*R)) )
    res_x = (2.0 * math.atan( (math.degrees(pixsz_x) / (2.0*R)) )) / math.cos(math.radians(lattd))

    return pixsz_x, pixsz_y, res_x, res_y


def bt2rad(bt_arr, wl):
    """
    The Planck Function
    Convert from temperature and wavelength
    to spectral radiance
    @wl: Central Wavelength (um)
    @bt: Brigthness Temperature (k) array
    http://ncc.nesdis.noaa.gov/data/planck.html
    """
    c1 = 1.191042E8 #W m-2 sr-1 um-4
    c2 = 1.4387752E4 #K um

    rad = c1/(wl**5 * (np.exp(c2/(wl * bt_arr)) - 1))
    return rad


def compute_avg_background(px, py, bnd):
    """
    """
    j = 200
    wd = bnd[px-j:px+j+1, py-j:py+j+1]
    if not wd.size:
        return 0., 0.
    
    wd_rad = bt2rad(wd, 3.9)
    zsc = stats.zscore(wd)
    #print(zsc)
    bck_rad = np.where(zsc > -0.5, np.nan, wd_rad)
    return np.nanmedian(bck_rad), wd_rad[j, j]

def compute_frp(pixsz_x, pixsz_y, pvalue, bkvalue):
    """
    @pixsz_x km
    @pixsz_y km
    @pvalue rad
    @bkvalue rad
    """
    sigma = 5.67E-8 # W m-2 K4
    a = 3.08E-9 # W m-2 sr-1 um-1 K-4
    t1 = sigma/a

    Apix = pixsz_x * pixsz_y # m2 * 10-6
    Lmir = pvalue
    Lmir_bk = bkvalue

    frp = Apix * t1 * (Lmir - Lmir_bk)

    return frp #MW


width = 5424
height = 5424
ulx, uly, lrx, lry = -5434894.701, 5434894.701, 5434895.218, -5434895.218


def xy2uv(x, y):
    u = int(width * (x - ulx)/(lrx - ulx))
    v = int(height * (uly - y)/(uly - lry))
    return u, v


pathInputSatAz = './sat_angles/sat_az.tif'
pathInputCh07 = 'OR_ABI-L2-CMIPF-M6C07_G16_s20211211940163_e20211211949482_c20211211949541.tif'

ds_satz = rasterio.open(pathInputSatAz)
satz = ds_satz.read(1)

ds_ch07 = rasterio.open(pathInputCh07)
ch07 = ds_ch07.read(1)

print(ulx, uly, xy2uv(ulx, uly))
print(lrx, lry, xy2uv(lrx, lry))

outfilename = 'GIM10_PC_FRP_202105011940.csv'
outcsv = []
with open('GIM10_PC_202105011940.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count > 0:
            x = float(row[0])
            y = float(row[1])
            lon = float(row[2])
            lat = float(row[3])
            i, j = xy2uv(x, y)
            stz = satz[i,j]
            szx, szy, resx, resy = compute_pixel_size( lat, stz )
            bkvalue, pvalue = compute_avg_background(i, j, ch07)
            frp = compute_frp(szx, szy, pvalue, bkvalue)
            row.append(frp)
            print(frp)
        else:
            row.append('FRP')
        outcsv.append(row)
        line_count += 1
    print(f'Líneas procesadas {line_count}.')


with open(outfilename, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in outcsv:
        csv_writer.writerow(row)

