#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Creado a partir del compute_frp_code-vitorino.py

# Version Wed 01 Mar 2023 06:37:14 PM UTC

# Alejandro Aguilar Sierra, asierra@unam.mx
# Uriel Mendoza, urielmendozacastillo@gmail.com

import math
import rasterio
import numpy as np
from scipy import stats
import csv
from netCDF4 import Dataset

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

def rad2bt(rad_arr, wl):
    """
    The Inverse Planck Function
    Convert from spectral radiance and wavelength
    to temperature
    @wl: Central Wavelength (um)
    @rad: Radiance array (mW)
    http://ncc.nesdis.noaa.gov/data/planck.html
    """
    c1 = 1.191042E8 #W m-2 sr-1 um-4
    c2 = 1.4387752E4 #K um

    rad = c2/(wl*np.log((c1)/(wl**5 * rad_arr + 1)))
    return rad

def compute_avg_background(px, py, bnd, bt):
    """
    """
    j = 200
    wd = bt[px-j:px+j+1, py-j:py+j+1]
    if not wd.size:
        return 0., 0.    
    wd_rad = bt2rad(wd, 3.9)
    #wd_rad = bnd[px-j:px+j+1, py-j:py+j+1]
    #print(wd)
    #print(wd_rad)
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

    print("pixsz_x:",pixsz_x)
    print("pixsz_y:",pixsz_y)
    sigma = 5.67E-8 # W m-2 K4
    a = 3.08E-9 # W m-2 sr-1 um-1 K-4
    t1 = sigma/a

    Apix = pixsz_x * pixsz_y # m2 * 10-6
    Lmir = pvalue
    Lmir_bk = bkvalue

    frp = Apix * t1 * (Lmir - Lmir_bk)

    return frp #MW


# Dimensiones y límites de una imagen GOES16 a 2km
width, height = 5424, 5424
ulx, uly, lrx, lry = -5434894.701, 5434894.701, 5434895.218, -5434895.218


# A partir de coordenadas geoestacionarias calcula la posición del pixel
# en la imagen raster.
def coordinates2ij(x, y):
    i = int(width * (x - ulx)/(lrx - ulx))
    j = int(height * (uly - y)/(uly - lry))
    return i, j


if __name__== "__main__":
    pathInputSatAz = 'data/goes16_local_zenith_angle.tif'
    pathInputCh07_bt = 'data/OR_ABI-L2-CMIPF-M6C07_G16_s20211211940163_e20211211949482_c20211211949541.tif'
    pathInputCh07 = 'data/OR_ABI-L1b-RadF-M6C07_G16_s20211211940163_e20211211949482_c20211211949522.nc'
    pathInputCSV = 'data/GIM10_PC_202105011940.csv'
    pathOutputCSV = 'data/GIM10_PC_FRP_202105011940.csv'
    
    ds_satz = rasterio.open(pathInputSatAz)
    satz = ds_satz.read(1)

    # Checar si es nc y hacer lo equivalente
    if '.nc' in pathInputCh07:
        ds = Dataset(pathInputCh07, "r", format="NETCDF4")
        ch07 = (ds['Rad'][:].data * ds['Rad'].scale_factor) + ds['Rad'].add_offset
        ds_ch07 = rasterio.open(pathInputCh07_bt)
        ch07_bt = ds_ch07.read(1)
    else:
        ds_ch07 = rasterio.open(pathInputCh07)
        ch07 = ds_ch07.read(1)

    #if 'L1b' in pathInputCh07:
        #ch07 = bt2rad(ch07, 3.9)
    #    ch07 = rad2bt(ch07, 3.9)
        
    outcsv = []
    with open(pathInputCSV) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            #
            if line_count > 0:
                x = float(row[0])
                y = float(row[1])
                lon = float(row[2])
                lat = float(row[3])
                i, j = coordinates2ij(x, y)
                stz = satz[i,j]
                szx, szy, resx, resy = compute_pixel_size( lat, stz )
                bkvalue, pvalue = compute_avg_background(i, j, ch07, ch07_bt)
                frp = compute_frp(szx, szy, pvalue, bkvalue)
                print('bkvale:',bkvalue)
                print('pvalue:',pvalue)
                print('frp:',frp)
                row.append(frp)
            else:
                row.append('FRP')
            outcsv.append(row)
            line_count += 1
        print(f'Líneas procesadas {line_count}.')

    with open(pathOutputCSV, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in outcsv:
            csv_writer.writerow(row)

