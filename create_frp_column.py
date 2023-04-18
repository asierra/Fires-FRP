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
from osgeo import gdal
from datetime import datetime

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

def nu_get_sun_ecliptic_longitude(jdate):
    """
    Ecliptic longitude of the sun at jdate-time
    https://github.com/pytroll/pyorbital/blob/master/pyorbital/astronomy.py
    """
    m_a = np.deg2rad(357.52910 + 35999.05030 * jdate -
                     0.0001559 * jdate * jdate -
                     0.00000048 * jdate * jdate * jdate)

    # mean longitude, deg
    l_0 = 280.46645 + 36000.76983 * jdate + 0.0003032 * jdate * jdate
    d_l = (1.914600 - 0.004817 * jdate - 0.000014 * jdate * jdate)
    d_l *= np.sin(m_a)
    d_l += (0.019993 - 0.000101 * jdate) * np.sin(2 * m_a)
    d_l += 0.000290 * np.sin(3 * m_a)

    # true longitude, deg
    l__ = l_0 + d_l
    return np.deg2rad(l__)

def nu_sun_ra_dec(jdate):
    """Right ascension and declination of the sun at *utc_time*.
    https://github.com/pytroll/pyorbital/blob/master/pyorbital/astronomy.py
    """
    eps = np.deg2rad(23.0 + 26.0 / 60.0 + 21.448 / 3600.0 -
                     (46.8150 * jdate + 0.00059 * jdate * jdate -
                      0.001813 * jdate * jdate * jdate) / 3600)

    eclon = nu_get_sun_ecliptic_longitude(jdate)
    x__ = np.cos(eclon)
    y__ = np.cos(eps) * np.sin(eclon)
    z__ = np.sin(eps) * np.sin(eclon)
    r__ = np.sqrt(1.0 - z__ * z__)

    # sun declination
    declination = np.arctan2(z__, r__)

    # right ascension
    right_ascension = 2 * np.arctan2(y__, (x__ + r__))
    return right_ascension, declination

def nu_local_hour_angle(jdate, longitude, right_ascension):
    """Hour angle at *utc_time* for the given *longitude* and
    *right_ascension*
    longitude in radians
    https://github.com/pytroll/pyorbital/blob/master/pyorbital/astronomy.py
    """
    theta = 67310.54841 + jdate * (876600 * 3600 + 8640184.812866 + jdate *
                                   (0.093104 - jdate * 6.2 * 10e-6))

    ### Greenwich mean sidereal utc_time, in radians.
    gmst = np.deg2rad(theta / 240.0) % (2 * np.pi)
    ### Local mean sidereal time, In radians.
    lmst = gmst + longitude

    return lmst - right_ascension

def get_jdate(dtobj):
    """
    https://github.com/pytroll/pyorbital/blob/master/pyorbital/astronomy.py
    """
    diffdt = np.datetime64(dtobj) - np.datetime64('2000-01-01T12:00')
    days = diffdt/np.timedelta64(1, 'D')

    return days / 36525.0

def get_alt_az(jdate, lon, lat):
    """Return sun altitude and azimuth from *utc_time*, *lon*, and *lat*.
    lon,lat in degrees
    https://github.com/pytroll/pyorbital/blob/master/pyorbital/astronomy.py
    """
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    ra, dec = nu_sun_ra_dec(jdate)
    h = nu_local_hour_angle(jdate, lon, ra)
    return (np.arcsin(np.sin(lat) * np.sin(dec) +
                      np.cos(lat) * np.cos(dec) * np.cos(h)),
            np.arctan2(-np.sin(h),
                       (np.cos(lat) * np.tan(dec) - np.sin(lat) * np.cos(h))))

def get_sunr_angle(lat, lon, jdate, sat_elev, sat_az):
    """
    The worst time with other procs
    real    2m35.487s
    user    0m39.688s
    sys     0m38.297s
    """
    sun_elev_rad, sun_az_rad = get_alt_az(jdate, lon, lat)
    sun_az_rad = np.where( sun_az_rad < 0, 2 * np.pi + sun_az_rad, sun_az_rad )

    sat_elev_rad = np.deg2rad(sat_elev)
    sat_az_rad = np.deg2rad(sat_az)

    # https://en.wikipedia.org/wiki/Great-circle_distance
    relaz_rad = np.absolute(sat_az_rad - sun_az_rad)

    dltphi = np.cos(sat_elev_rad) * np.cos(relaz_rad)
    sunr = np.rad2deg(np.arccos(dltphi))

    sun_az = np.rad2deg(sun_az_rad)
    sun_elev = np.rad2deg(sun_elev_rad)

    return sunr, sun_elev, sun_az

def compute_stz(dtobj, pathLat, pathLon ,pathC07, pathElev, pathAz):
    """
    Compute the solar zenith angle
    dtobj = datetime.strptime("%s%s" % (date, hour), "%Y%m%d%H%M")
    """
    lattd = np.fromfile(pathLat, dtype='float32')
    lontd = np.fromfile(pathLon, dtype='float32')
    # Workaround for the testing phase of the satellite at 2017
    ds = Dataset(pathC07, 'r')
    lg0 = ds.variables['goes_imager_projection'].longitude_of_projection_origin
    if lg0 == -89.5:
            lontd = lontd - 14.5
            
    sat_elev = np.fromfile( pathElev, dtype='int16')
    sat_az = np.fromfile(pathAz, dtype='int16')
    
    jdate = get_jdate(dtobj)
    sunr, sun_elev, sun_az = get_sunr_angle(lattd, lontd, jdate, sat_elev, sat_az)

    sunz = np.absolute(np.round(sun_elev) - 90)
    satz = np.absolute(sat_elev - 90)
    relaz = np.absolute(sun_az - sat_az)
    relaz.resize(5424,5424)

    sunr.resize(5424,5424)
    sunz.resize(5424,5424)
    sun_az.resize(5424,5424)

    satz = np.absolute(sat_elev - 90.)
    satz.resize(5424,5424)

    return satz

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

    # Datos de temperatura de brillo de la banda 7
    pathInputSatAz = 'data/goes16_local_zenith_angle.tif'
    pathInputCh07_bt = 'data/OR_ABI-L2-CMIPF-M6C07_G16_s20211211940163_e20211211949482_c20211211949541.nc'
    pathInputCh07_bt_tif = 'data/OR_ABI-L2-CMIPF-M6C07_G16_s20211211940163_e20211211949482_c20211211949541.tif'
    pathInputCh07 = 'data/OR_ABI-L1b-RadF-M6C07_G16_s20211211940163_e20211211949482_c20211211949522.nc'
    pathInputCSV = 'data/GIM10_PC_202105011940_muestreo.csv'
    pathOutputCSV = 'data/GIM10_PC_FRP_202105011940.csv'

    # Datos de navegación
    pathLat = 'data/navf_latitude.bin'
    pathLon = 'data/navf_longitude.bin'
    pathElev = 'data/sat_elev.bin'
    pathAz = 'data/sat_az.bin'
    
    # Obtiene el tiempo de la imagen
    print(pathInputCh07.split('/')[-1].split('_')[3])
    dtobj = datetime.strptime(pathInputCh07_bt.split('/')[-1].split('_')[3], 's%Y%j%H%M%S%f')

    # Obtiene satz
    #ds_satz = gdal.Open(pathInputSatAz)
    #satz = ds_satz.ReadAsArray()
    satz = compute_stz(dtobj, pathLat, pathLon, pathInputCh07_bt, pathElev, pathAz)

    # Checar si es nc y hacer lo equivalente
    if '.nc' in pathInputCh07_bt:
        ds = Dataset(pathInputCh07, "r", format="NETCDF4")
        ch07_rad = (ds['Rad'][:].data * ds['Rad'].scale_factor) + ds['Rad'].add_offset
        ds_ch07 = Dataset(pathInputCh07_bt, "r", format="NETCDF4")
        ch07_bt = ds_ch07['CMI'][:].data
        #ch07_bt = (ds_ch07['CMI'][:].data * ds_ch07['CMI'].scale_factor) + ds_ch07['CMI'].add_offset
        #ds_ch07 = gdal.Open(pathInputCh07_bt)
        #ch07_bt = ds_ch07.ReadAsArray()
        ds_ras = rasterio.open(pathInputCh07_bt_tif)
    else:
        ds_ch07 = rasterio.open(pathInputCh07)
        ch07 = ds_ch07.read(1)
        #ds_ch07 = gdal.Open(pathInputCh07)
        #ch07 = ds_ch07.ReadAsArray()

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
                #i, j = coordinates2ij(x, y)
                # Con rasterio
                transform = ds_ras.transform
                i, j = ds_ras.index(x, y)
                print(x,y,i,j,ch07_bt[j,i])
                stz = satz[i,j]
                szx, szy, resx, resy = compute_pixel_size( lat, stz )
                bkvalue, pvalue = compute_avg_background(j, i, ch07_rad, ch07_bt)
                frp = compute_frp(szx, szy, pvalue, bkvalue)
                print('bkvale:',bkvalue, 'pvalue:',pvalue, 'frp:',frp)
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

