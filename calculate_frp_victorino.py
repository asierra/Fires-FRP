#!/home/queimadas/miniconda3/envs/qmdpy38/bin/python

#import gdal as gdal
import numpy as np
import pandas as pd
import os
import sys
import re
import glob
import math
from datetime import datetime, timedelta
import netCDF4 as nc
#from pyorbital import astronomy
import numba
from scipy import stats
from pyproj import Proj

#from pyproj import Proj, transform

# Central Wavelength (um)
# https://www.goes-r.gov/education/ABI-bands-quick-info.html
WAVELEN_B1 = 0.47 # Blue - Visible
WAVELEN_B2 = 0.64 # Red  - Visible - B1:NOAA
WAVELEN_B3 = 0.86 # Veggie - Visible - B2:NOAA
WAVELEN_B5 = 1.61 # Snow/Ice - NearIR - B3A:NOAA
WAVELEN_B6 = 2.26 # Cloud Particle Size  - NearIR
WAVELEN_B7 = 3.9 # Shortwave Window - IR with reflected daytime component
WAVELEN_B10 = 7.3 # Lower-level Water Vapor - IR
WAVELEN_B14 = 11.2 # IR Longwave Window  - IR
WAVELEN_B15 = 12.3 # "Dirty" Longwave Window - IR


DEBUG=False
DIM=5424
NCDFDIR="/home/queimadas/goes16"
ANGLDIR="/dados/goes16/sat_angles"
DIROUT="/dados/goes16/data/hotspot/2023"
#DIROUT="/mnt/g/pruebas_lanot/qmdmap"

@numba.jit(nopython=True)
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

@numba.jit(forceobj=True, parallel=True)
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
    
@numba.jit(nopython=True)
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
    #ifov = 2.0 # urad ?? em km

    difLat = math.radians(math.fabs( lat0 - lattd ))

    av = math.sin(math.radians(180 - satz)) * (R / (R + h))
    ac = satz - math.degrees(math.asin(av))
    d = R * math.sin(math.radians(ac)) / math.sin(av)

    pixsz_y = d * ifov / (h * math.cos(difLat))
    pixsz_x = pixsz_y * math.cos(difLat) / math.sin(math.radians(90 - satz))

    res_y = 2.0 * math.atan( (math.degrees(pixsz_y) / (2.0*R)) )
    res_x = (2.0 * math.atan( (math.degrees(pixsz_x) / (2.0*R)) )) / math.cos(math.radians(lattd))

    return pixsz_x, pixsz_y, res_x, res_y


def get_time_of_line(ncfile):
    """
    """

    #DIM = 5424
    dt_epoch = datetime(2000, 1, 1, 12, 00, 00)

    ds = nc.Dataset(ncfile, 'r')
    tb = ds.variables['time_bounds']
    #t = ds.variables['t']
    l0 = float(tb[0])
    lf = float(tb[1])
    #print l0, lf
    fstln = dt_epoch + timedelta(seconds=l0)
    endln = dt_epoch + timedelta(seconds=lf)

    fsttl = timedelta(minutes=fstln.minute, seconds=fstln.second)
    endtl = timedelta(minutes=endln.minute, seconds=endln.second)
    tmdlt = (endtl - fsttl)/DIM

    return fstln, tmdlt

# Rebin function from
# https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
def rescale( shp, nshp ):
    """
    @shp Numpy vector 2D to reshape
    @nshp New shape value to transform @shp vector
    @return Reshaped vector
    """
    sh = nshp[0], shp.shape[0]//nshp[0], nshp[1], shp.shape[1]//nshp[1]
    return shp.reshape(sh).mean(-1).mean(1)

def nc2bin(ncfile, binfile=None):
    """
    """
    assert os.path.isfile(ncfile)

    ds = nc.Dataset(ncfile, 'r')
    ncdata = None
    if "CMI" in ds.variables:
        ncdata = ds.variables['CMI'][:].data
    else:
        ncdata = ds.variables['Rad'][:].data

    if ncdata.shape[0] > 5424:
        #ncdata = rescale(ncdata, [10848, 10848]) * 100
        ncdata = rescale(ncdata, [5424, 5424])

    if binfile is not None:
        ncdata.astype('float32').tofile(binfile)
        assert os.path.isfile(binfile), "Arquivo binario nao criado %s" % binfile
        print(binfile)

    return ncdata

def nav2bin( ncfile, binprefix=None ):
    """
    """
    assert os.path.isfile(ncfile)

    ds = nc.Dataset(ncfile, 'r')
    assert ds.variables.has_key("Latitude")
    assert ds.variables.has_key("Longitude")

    latdata = ds.variables['Latitude'][:].data
    londata = ds.variables['Longitude'][:].data

    if binprefix is not None:
        latbin = binprefix + "_latitude.bin"
        latdata.astype('float32').tofile(latbin)
        assert os.path.isfile(latbin), "Arquivo binario nao criado %s" % latbin

        lonbin = binprefix + "_longitude.bin"
        londata.astype('float32').tofile(lonbin)
        assert os.path.isfile(lonbin), "Arquivo binario nao criado %s" % lonbin

        print(latbin, lonbin)

    return latdata, londata

def get_ncfile_sdt(fname):
    """
    """
    #OR_ABI-L1b-RadF-M6C02_G16_s20192311430197_e20192311439505_c20192311439551.nc
    fname_patt='OR_ABI-(L1b|L2)-(RadF|MCMIPF|CalF)-(M6C|M4C|M3|M3C)(\d{2})_G16_s(.*)_e.*_c.*nc'
    m = re.search(fname_patt, os.path.basename(fname) )
    assert m != None
    assert len(m.groups()) == 5

    sdt = m.group(5)
    dtobj = datetime.strptime(sdt, "%Y%j%H%M%S%f")
    return dtobj

def get_channel_fnames(dirin, ch, dtobj, sdt=None):
    """
    """
    abi = "ABI-L1b-RadF"
    if ch >= 7:
        abi = "ABI-L2-CMIPF"

     # ${dirin}/2018/06/01/ABI-L1b-RadF/11/C02/OR_ABI-L1b-RadF-
     # M3C02_G16_s20181521115477_e20181521126244_c20181521126281.nc
    if sdt == None:
        sdt = dtobj.strftime("%Y%j%H")
        
    pth = "%s/%02d/%02d/%02d/%s/%02d/C%02d/OR_%s-M*C%02d_G16_s%s*.nc" % \
        (dirin, dtobj.year, dtobj.month, dtobj.day,
         abi, dtobj.hour, ch, abi, ch, sdt)

    chfnames = glob.glob( pth )
    #print(chfnames, pth)

    #assert len(chfnames) >= 1
    if len(chfnames) == 0:
        return None

    return chfnames

def get_angle_fname(dirin, angle, sdt):
    """
    """
    patt = "OR_ABI-L1b-CalF-M*C07_G16_s"
    pth = "%s/%s%s_e*_c*_%s_not.bin" % (dirin, patt, sdt, angle)
    afname = glob.glob( pth )
    assert len(afname) == 1

    return afname[0]

def get_navf_fname(dirin):
    """
    """
    patt = "CG_ABI-L2-NAVF-M3_G16"
    dirin = "/mnt/g/pruebas_lanot/20181204"
    pth = "%s/navf_from_csppgeo/%s*.nc" % (dirin, patt)

    afname = glob.glob( pth )
    assert len(afname) == 1
    return afname[0]

def get_jdate(dtobj):
    """
    https://github.com/pytroll/pyorbital/blob/master/pyorbital/astronomy.py
    """
    diffdt = np.datetime64(dtobj) - np.datetime64('2000-01-01T12:00')
    days = diffdt/np.timedelta64(1, 'D')

    return days / 36525.0

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
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

@numba.jit(nopython=True, parallel=True)
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

@numba.jit(nopython=True, parallel=True)
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


@numba.jit(nopython=True, parallel=True)
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


def proc_qmdmap(dtobj, ch02f, ch03f, ch07f, ch14f, ch06f, ch10f):
    """
    """
    lattd = np.fromfile(ANGLDIR + "/navf_latitude.bin", dtype='float32')
    lontd = np.fromfile(ANGLDIR + "/navf_longitude.bin", dtype='float32')
    # Workaround for the testing phase of the satellite at 2017
    ds = nc.Dataset(ch07f, 'r')
    lg0 = ds.variables['goes_imager_projection'].longitude_of_projection_origin
    if lg0 == -89.5:
            lontd = lontd - 14.5
            
    sat_elev = np.fromfile(ANGLDIR + "/sat_elev.bin", dtype='int16')
    sat_az = np.fromfile(ANGLDIR + "/sat_az.bin", dtype='int16')
    
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

    ch02 = None
    ch03 = None
    ch07 = None
    ch14 = None
    ch06 = None
    ch10 = None

    sdt = dtobj.strftime('%Y%m%d%H%M')
    ch02 = nc2bin(ch02f)
    ch03 = nc2bin(ch03f)
    ch07 = nc2bin(ch07f)
    ch14 = nc2bin(ch14f)
    ch10 = nc2bin(ch10f)
                
    msk_sun = np.where( ( (relaz >= 165.) & (relaz <= 195.) ) | (sunr <= 20.),
                        0, 1 )
    tmfst, tmdlt = get_time_of_line(ch07f)

    gProj = Proj("+proj=geos +h=35786023.000000 +a=6378137.000000 +b=6356752.314140 +f=0.0033528106811935606650 +lat_0=0.000000 +lon_0=-75.000000 +sweep=x +no_defs")

    lattd.resize(5424,5424)
    lontd.resize(5424,5424)
    lonlat = []
    
    qmd14 = np.where( (ch14 >= 200) & (ch14 < 450), 1, 0 )
    
    ch07_edge = np.where(sunr < 3, 26.8, 26.8 * (sunr**0.12))
    ch07_edge_day = np.where((sunz < 100) & (ch07 > ch07_edge + 273) & (ch07 < 450), 1, 0)
    ch07_edge_night = np.where((sunz >= 100) & (ch07 >= 301) & (ch07 < 450), 1, 0)
    qmd07 = np.where(sunz >= 100, ch07_edge_night, ch07_edge_day)
    
    chdif = ch07 - ch14
    chdif_edge_day = np.where((sunz < 100) & (chdif > 17), 1, 0)
    chdif_edge_night = np.where((sunz >= 100) & (chdif > 17), 1, 0)
    qmddif = np.where(sunz > 90, chdif_edge_night, chdif_edge_day)
    
    qmap = qmd07 * qmd14 * qmddif * msk_sun
    idx=np.where( qmap == 1, )
    
    for k in range(len(idx[0])):
        i, j = idx[0][k], idx[1][k]
        lotd, latd = lontd[i][j], lattd[i][j]
        xgeos, ygeos = gProj(lotd, latd, inverse=False)
        
        stz = satz[i][j]

        tm = tmfst + (tmdlt * (i-1) )

        szx, szy, resx, resy = compute_pixel_size( latd, stz )

        bkvalue, pvalue = compute_avg_background(i, j, ch07)
        frp = compute_frp(szx, szy, pvalue, bkvalue)


app=sys.argv[0]
if len(sys.argv) < 3:
    print("USAGE: %s 20180630 2100" % app)
    exit(0)

date=sys.argv[1]
hour=sys.argv[2]


saveband=False
if len(sys.argv) == 5:
    saveband=True

dtobj = datetime.strptime("%s%s" % (date, hour), "%Y%m%d%H%M")

sdt = dtobj.strftime("%Y%j%H%M")
ch02fs = get_channel_fnames(NCDFDIR, 2, dtobj, sdt)

channels = [3, 7, 14, 10]
ch02fs.sort()
for ch02f in ch02fs:
    dtobj = get_ncfile_sdt(ch02f)
    print(dtobj.isoformat())
    sdt = dtobj.strftime("%Y%j%H%M")

    fnames = []
    for i in channels:
        fname = get_channel_fnames(NCDFDIR, i, dtobj, sdt)
        if fname is None:
            break
        else:
            fnames.append(fname[0])
    if len(fnames) == 4:
        proc_qmdmap(dtobj, ch02f, fnames[0], fnames[1], fnames[2], None, fnames[3])
exit(0)
