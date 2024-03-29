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


szx, szy, resx, resy = compute_pixel_size( latd, stz )
bkvalue, pvalue = compute_avg_background(i, j, ch07)
#print(bkvalue, pvalue)
frp = compute_frp(szx, szy, pvalue, bkvalue)
