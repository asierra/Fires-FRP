import math
import rasterio 

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
    if satz==0 or satz ==180 or satz==360:
        satz = 1
    av = math.sin(math.radians(180 - satz)) * (R / (R + h))
    #print(math.sin(av))
    ac = satz - math.degrees(math.asin(av))
    d = R * math.sin(math.radians(ac)) / math.sin(av)

    pixsz_y = d * ifov / (h * math.cos(difLat))
    pixsz_x = pixsz_y * math.cos(difLat) / math.sin(math.radians(90 - satz))

    res_y = 2.0 * math.atan( (math.degrees(pixsz_y) / (2.0*R)) )
    res_x = (2.0 * math.atan( (math.degrees(pixsz_x) / (2.0*R)) )) / math.cos(math.radians(lattd))

    return pixsz_x, pixsz_y, res_x, res_y   

pathInputSatAz = './sat_angles/sat_az.tif'
pathInputLattd = './sat_angles/lattd.tif' 

ds_satz = rasterio.open(pathInputSatAz)
ds_lattd= rasterio.open(pathInputLattd)

satz = ds_satz.read(1)
lattd = ds_lattd.read(1)

print(ds_satz.width, ds_satz.height)
print(ds_lattd.width, ds_lattd.height)
print(type(satz[0,0]))
for i in range(satz.shape[0]):
    for j in range(satz.shape[1]):

        pixsz_x, pixsz_y, res_x, res_y  = compute_pixel_size(lattd[i,j], satz[i,j])
        print(pixsz_x, pixsz_y, res_x, res_y)
        #print(satz[i,j])
        
