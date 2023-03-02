#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Calcula el Local Zenith Angle del GOES 16 para cualquier punto en el
# disco completo.

# Version Wed 01 Mar 2023 10:25:05 PM UTC

# Alejandro Aguilar Sierra, asierra@unam.mx

import math

def goes16_zenith_angle(lat, lon):
    lat_g16, lon_g16 = 0.0, -75.0
    H = 42164.16
    r_eq = 6378.137

    rlat_g16, rlon_g16 = math.radians(lat_g16), math.radians(lon_g16)
    rlat, rlon = math.radians(lat), math.radians(lon)
    beta = math.acos(math.cos(rlat - rlat_g16) * math.cos(rlon - rlon_g16))
    sqarg = H*H + r_eq*r_eq - 2*H*r_eq*math.cos(beta)
    if sqarg >= 0:
        denom = math.sqrt(sqarg)
    else:
        print(beta, sqarg, math.cos(rlat - rlat_g16) ,math.cos(rlon - rlon_g16))
        return -1
    tmp = H*math.sin(beta)/denom
    if abs(tmp) <= 1:
        local_zenith_angle = math.asin(tmp)
    else:
        return -1.0

    return math.degrees(local_zenith_angle)

print(goes16_zenith_angle(0.0, -75.0))
print(goes16_zenith_angle(20.0, -75.0))
print(goes16_zenith_angle(40.0, -75.0))
print(goes16_zenith_angle(60.0, -75.0))
    
    
