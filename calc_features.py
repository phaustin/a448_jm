import numpy as np
from wrf import getvar, interplevel, ALL_TIMES, extract_times
import xarray as xr
from netCDF4 import Dataset

def calc_SI(t,P,z,LCL,dz):
    # showalter index: T50 - Tp50(from 850)
    lapse_d = 0.0098 # °K /m
    g = 9.8 # m s-2
    Lv = 2.25e6 # J kg-1
    Rsd = 287.0528 # J kg-1 K-1
    Rsw = 461.5 # J kg-1 K-1
    Cp = 1004 # J kg-1 K-1
    e_0 = 611.3 # Pa
    b = 5423 # K
    T0 = 273.15 # K
    epsilon = 0.622 #kg kg-1

    t_500 = interplevel(t,P,50000)
    z_500 = interplevel(z,P,50000)
    z_850nan = interplevel(z,P,85000)
    t_850nan = interplevel(t,P,85000)

    # there are some missing values at the 85 kPa level because it intersects with some high mountains. To fix this, just use the surface value.
    z_850 = np.where(np.isnan(z_850nan), z[0,:,:], z_850nan)
    t_850 = np.where(np.isnan(t_850nan), t[0,:,:], t_850nan)

    # find where LCL is lower than 85 kPa
    low_LCL_mask = np.where(LCL < z_850,1,0)

    # where LCL<z85, the dry temperature difference is 0 and 
    dry_temp_diff = -lapse_d*(LCL-z_850)
    t_diff_LCL = np.where(low_LCL_mask,0, dry_temp_diff)

    es = e_0 * np.exp(b*(1/T0 - 1/t)) # saturated vapour pressure (Pa)
    r = epsilon * es / (P - es) # mixing ratio

    # wet adiabatic lapse rate in °K/m
    lapse_w = g * (1 + Lv*r/(Rsd*t)) / (Cp + Lv**2*r/(Rsw*t**2))

    # makes 3D array of wet lapse rate where LCL < z < z_500 
    lapse_3d_1 = np.where(z >= LCL, lapse_w, 0)
    lapse_3d_2 = np.where(P > 50000,lapse_3d_1,0)

    # integrate vertically
    t_diff_w = np.sum(-lapse_3d_2*dz, axis=0)

    # temp of air parcel lifted to 50000 Pa
    t_parcel_500 = t_850 + t_diff_LCL + t_diff_w 

    # Showalter Index
    SI = t_500 - t_parcel_500

    # clip the values greater than 0 because this indicates stable atmosphere 
    SI_sniped = np.where(SI>6,6,SI)

    return SI_sniped

def calc_LI(t,P,z,LCL,dz):
    # Lifted Index calculation
    lapse_d = 0.0098 # °K /m
    g = 9.8 # m s-2
    Lv = 2.25e6 # J kg-1
    Rsd = 287.0528 # J kg-1 K-1
    Rsw = 461.5 # J kg-1 K-1
    Cp = 1004 # J kg-1 K-1
    e_0 = 611.3 # Pa
    b = 5423 # K
    T0 = 273.15 # K
    epsilon = 0.622 #kg kg-1

    t_0 = t[0,:,:]

    z_500 = interplevel(z,P,50000)
    t_500 = interplevel(t,P,50000)

    t_diff_LCL = - lapse_d * LCL

    es = e_0 * np.exp(b*(1/T0 - 1/t)) # saturated vapour pressure (Pa)
    r = epsilon * es / (P - es) # mixing ratio

    # wet adiabatic lapse rate in °K/m
    lapse_w = g * (1 + Lv*r/(Rsd*t)) / (Cp + Lv**2*r/(Rsw*t**2))

    # makes 3D array of wet lapse rate where LCL < z < z_500 
    lapse_3d_1 = np.where(z >= LCL, lapse_w, 0)
    lapse_3d_2 = np.where(P > 50000,lapse_3d_1,0)

    # integrate vertically
    t_diff_w = np.sum(-lapse_3d_2*dz, axis=0)

    # temp of air parcel lifted to 50000 Pa
    t_parcel_500 = t_0 + t_diff_LCL + t_diff_w 

    # Lifted index
    LI = t_500 - t_parcel_500

    # snip at 6 because values greater than 6 indicate stability
    LI_sniped = np.where(LI>6,6,LI)

    return LI_sniped 

def calc_Kindex(t,P,Td):
    t_500 = interplevel(t,P,50000)
    t_700nan = interplevel(t,P,70000)
    td_700nan = interplevel(Td,P,70000)
    t_850nan = interplevel(t,P,85000)
    td_850nan = interplevel(Td,P,85000)

    # there are some missing values at the 85 kPa level because it intersects with some high mountains. To fix this, just use the surface values where there are nans 
    t_850 = np.where(np.isnan(t_850nan), t[0,:,:], t_850nan)
    td_850 = np.where(np.isnan(td_850nan), Td[0,:,:], td_850nan)
    t_700 = np.where(np.isnan(t_700nan), t[0,:,:], t_700nan)
    td_700 = np.where(np.isnan(td_700nan), Td[0,:,:], td_700nan)

    K = t_850 + td_850 - t_500 - (t_700-td_700)

    return K

def calc_TotalTotal(t,P,Td):
    t_500 = interplevel(t,P,50000)
    t_850nan = interplevel(t,P,85000)
    td_850nan = interplevel(Td,P,85000)

    # there are some missing values at the 85 kPa level because it intersects with some high mountains. To fix this, just use the surface values where there are nans 
    t_850 = np.where(np.isnan(t_850nan), t[0,:,:], t_850nan)
    td_850 = np.where(np.isnan(td_850nan), Td[0,:,:], td_850nan)

    # Total Totals index
    VT = t_850 - t_500
    CT = td_850 - t_500
    TT = VT + CT

    return TT

def calc_SWI(t,P,Td,wspd,wdir):
    t_850nan = interplevel(t,P,85000)
    td_850nan = interplevel(Td,P,85000)
    wspd_850nan = interplevel(wspd,P,85000)
    wdir_850nan = interplevel(wdir,P,85000)
    wspd_500 = interplevel(wspd,P,50000)
    wdir_500 = interplevel(wdir,P,50000)

    # there are some missing values at the 85 kPa level because it intersects with some high mountains. To fix this, just use the surface values where there are nans 
    t_850 = np.where(np.isnan(t_850nan), t[0,:,:], t_850nan)
    td_850 = np.where(np.isnan(td_850nan), Td[0,:,:], td_850nan)
    wspd_850 = np.where(np.isnan(wspd_850nan), wspd[0,:,:], wspd_850nan)
    wdir_850 = np.where(np.isnan(wdir_850nan), wdir[0,:,:], wdir_850nan)

    # Severe weather index (SWI)
    TT = calc_TotalTotal(t,P,Td)

    # first term
    term_11 = 20*(TT - 49)
    term_1 = np.where(term_11<0,0,term_11)

    # second term
    term_22 = 12*(td_850-273)
    term_2 = np.where(term_22<0,0,term_22)

    # Last term
    wdir8 = np.where(wdir_850<130,0,wdir_850)
    wdir85 = np.where(wdir_850>250,0,wdir8)
    wdir5 = np.where(wdir_500<210,0,wdir_500)
    wdir50 = np.where(wdir_500>310,0,wdir5)
    term = np.where((wdir50 - wdir85)<0,0,(wdir50 - wdir85))
    conv = 1.94584 # kts / m -s
    boolwspd = np.logical_and(wspd_850 > 15*conv, wspd_500 > 15*conv)
    wspd_condition = np.where(boolwspd,0,1)

    term_55 = 125*(np.sin(term)+ 0.2)*wspd_condition
    term_5 = np.where(term_55<0,0,term_55)

    SWI = term_1 + term_2 + 2*wspd_850 + wspd_500 + term_5
    return SWI 

def calc_Qs(qi,qs,qg,dz):
    Qi = np.log(np.sum(qi*dz, axis=0)+1)
    Qs = np.log(np.sum(qs*dz, axis=0)+1)
    Qg = np.log(np.sum(qg*dz, axis=0)+1)

    Q = (Qi+Qs)/2
    Q = np.array(Q).flatten()

    return Qi, Qs, Qg, Q

def calc_geop(geop,P):
    g = 9.8 #m/s2
    ds_clim = Dataset('./geop_clim.nc') # geopotential averages over all the training data
    geop50_clim = np.array(ds_clim['geop50'])
    geop70_clim = np.array(ds_clim['geop70'])
    geop100_clim = np.array(ds_clim['geop100'])
    geop_50 = interplevel(geop,P,50000)/g # geopotential height at 50 kPa (m)
    geop_70 = interplevel(geop,P,70000)/g
    geop_100 = geop[0,:,:]/g

    # geopotential anomalies 
    G50 = geop_50 - geop50_clim
    G70 = geop_70 - geop70_clim
    G100 = geop_100 - geop100_clim
    D57 = G50 - G70
    D510 = G50 - G100
    D710 = G70 - G100

    return G50, G70, G100, D57, D510, D710

def calc_time_cyclic(lon,tt_utc):
    # calculate the cyclic time
    # 10 times zone bounds, a standard time-zone is 15° wide (360/24=15°)
    t_zones = np.arange(-180,-25,15)

    # convert to UTC 
    t_zone_offset = np.arange(-11,-1)

    # initialize the 2D time array
    tt = np.zeros_like(lon)
    for i,off in enumerate(t_zone_offset):

        # find the timezone in the lon array 
        condish1 = lon.where(lon>t_zones[i])
        condish2 = condish1.where(lon<t_zones[i+1])

        # calculate the offset time
        local_time = tt_utc + off +8
        if local_time < 0:
            # make sure local time isnt negative 
            local_time += 24

        if local_time >= 24:
            local_time -= 24

        tt = np.where(condish2<0,tt+local_time,tt)

        tt_cyclic = 1-np.sin(tt*np.pi/24)

    return tt_cyclic

def calc_wmax(w):
    wmax = np.max(w,axis=0)
    log_wmax = np.log(wmax+1)
    clip_wmax = np.where(log_wmax>0.6,0.6,wmax)

    return clip_wmax