import os
bridges2_flag = True
cheyenne_flag = False
if bridges2_flag:
    os.chdir('/ocean/projects/ees220005p/janniy/python_fortran_coarse_graining')
else:
    os.chdir('/glade/u/home/janniy/python_fortran_coarse_graining')

# os.chdir('/glade/u/home/janniy/python_fortran_coarse_graining')
import numpy as np
# import advect_scalar3D_no_netcdf
# import advect_scalar3D_no_netcdf3
# import advect2_mom_z_f2py3
import matplotlib.pyplot as plt
import pickle
from netCDF4 import Dataset
import time
# import xarray as xr
import xarray as xr
import math

from scipy.interpolate import interp1d




# functions:
def calc_precip(q, rho, z):
# surface precipitation rate given tendency of specific humidity
    precip = -vertical_integral(q, rho, z)
    return precip

def vertical_integral(data, rho, z):
# vertical integral with respect to sigma
    rho_dz = vertical_diff(rho, z)
    int_data = np.sum(data * rho_dz[:,None], axis=0)
    return int_data

def vertical_diff(rho, z):
    # follow vertical differencing from setgrid.f90 in SAM
    # changed indexing from starting at 1 to 0
    nzm = z.size
    adz = np.zeros(nzm)
    dz = 0.5*(z[0]+z[1])
    adz[0] = 1.
    for k in range(1,nzm-1): # range doesn't include stopping number
        adz[k] = 0.5*(z[k+1]-z[k-1])/dz

    adz[nzm-1] = (z[nzm-1]-z[nzm-2])/dz

    rho_dz = adz*dz*rho

    return rho_dz



def sam_qsat(T,p):
# formulation used in SAM

    esatw = sam_esatw(T) / 100
    esati = sam_esati(T) / 100

    qw = 0.622 * esatw/np.maximum(esatw,p-esatw);
    qi = 0.622 * esati/np.maximum(esati,p-esati);

    a_bg = 1.0/(tbgmax-tbgmin)
    omn = np.maximum(0.0,np.minimum(1.0,(T-tbgmin)*a_bg))
    qsat = omn*qw+(1-omn)*qi

    return qsat


def sam_esatw(T):
# SAM saturation vapor pressure for liquid water
# from sat.f90
# T is in K
# e is converted here to Pa

    a0 = 6.105851
    a1 = 0.4440316
    a2 = 0.1430341e-1
    a3 = 0.2641412e-3
    a4 = 0.2995057e-5
    a5 = 0.2031998e-7
    a6 = 0.6936113e-10
    a7 = 0.2564861e-13
    a8 = -0.3704404e-15

    dt = np.maximum(-80.,T-273.16)

    e = a0 + dt*(a1+dt*(a2+dt*(a3+dt*(a4+dt*(a5+dt*(a6+dt*(a7+a8*dt)))))))

    return e*100


def sam_esati(T):
# SAM saturation vapor pressure for ice
# from sat.f90
# T is in K
# e is converted here to Pa

    a0 = 6.11147274
    a1 = 0.503160820
    a2 = 0.188439774e-1
    a3 = 0.420895665e-3
    a4 = 0.615021634e-5
    a5 = 0.602588177e-7
    a6 = 0.385852041e-9
    a7 = 0.146898966e-11
    a8 = 0.252751365e-14

    dt = np.maximum(-80.,T-273.16)

    e = a0 + dt*(a1+dt*(a2+dt*(a3+dt*(a4+dt*(a5+dt*(a6+dt*(a7+a8*dt)))))))

    return e*100

def sam_qsatw(t,p):
# % t is temperature (K)
# % p is pressure    (mb)
# % from sat.f90

    esat = sam_esatw(t) /100 #(the 100 factor is to be consistant with the matlab)
    q = 0.622 * esat/np.maximum(esat,p-esat)

    return q
#
# def precip_flux_from_tendency(qfall_tend, rho, z):
#    lhs_tend = vertical_integral(t_tend,rho,z)
#
#    a_pr = 1.0/(tprmax-tprmin)
#    omp = np.maximum(0.0,np.minimum(1.0,(tabs-tprmin)*a_pr))
#    fac = (L + Lf*(1.0-omp))/cp
#    # the following is correct since the sgs flux divergence in q_tend integrates to zero
#    if 'qpout' in output_vert_vars:
#         # precip = 0 # No need to correct for precipitation since it is in t_tend
#         rhs_tend = -fac[:, 0] * vertical_integral(q_tend + o_dict['qpout'], rho, z)
#    else:
#         precip = calc_precip(q_tend,rho,z, output_vert_vars, o_dict)
#         rhs_tend = fac[:,1]*precip
#
#    tend_normalized = (lhs_tend-rhs_tend)/vertical_integral(1.0, rho, z)
#
#    return tend_normalized
def sam_qsati(t,p):
        '''% t is temperature (K)
        % p is pressure    (mb)
        % from sat.f90'''

        esat=sam_esati(t) /100 #(the 100 factor is to be consistant with the matlab)
        qtemp = 0.622*esat/(np.maximum(esat,p-esat))
        return qtemp


def sam_dtesati(t):
    '''% t is temperature (K)
    % from sat.f90'''

    a0 = 0.503223089
    a1 = 0.377174432e-1
    a2 = 0.126710138e-2
    a3 = 0.249065913e-4
    a4 = 0.312668753e-6
    a5 = 0.255653718e-8
    a6 = 0.132073448e-10
    a7 = 0.390204672e-13
    a8 = 0.497275778e-16

    dt = max(-80.,t-273.16)

    de = a0 + dt*(a1+dt*(a2+dt*(a3+dt*(a4+dt*(a5+dt*(a6+dt*(a7+a8*dt)))))))
    return de

def sam_dtqsati(t,p):
    '''% t is temperature (K)
    % p is pressure    (mb)
    % from sat.f90'''

    dq=0.622*sam_dtesati(t)/p
    return dq


def sam_dtesatw(t):
    '''% t is temperature (K)
    % from sat.f90'''

    a0 = 0.443956472
    a1 = 0.285976452e-1
    a2 = 0.794747212e-3
    a3 = 0.121167162e-4
    a4 = 0.103167413e-6
    a5 = 0.385208005e-9
    a6 = -0.604119582e-12
    a7 = -0.792933209e-14
    a8 = -0.599634321e-17

    dt = max(-80.,t-273.16)

    de = a0 + dt*(a1+dt*(a2+dt*(a3+dt*(a4+dt*(a5+dt*(a6+dt*(a7+a8*dt)))))))
    return de

def sam_dtqsatw(t,p):
    '''% t is temperature (K)
    % p is pressure    (mb)
    % from sat.f90'''

    dq=0.622*sam_dtesatw(t)/p
    return dq

def esat(T):
    a_bg = 1.0/(tbgmax-tbgmin)
    omn = np.maximum(0.0,np.minimum(1.0,(T-tbgmin)*a_bg))
    return (T>tbgmax)*sam_esatw(T) + (T<tbgmin) * sam_esati(T) + (T<=tbgmax)*(T>=tbgmin)*(omn*sam_esatw(T) + (1-omn)*sam_esati(T))



def squeeze_reshape_var(var):
    var = np.moveaxis(np.squeeze(var),(0,1,2),(2,1,0))
    return var


def reshape_add_bound_dims(var, ext_dim):
    # var = np.moveaxis(np.squeeze(var),(0,1,2),(2,1,0))
    # print(var.shape)
    var_bound = np.zeros((var.shape[0] + ext_dim, var.shape[1] + ext_dim, var.shape[2]))
    var_bound[2:-2, 2:-2, :] = var
    var_bound[0, 2:-2, :] = var[-2, :, :]
    var_bound[1, 2:-2, :] = var[-1, :, :]

    var_bound[-1, 2:-2, :] = var[1, :, :]
    var_bound[-2, 2:-2, :] = var[0, :, :]

    var_bound[:, -1, :] = 0
    var_bound[:, -2, :] = 0

    var_bound[:, 0, :] = 0
    var_bound[:, 1, :] = 0

    return var_bound


def coarse_grain(field, coarseness = 16):
    # Suppose the 2D array is pop_density
    if len(field.shape) == 3:
        temp = field.reshape((field.shape[0] // coarseness, coarseness,
                                    field.shape[1] // coarseness, coarseness,field.shape[2]))
    elif len(field.shape) == 2:
            temp = field.reshape((field.shape[0] // coarseness, coarseness,
                                    field.shape[1] // coarseness, coarseness))

    elif len(field.shape) == 1:
        temp = field.reshape((field.shape[0] // coarseness, coarseness))
    else:
        raise ValueError('dimentions should be 2 or 3 or 1')

    if len(field.shape) == 2 or len(field.shape) == 3:
        field_coarse = np.mean(temp, axis=(1,3))
    else:
        field_coarse = np.mean(temp, axis=(1))
    return field_coarse


# This is a slow coarse graining!# Cannot slice twice...
# https://stackoverflow.com/questions/56328467/numpy-slicing-with-multiple-tuples

# Need to use this to average along one axis and then do the other axis with tuples..

def coarse_grain_x(field, factor=16, x_var=False, therm_var=False, y_var=False, therm_grid=False, cgrid=False, u_grid=False):
    ix = int(field.shape[0] / factor)
    jy = int(field.shape[1] / factor)
    if len(field.shape) == 3:
        temp = field.reshape((field.shape[0], field.shape[1] // factor, factor, field.shape[2]))
        field_coarse_y = np.mean(temp, axis=(2))
        kz = field.shape[2]
        field_coarse = np.zeros((ix, jy, kz))
    if len(field.shape) == 2:
        temp = field.reshape((field.shape[0], field.shape[1] // factor, factor))
        field_coarse_y = np.mean(temp, axis=(2))
        field_coarse = np.zeros((ix, jy))

    weight1 = (factor - 1) / factor
    weight2 = 1 - weight1

    # ix = int(field.shape[0] / factor)
    # jy = int(field.shape[1] / factor)
    # kz = field.shape[2]
    # field_coarse = np.zeros((ix, jy, kz))
    multiple_space = factor

    half_space = int(multiple_space / 2)

    for i in range(ix):
        if x_var:
            if cgrid:  # keep on a C-grid:
                if i == 0:
                    x1 = tuple(range(0, half_space))
                    x2 = tuple(range(field.shape[0] - half_space + 2 - 1, field.shape[0]))
                    x_tot = x1 + x2
                    # i_indices_x = [1:multiple_space/2,size(u,3)-multiple_space./2+2:size(u,3)];
                    x3 = (half_space, field.shape[0] - half_space)
                    # i_indices_x2 = [multiple_space/2+1,size(u,3)-multiple_space./2 + 1];
                else:
                    x1 = (i) * multiple_space + 1 - half_space
                    x2 = (i + 1) * multiple_space - half_space
                    x_tot = tuple(range(x1, x2))
                    # i_indices_x = [(i-1)*multiple_space+1 - multiple_space/2 + 1:(i*multiple_space-multiple_space/2)]; %I need to actually take one more data point - so there are data points that are taken twice - with half the weight
                    x3 = ((i) * multiple_space - half_space, (i + 1) * multiple_space - half_space)
                    # i_indices_x2 = [(i-1)*multiple_space+1 - multiple_space/2,i*multiple_space - multiple_space/2 + 1];
            # elif therm_grid and u_grid: # If I want w on a u grid
            elif therm_grid:  # collocated grid
                if i == ix - 1:
                    x1 = (i) * multiple_space + 1
                    x2 = (i + 1) * multiple_space
                    # i_indices_x = [(i-1)*multiple_space+2:(i)*multiple_space]; %I need to actually take one more data point - so there are data points that are taken twice - with half the weight
                    x_tot = tuple(range(x1, x2))
                    x3 = ((i) * multiple_space, 0)
                    # i_indices_x2 = [(i-1)*multiple_space+1,1];
                else:
                    x1 = (i) * multiple_space + 1
                    x2 = (i + 1) * multiple_space
                    x_tot = tuple(range(x1, x2))
                    #      i_indices_x = [(i-1)*multiple_space+2:(i)*multiple_space]; %I need to actually take one more data point - so there are data points that are taken twice - with half the weight
                    x3 = ((i) * multiple_space, (i + 1) * multiple_space)
            #     i_indices_x2 = [(i-1)*multiple_space+1,(i)*multiple_space+1];


        elif therm_var:  # For coarse graining thermodynamic variables on the u grid
            # print(
            #     'Need to verify that I coarse grained correctly w on u. After I wrote f90 routine for advection with no c grid')
            if u_grid:
                if i == 0:
                    x1 = tuple(range(0, half_space))
                    x2 = tuple(range(field.shape[0] - half_space, field.shape[0]))
                    x_tot = x1 + x2
                    x3 = x_tot
                    # i_indices_w = [1:multiple_space/2,size(u_high,3)-multiple_space./2+1:size(u_high,3)];
                else:
                    x1 = (i) * multiple_space - half_space
                    x2 = (i + 1) * multiple_space - half_space
                    x_tot = tuple(range(x1, x2))
                    x3 = x_tot
                    # i_indices_w = [(i-1)*multiple_space+1 - multiple_space/2:(i*multiple_space-multiple_space/2)];
                # print(field[x_tot,y_tot,k].shape)
                # print(np.mean(field[x_tot,y_tot,k]).shape)
        # elif y_var:
        #     if u_grid:
        #         if i == 0:
        #             x1 = tuple(range(0, half_space))
        #             x2 = tuple(range(field.shape[0] - half_space, field.shape[0]))
        #             x_tot = x1 + x2
        #             x3 = x_tot
        #             # i_indices_w = [1:multiple_space/2,size(u_high,3)-multiple_space./2+1:size(u_high,3)];
        #         else:
        #             x1 = (i) * multiple_space - half_space
        #             x2 = (i + 1) * multiple_space - half_space
        #             x_tot = tuple(range(x1, x2))
        #             x3 = x_tot
        if len(field.shape) == 3:
            temp1 = np.mean(field_coarse_y[x_tot, :, :], axis=0)
            temp2 = np.mean(field_coarse_y[x3, :, :], axis=0)
            field_coarse[i, :, :] = temp1 * weight1 + temp2 * weight2
        if len(field.shape) == 2:
            temp1 = np.mean(field_coarse_y[x_tot, :,], axis=0)
            temp2 = np.mean(field_coarse_y[x3, :], axis=0)
            field_coarse[i, :] = temp1 * weight1 + temp2 * weight2

    return field_coarse


# This is a slow coarse graining!# Cannot slice twice...
# https://stackoverflow.com/questions/56328467/numpy-slicing-with-multiple-tuples

# Need to use this to average along one axis and then do the other axis with tuples..

# def calc_rho_rhow(adz, dz, ggr, rgas, tabs0, pres0,z, q0):
#     '''I need to provide'''
#     nz = adz.shape[0] + 1
#     zi = np.zeros(adz.shape[0] + 1)
#     presi = np.zeros(adz.shape[0] + 1)
#     pres = np.zeros(adz.shape[0])
#     rho = np.zeros(adz.shape[0])
#     rhow = np.zeros(adz.shape[0] + 1)
#
#     zi[0] = 0.
#     for k in range(1,nz):
#         zi[k] = zi[k - 1] + adz[k - 1] * dz
#
#     presi[0] = pres0
#     for k in range(0, nz-1): #This is how the density is actually calculated:
#         q0[k] = q0[k]/1000 #Convect units
#         tv0(k) = t0(k) * (1. + 0.61 * q0(k))
#         presr(k + 1) = presr(k) - ggr / cp / tv0(k) * (zi(k + 1) - zi(k))
#         presi(k + 1) = 1000. * presr(k + 1) ** (cp / rgas)
#         pres(k) = exp(log(presi(k)) + log(presi(k + 1) / presi(k)) * &
#                       (z(k) - zi(k)) / (zi(k + 1) - zi(k)))
#
#
#         # presi[k + 1] = presi[k] * np.exp(-ggr / rgas / tabs0[k] * (zi[k + 1] - zi[k]))
#         # pres[k] = 0.5 * (presi[k] + presi[k + 1])
#         # rho[k] = (presi[k] - presi[k + 1]) / (zi[k + 1] - zi[k]) / ggr * 100.
#
#     for k in range(1, nz-1):
#         rhow[k] = (pres[k - 1] - pres[k]) / (z[k] - z[k - 1]) / ggr * 100.
#
#     rhow[0] = 2 * rhow[1] - rhow[2]
#     rhow[nz-1] = 2 * rhow[nz-3] - rhow[nz - 2]
#     print('check rho and rhow and compare against simulation output! Note that in Fortran there are else ifs cases of calculating rho')
#
#     # presr(1) = (pres0 / 1000.) ** (rgas / cp)
#     # presi(1) = pres0
#
#
#     # Think if I should use this formulation for pres ?
#     # in pressz
#     # do    k = 1, nzm
#     # tv0(k) = tabs0(k) * prespot(k) * (1. + 0.61 * q0(k))
#     # presr(k + 1) = presr(k) - ggr / cp / tv0(k) * (zi(k + 1) - zi(k))
#     # presi(k + 1) = 1000. * presr(k + 1) ** (cp / rgas)
#     # pres(k) = exp(log(presi(k)) + log(presi(k + 1) / presi(k)) * &
#     #               (z(k) - zi(k)) / (zi(k + 1) - zi(k)))
#     # prespot(k) = (1000. / pres(k)) ** (rgas / cp)
#
#     return rho, rhow


# rho   1.154393       1.146271       1.135392       1.120565       1.100753
#    1.074795       1.042174       1.003638      0.9608132      0.9155130
#   0.8693907      0.8237123      0.7792484      0.7364439      0.6955087
#   0.6564057      0.6191757      0.5838215      0.5502272      0.5182902
#   0.4880713      0.4596207      0.4325974      0.4067678      0.3820913
#   0.3585862      0.3362901      0.3151487      0.2950976      0.2760165
#   0.2578100      0.2404558      0.2238967      0.2080857      0.1929768
#   0.1782718      0.1635249      0.1479584      0.1321913      0.1176219
#   0.1042457      9.1486968E-02  7.8266241E-02  6.4106166E-02  5.0210509E-02
#   3.8687121E-02  2.9772520E-02  2.2970462E-02
#  rhow   1.159861       1.150856       1.141851       1.129280       1.112242
#    1.089587       1.060366       1.024595      0.9834931      0.9389222
#   0.8927763      0.8465717      0.8013404      0.7576334      0.7157270
#   0.6757049      0.6375358      0.6012505      0.5667928      0.5340319
#   0.5029629      0.4736448      0.4459152      0.4194965      0.3942511
#   0.3701622      0.3472731      0.3255581      0.3049718      0.2854079
#   0.2667691      0.2489938      0.2320403      0.2158576      0.2004026
#   0.1854957      0.1707619      0.1555884      0.1399319      0.1248395
#   0.1109950      9.8136477E-02  8.5406758E-02  7.1737327E-02  5.7283621E-02
#   4.4221468E-02  3.3956394E-02  2.6152711E-02  1.8349029E-02
#   0.3701622      0.3472731      0.3255581      0.3049718      0.2854079
#   0.2667691      0.2489938      0.2320403      0.2158576      0.2004026
#   0.1854957      0.1707619      0.1555884      0.1399319      0.1248395
#   0.1109950      9.8136477E-02  8.5406758E-02  7.1737327E-02  5.7283621E-02
#   4.4221468E-02  3.3956394E-02  2.6152711E-02  1.8349029E-02



def coarse_grain_y(field, factor=16, y_var=True, therm_var=False, therm_grid=False, cgrid=False, v_grid=False):
    # Expexts to get x,y,z field
    ix = int(field.shape[0] / factor)
    jy = int(field.shape[1] / factor)

    if len(field.shape) == 3:
        temp = field.reshape((field.shape[0] // factor, factor, field.shape[1], field.shape[2]))
        kz = field.shape[2]
        field_coarse = np.zeros((ix, jy, kz))

    if len(field.shape) == 2:
        temp = field.reshape((field.shape[0] // factor, factor, field.shape[1]))
        field_coarse = np.zeros((ix, jy))

    field_coarse_y = np.mean(temp, axis=(1))

    weight1 = (factor - 1) / factor
    weight2 = 1 - weight1

    # ix = int(field.shape[0] / factor)
    # jy = int(field.shape[1] / factor)
    # kz = field.shape[2]
    # field_coarse = np.zeros((ix, jy, kz))
    multiple_space = factor

    half_space = int(multiple_space / 2)

    for j in range(jy):
        if y_var:
            if cgrid:  # keep on a C-grid:
                if j == 0:
                    y1 = 0
                    y2 = 4
                    y_tot = tuple(range(y1, y2))
                    # j_indices_y = 1;
                    y3 = (4)
                    # j_indices_y2 =1;

                else:
                    y1 = (j) * multiple_space + 1 - half_space
                    y2 = (j + 1) * multiple_space - half_space
                    y_tot = tuple(range(y1, y2))
                    # j_indices_y = [(j-1)*multiple_space+1 - multiple_space/2 + 1:(j*multiple_space - multiple_space/2)]; %I need to actually take one more data point - so there are data points that are taken twice - with half the weight
                    y3 = ((j) * multiple_space - half_space, (j + 1) * multiple_space - half_space)
                    # j_indices_y2 = [(j-1)*multiple_space+1 - multiple_space/2,j*multiple_space - multiple_space/2 + 1];
            elif therm_grid:  # collocated grid on thermodynamic grid
                if j == jy - 1:
                    y1 = (j) * multiple_space + 1
                    y2 = (j + 1) * multiple_space
                    y_tot = tuple(range(y1, y2))
                    y3 = ((j) * multiple_space)  # Not cyclical so cannot get the northmost point
                    # i_indices_x2 = [(i-1)*multiple_space+1,1];
                else:
                    y1 = (j) * multiple_space + 1
                    y2 = (j + 1) * multiple_space
                    y_tot = tuple(range(y1, y2))
                    y3 = ((j) * multiple_space, (j + 1) * multiple_space)

        elif therm_var:  # For coarse graining thermodynamic variables on the u grid
            # print(
            #     'Need to verify that I coarse grained correctly w on v. After I wrote f90 routine for advection with no c grid')

            # j_indices_y = [1:4]; %Not cyclical so might be an issue...
            #                     j_indices_y2 =5;
            #                     j_indices_w = [1:4];
            #                 else
            #                     j_indices_y = [(j-1)*multiple_space+1 - multiple_space/2 + 1:(j*multiple_space - multiple_space/2)]; %I need to actually take one more data point - so there are data points that are taken twice - with half the weight
            #                     j_indices_y2 = [(j-1)*multiple_space+1 - multiple_space/2,j*multiple_space - multiple_space/2 + 1];
            #                     j_indices_w = [(j-1)*multiple_space+1 - multiple_space/2 :(j*multiple_space - multiple_space/2)];
            #                 end

            if v_grid:
                if j == 0:
                    y1 = 0
                    y2 = 4
                    y_tot = tuple(range(y1, y2))
                    y3 = (4) # I used twice due to dimentional constraints...
                    # j_indices_w = [1:4];
                else:
                    y1 = (j) * multiple_space - half_space
                    y2 = (j + 1) * multiple_space - half_space
                    y_tot = tuple(range(y1, y2))
                    y3 = y_tot
                    # j_indices_w = [(j-1)*multiple_space+1 - multiple_space/2 :(j*multiple_space - multiple_space/2)];
        if len(field.shape) == 3:
            temp1 = np.mean(field_coarse_y[:, y_tot, :], axis=1)
            if len(field_coarse_y[:, y3, :].shape) == 3:
                temp2 = np.mean(field_coarse_y[:, y3, :], axis=1)
            else:
                temp2 = field_coarse_y[:, y3, :]

            field_coarse[:, j, :] = temp1 * weight1 + temp2 * weight2

        if len(field.shape) == 2:
            temp1 = np.mean(field_coarse_y[:, y_tot], axis=1)
            if len(field_coarse_y[:, y3].shape) == 2:
                temp2 = np.mean(field_coarse_y[:, y3], axis=1)
            else:
                temp2 = field_coarse_y[:, y3]
            field_coarse[:, j] = temp1 * weight1 + temp2 * weight2

    return field_coarse

def calc_mass_flux(w_high, w_coarse, qn_high, rhow, num_x, num_y, num_z, res, threshold_w = 0.1):
    print('This is inefficient... ')
    w_prime = np.zeros((num_x, num_y, num_z))
    updraft_high = np.zeros((num_x, num_y, num_z))

    for i in range(num_x):
        i_coarse = math.floor((i - 1) / res)
        for j in range(num_y):
            j_coarse = math.floor((j - 1) / res)
            for k in range(num_z):
                w_prime[i, j, k] = w_high[i, j, k] - w_coarse[i_coarse, j_coarse, i_coarse]
                if (np.abs(w_prime[i, j, k]) > threshold_w) and (qn_high[i, j, k] > 0):
                    updraft_high[i, j, k] = rhow[k] * w_prime[i, j, k]

    return updraft_high

def calc_mass_flux_efficient(w_high, w_coarse, qn_high, rhow, num_x, num_y, num_z, res, threshold_w = 0.1):

    w_prime = np.zeros((num_x, num_y, num_z))
    updraft_high = np.zeros((num_x, num_y, num_z))
    w_coarse_extended = np.repeat(w_coarse, res, axis=0)
    w_coarse_extended = np.repeat(w_coarse_extended, res, axis=1)
    w_prime = w_high - w_coarse_extended
    updraft_high = np.where((np.abs(w_prime) > threshold_w) & (qn_high > 0), rhow[None,None,:-1] * w_prime, 0.0)
    return updraft_high



def readnc_flip_axis(text_print, namevar, filename, ncname):
    f = Dataset(filename, mode='r')
    nc_var = f.variables[ncname][:]

    if len(nc_var.shape) ==3:
        ncname_mx = np.moveaxis(nc_var,(0,1,2),(2,1,0))
        print(text_print)
        print(np.mean(np.abs(ncname_mx[1:-1, 1:-1, :])))
        print(np.mean(np.abs(namevar[1:-1, 1:-1, :] - ncname_mx[1:-1, 1:-1, :])))
        f.close()
        fig, axs = plt.subplots(2, 2, figsize=(15, 8))
        n = 20
        lev = 10
        var_f2py = namevar[:, :, lev]
        var_matlab = ncname_mx[:, :, lev]

    else:
        ncname_mx = np.moveaxis(nc_var, (0, 1), (1, 0))
        print(text_print)
        print(np.mean(np.abs(ncname_mx[1:-1, 1:-1])))
        print(np.mean(np.abs(namevar[1:-1, 1:-1] - ncname_mx[1:-1, 1:-1])))
        f.close()
        fig, axs = plt.subplots(2, 2, figsize=(15, 8))
        n = 20
        lev = 10
        var_f2py = namevar[:, :]
        var_matlab = ncname_mx[:, :]

    range11 = np.max(np.max(np.abs(var_matlab)))
    levels = np.linspace(-range11, range11, n + 1)
    img1 = axs[0, 0].contourf(var_f2py, levels=levels, cmap='coolwarm')
    img2 = axs[1, 0].contourf(var_matlab, levels=levels, cmap='coolwarm')
    range22 = np.max(np.max(np.abs(var_f2py-var_matlab)))
    levels2 = np.linspace(-range22, range22, n + 1)
    img3 = axs[0, 1].contourf(var_f2py-var_matlab, levels=levels2, cmap='coolwarm')

    fig.colorbar(img1, ax=axs[0, 0])
    fig.colorbar(img2, ax=axs[1, 0])
    fig.colorbar(img3, ax=axs[0, 1])
    path = '/glade/u/home/janniy/python_fortran_coarse_graining/python_code/plots_temp/'
    plt.savefig(path + ncname + ".png",
                bbox_inches="tight",
                transparent=True)


    plt.close()

def calc_subgrid(field_coarse, field_resolved):
    return field_coarse - field_resolved

def create_data_array(field, field_txt, dict1 ,x_coarse, y_coarse,z,  dataset,
                      filename = 'none' , moisture_flag = False, test_mode = False,
                      unit1 = '_units',desc1='_desc'):

    if moisture_flag:
        fact = 1000.0 # To change the units of moisture related variables
    else:
        fact = 1.0
    if len(field.shape) == 3:
        array = xr.DataArray(
            data=np.moveaxis(field, (0, 1, 2), (2, 1, 0)) * fact,
            dims=['z', 'y', 'z'],
            coords=dict(
                z=('z', z),
                y=('y', y_coarse),
                x=('x', x_coarse),
            ),
            attrs=dict(
                description=dict1[field_txt+desc1],
                units=dict1[field_txt+unit1],
            )
        )
    elif len(field.shape) == 2:
        array = xr.DataArray(
            data=np.moveaxis(field, (0, 1), (1, 0)) * fact,
            dims=['y', 'x'],
            coords=dict(
                y=('y', y_coarse),
                x=('x', x_coarse),
            ),
            attrs=dict(
                description=dict1[field_txt+desc1],
                units=dict1[field_txt+unit1],
            )
        )
    elif len(field.shape) == 1:
        array = xr.DataArray(
            data=field * fact,
            dims=['z'],
            coords=dict(
                z=('z', z),
            ),
            attrs=dict(
                description=dict1[field_txt+desc1],
                units=dict1[field_txt+unit1],
            )
        )
    if test_mode: # Compare to matlab version
        readnc_flip_axis(name, field*fact, filename, dict1[field_txt])

    dataset[dict1[field_txt]] = array




def create_data_array_complete(field, nc_name, description, units,
                               x_coarse, y_coarse,z,  dataset,
                      filename = 'none' , moisture_flag = False, test_mode = False):
    if np.isnan(field).any():
        raise Exception('there is a nan in ' + nc_name)

    if moisture_flag:
        fact = 1000.0 # To change the units of moisture related variables
    else:
        fact = 1.0
    if len(field.shape) == 3:
        array = xr.DataArray(
            data=np.moveaxis(field, (0, 1, 2), (2, 1, 0)).astype(np.float32) * fact,
            dims=['z', 'y', 'x'],
            coords=dict(
                z=('z', z),
                y=('y', y_coarse),
                x=('x', x_coarse),
            ),
            attrs=dict(
                description=description,
                units=units,
            )
        )
    elif len(field.shape) == 2:
        array = xr.DataArray(
            data=np.moveaxis(field, (0, 1), (1, 0)).astype(np.float32) * fact,
            dims=['y', 'x'],
            coords=dict(
                y=('y', y_coarse),
                x=('x', x_coarse),
            ),
            attrs=dict(
                description=description,
                units=units,
            )
        )
    elif len(field.shape) == 1:
        array = xr.DataArray(
            data=field.astype(np.float32) * fact,
            dims=['z'],
            coords=dict(
                z=('z', z),
            ),
            attrs=dict(
                description=description,
                units=units,
            )
        )
    if test_mode: # Compare to matlab version
        readnc_flip_axis(nc_name, field*fact, filename, nc_name)

    dataset[nc_name] = array





def create_data_array_complete_gsam(field, nc_name, description, units,
                               lon, lat,z,  dataset,
                      filename = 'none' , moisture_flag = False, test_mode = False):
    if np.isnan(field).any():
        raise Exception('there is a nan in ' + nc_name)

    if moisture_flag:
        fact = 1000.0 # To change the units of moisture related variables
    else:
        fact = 1.0
    if len(field.shape) == 3:
        array = xr.DataArray(
            data=np.moveaxis(field, (0, 1, 2), (2, 1, 0)).astype(np.float32) * fact,
            dims=['z', 'lat', 'lon'],
            coords=dict(
                z=('z', z),
                lat=('lat', lat),
                lon=('lon', lon),
            ),
            attrs=dict(
                description=description,
                units=units,
            )
        )
    elif len(field.shape) == 2:
        array = xr.DataArray(
            data=np.moveaxis(field, (0, 1), (1, 0)).astype(np.float32) * fact,
            dims=['lat', 'lon'],
            coords=dict(
                lat=('lat', lat),
                lon=('lon', lon),
            ),
            attrs=dict(
                description=description,
                units=units,
            )
        )
    elif len(field.shape) == 1:
        array = xr.DataArray(
            data=field.astype(np.float32) * fact,
            dims=['z'],
            coords=dict(
                z=('z', z),
            ),
            attrs=dict(
                description=description,
                units=units,
            )
        )
    if test_mode: # Compare to matlab version
        readnc_flip_axis(nc_name, field*fact, filename, nc_name)

    dataset[nc_name] = array





def expand_dict_var_desc_unit(dict1,var_name, var_map, desc, unit, moisture = False, unit1 = '_units',desc1='_desc',moist = 'moist'):
    '''Add to dict variable, unit and description'''
    # dict0 = dict()
    # dict0[var_name] = var_map
    dict1[var_name] = var_map
    dict1[var_name + desc1] = desc
    dict1[var_name + unit1] = unit
    # dict1[var_name + moist] = moisture


def create_var_dict():

    var_dict = dict()
    expand_dict_var_desc_unit(var_dict, 'p', 'p',
                              'reference pressure (changes every snapshot)',
                              'hPa')

    expand_dict_var_desc_unit(var_dict, 'rho', 'rho',
                              'reference density',
                              'kg/m^3')

    expand_dict_var_desc_unit(var_dict, 'u_coarse', 'U',
                              'coarse grained zonal wind (c-grid)',
                              'm/s')

    expand_dict_var_desc_unit(var_dict, 'v_coarse', 'V',
                              'coarse grained meridional wind (c-grid)',
                              'm/s')

    expand_dict_var_desc_unit(var_dict, 'w_coarse', 'W',
                              'coarse grained vertical wind',
                              'm/s')

    expand_dict_var_desc_unit(var_dict, 'tabs_coarse', 'TABS',
                              'coarse grained temperature (beginning of time step)',
                              'K')


    expand_dict_var_desc_unit(var_dict, 'tabs_resolved_init', 'TABS_RESOLVED_INIT',
                              'resolved temperature (beginning of time step)',
                              'K')

    expand_dict_var_desc_unit(var_dict, 't_coarse', 'T',
                              'coarse-grained modified energy variable (see Yuval and OGorman 2020)',
                              'K')

    expand_dict_var_desc_unit(var_dict, 'tfull_coarse', 'TFULL_INIT',
                              'coarse-grained liquid/ice water moist static energy',
                              'K')

    expand_dict_var_desc_unit(var_dict, 'qv_coarse', 'Q',
                              'coarse-grained water vapor mixing ratio',
                              'g/kg', moisture = True)

    expand_dict_var_desc_unit(var_dict, 'qn_coarse_end', 'QN',
                              'coarse-grained cloud water+ice mixing ratio at end of time step',
                              'g/kg', moisture = True)

    expand_dict_var_desc_unit(var_dict, 'qn_coarse', 'QN_COARSE_INIT',
                              'coarse-grained cloud water+ice mixing ratio at beginning of time step',
                              'g/kg', moisture = True)

    expand_dict_var_desc_unit(var_dict, 'qn_resolved_init', 'QN_RESOLVED_INIT',
                              'resolved cloud water+ice mixing ratio at beginning of time step',
                              'g/kg', moisture = True)


    var1 = 'qp_coarse'
    var_dict['qp_coarse'] = 'QP_COARSE_INIT'
    var_dict['qp_coarse_desc'] = 'coarse-grained preciptating water mixing ratio'
    var_dict['qp_coarse_units'] = 'g/kg'

    var1 = 'Qrad_coarse'
    var_dict['Qrad_coarse'] = 'QRAD'
    var_dict['Qrad_coarse_desc'] = 'coarse-grained heating due to radiation'
    var_dict['Qrad_coarse_units'] = 'K/day'

    #Scallar advection
    var1 = 'tfull_flux_x_out_coarse'
    var_dict['tfull_flux_x_out_coarse'] = 'TFULL_FLUX_COARSE_X'
    var_dict['tfull_flux_x_out_coarse_desc'] = 'coarse-grained zonal advective flux of liquid/ice water moist static energy'
    var_dict['tfull_flux_x_out_coarse_units'] = 'K kg/m^2/s'

    var1 = 'tfull_flux_y_out_coarse'
    var_dict['tfull_flux_y_out_coarse'] = 'TFULL_FLUX_COARSE_Y'
    var_dict['tfull_flux_y_out_coarse_desc'] = 'coarse-grained meridional advective flux of liquid/ice water moist static energy'
    var_dict['tfull_flux_y_out_coarse_units'] = 'K kg/m^2/s'

    var1 = 'tfull_flux_z_out_coarse'
    var_dict['tfull_flux_z_out_coarse'] = 'TFULL_FLUX_COARSE_Z'
    var_dict['tfull_flux_z_out_coarse_desc'] = 'coarse-grained vertical advective flux of liquid/ice water moist static energy'
    var_dict['tfull_flux_z_out_coarse_units'] = 'K kg/m^2/s'

    var1 = 't_flux_x_out_coarse'
    var_dict['t_flux_x_out_coarse'] = 'T_FLUX_COARSE_X'
    var_dict['t_flux_x_out_coarse_desc'] = 'coarse-grained zonal advective flux of modified energy variable'
    var_dict['t_flux_x_out_coarse_units'] = 'K kg/m^2/s'

    var1 = 't_flux_y_out_coarse'
    var_dict['t_flux_y_out_coarse'] = 'T_FLUX_COARSE_Y'
    var_dict['t_flux_y_out_coarse_desc'] = 'coarse-grained meridional advective flux of modified energy variable'
    var_dict['t_flux_y_out_coarse_units'] = 'K kg/m^2/s'

    var1 = 't_flux_z_out_coarse'
    var_dict['t_flux_z_out_coarse'] = 'T_FLUX_COARSE_Z'
    var_dict['t_flux_z_out_coarse_desc'] = 'coarse-grained vertical advective flux of modified energy variable'
    var_dict['t_flux_z_out_coarse_units'] = 'K kg/m^2/s'

    var1 = 'q_flux_x_out_coarse'
    var_dict['q_flux_x_out_coarse'] = 'QT_FLUX_COARSE_X'
    var_dict['q_flux_x_out_coarse_desc'] = 'coarse-grained zonal advective flux of non-precipitating water (qt) mixing raio'
    var_dict['q_flux_x_out_coarse_units'] = 'g/kg kg/m^2/s'

    var1 = 'q_flux_y_out_coarse'
    var_dict['q_flux_y_out_coarse'] = 'QT_FLUX_COARSE_Y'
    var_dict['q_flux_y_out_coarse_desc'] = 'coarse-grained meridional advective flux of non-precipitating water (qt) mixing raio'
    var_dict['q_flux_y_out_coarse_units'] = 'g/kg kg/m^2/s'

    var1 = 'q_flux_z_out_coarse'
    var_dict['q_flux_z_out_coarse'] = 'QT_FLUX_COARSE_Z'
    var_dict['q_flux_z_out_coarse_desc'] = 'coarse-grained vertical advective flux of non-precipitating water (qt) mixing raio'
    var_dict['q_flux_z_out_coarse_units'] = 'g/kg kg/m^2/s'

    var1 = 'qp_flux_x_out_coarse'
    var_dict['qp_flux_x_out_coarse'] = 'QP_FLUX_COARSE_X'
    var_dict['qp_flux_x_out_coarse_desc'] = 'coarse-grained zonal meridional flux of precipitating water (qp) mixing raio'
    var_dict['qp_flux_x_out_coarse_units'] = 'g/kg kg/m^2/s'

    var1 = 'qp_flux_y_out_coarse'
    var_dict['qp_flux_y_out_coarse'] = 'QP_FLUX_COARSE_Y'
    var_dict['qp_flux_y_out_coarse_desc'] = 'coarse-grained vertical meridional flux of precipitating water (qp) mixing raio'
    var_dict['qp_flux_y_out_coarse_units'] = 'g/kg kg/m^2/s'

    var1 = 'qp_flux_z_out_coarse'
    var_dict['qp_flux_z_out_coarse'] = 'QP_FLUX_COARSE_Z'
    var_dict['qp_flux_z_out_coarse_desc'] = 'coarse-grained vertical advective flux of precipitating water (qp) mixing raio'
    var_dict['qp_flux_z_out_coarse_units'] = 'g/kg kg/m^2/s'

    # subgrid Scallar advection
    var1 = 'tfull_flux_x_out_subgrid'
    var_dict['tfull_flux_x_out_subgrid'] = 'TFULL_FLUX_X'
    var_dict['tfull_flux_x_out_subgrid_desc'] = 'subgrid zonal advective flux of liquid/ice water moist static energy'
    var_dict['tfull_flux_x_out_subgrid_units'] = 'K kg/m^2/s'

    var1 = 'tfull_flux_y_out_subgrid'
    var_dict['tfull_flux_y_out_subgrid'] = 'TFULL_FLUX_Y'
    var_dict['tfull_flux_y_out_subgrid_desc'] = 'subgrid meridional advective flux of liquid/ice water moist static energy'
    var_dict['tfull_flux_y_out_subgrid_units'] = 'K kg/m^2/s'

    var1 = 'tfull_flux_z_out_subgrid'
    var_dict['tfull_flux_z_out_subgrid'] = 'TFULL_FLUX_Z'
    var_dict['tfull_flux_z_out_subgrid_desc'] = 'subgrid vertical advective flux of liquid/ice water moist static energy'
    var_dict['tfull_flux_z_out_subgrid_units'] = 'K kg/m^2/s'

    var1 = 't_flux_x_out_subgrid'
    var_dict['t_flux_x_out_subgrid'] = 'T_FLUX_X'
    var_dict['t_flux_x_out_subgrid_desc'] = 'subgrid zonal advective flux of modified energy variable'
    var_dict['t_flux_x_out_subgrid_units'] = 'K kg/m^2/s'

    var1 = 't_flux_y_out_subgrid'
    var_dict['t_flux_y_out_subgrid'] = 'T_FLUX_Y'
    var_dict['t_flux_y_out_subgrid_desc'] = 'subgrid meridional advective flux of modified energy variable'
    var_dict['t_flux_y_out_subgrid_units'] = 'K kg/m^2/s'

    var1 = 't_flux_z_out_subgrid'
    var_dict['t_flux_z_out_subgrid'] = 'T_FLUX_Z'
    var_dict['t_flux_z_out_subgrid_desc'] = 'subgrid vertical advective flux of modified energy variable'
    var_dict['t_flux_z_out_subgrid_units'] = 'K kg/m^2/s'

    var1 = 'q_flux_x_out_subgrid'
    var_dict['q_flux_x_out_subgrid'] = 'QT_FLUX_X'
    var_dict['q_flux_x_out_subgrid_desc'] = 'subgrid zonal advective flux of non-precipitating water (qt) mixing raio'
    var_dict['q_flux_x_out_subgrid_units'] = 'g/kg kg/m^2/s'

    var1 = 'q_flux_y_out_subgrid'
    var_dict['q_flux_y_out_subgrid'] = 'QT_FLUX_Y'
    var_dict['q_flux_y_out_subgrid_desc'] = 'subgrid meridional advective flux of non-precipitating water (qt) mixing raio'
    var_dict['q_flux_y_out_subgrid_units'] = 'g/kg kg/m^2/s'

    var1 = 'q_flux_z_out_subgrid'
    var_dict['q_flux_z_out_subgrid'] = 'QT_FLUX_Z'
    var_dict['q_flux_z_out_subgrid_desc'] = 'subgrid vertical advective flux of non-precipitating water (qt) mixing raio'
    var_dict['q_flux_z_out_subgrid_units'] = 'g/kg kg/m^2/s'

    var1 = 'qp_flux_x_out_subgrid'
    var_dict['qp_flux_x_out_subgrid'] = 'QP_FLUX_X'
    var_dict['qp_flux_x_out_subgrid_desc'] = 'subgrid zonal advective flux of precipitating water (qp) mixing raio'
    var_dict['qp_flux_x_out_subgrid_units'] = 'g/kg kg/m^2/s'

    var1 = 'qp_flux_y_out_subgrid'
    var_dict['qp_flux_y_out_subgrid'] = 'QP_FLUX_Y'
    var_dict['qp_flux_y_out_subgrid_desc'] = 'subgrid meridional advective flux of precipitating water (qp) mixing raio'
    var_dict['qp_flux_y_out_subgrid_units'] = 'g/kg kg/m^2/s'

    var1 = 'qp_flux_z_out_subgrid'
    var_dict['qp_flux_z_out_subgrid'] = 'QP_FLUX_Z'
    var_dict['qp_flux_z_out_subgrid_desc'] = 'subgrid vertical advective flux of precipitating water (qp) mixing raio'
    var_dict['qp_flux_z_out_subgrid_units'] = 'g/kg kg/m^2/s'

    #diffusion
    var1 = 't_diff_flux_x_coarse'
    var_dict['t_diff_flux_x_coarse'] = 'T_DIFF_F_COARSE_X'
    var_dict['t_diff_flux_x_coarse_desc'] = 'coarse-grained zonal diffusive flux of modified energy variable'
    var_dict['t_diff_flux_x_coarse_units'] = 'K*m/s'

    var1 = 't_diff_flux_y_coarse'
    var_dict['t_diff_flux_y_coarse'] = 'T_DIFF_F_COARSE_Y'
    var_dict['t_diff_flux_y_coarse_desc'] = 'coarse-grained meridional diffusive flux of modified energy variable'
    var_dict['t_diff_flux_y_coarse_units'] = 'K*m/s'

    var1 = 't_diff_flux_z_coarse'
    var_dict['t_diff_flux_z_coarse'] = 'T_DIFF_F_COARSE_Z'
    var_dict['t_diff_flux_z_coarse_desc'] = 'coarse-grained vertical diffusive flux of modified energy variable'
    var_dict['t_diff_flux_z_coarse_units'] = 'K kg/m^2/s'

    var1 = 'tfull_diff_flux_x_coarse'
    var_dict['tfull_diff_flux_x_coarse'] = 'TFULL_DIFF_F_COARSE_X'
    var_dict['tfull_diff_flux_x_coarse_desc'] = 'coarse-grained zonal diffusive flux of liquid/ice water moist static energy'
    var_dict['tfull_diff_flux_x_coarse_units'] = 'K*m/s'

    var1 = 'tfull_diff_flux_y_coarse'
    var_dict['tfull_diff_flux_y_coarse'] = 'TFULL_DIFF_F_COARSE_Y'
    var_dict['tfull_diff_flux_y_coarse_desc'] = 'coarse-grained meridional diffusive flux of liquid/ice water moist static energy'
    var_dict['tfull_diff_flux_y_coarse_units'] = 'K*m/s'

    var1 = 'tfull_diff_flux_z_coarse'
    var_dict['tfull_diff_flux_z_coarse'] = 'TFULL_DIFF_F_COARSE_Z'
    var_dict['tfull_diff_flux_z_coarse_desc'] = 'coarse-grained vertical diffusive flux of liquid/ice water moist static energy'
    var_dict['tfull_diff_flux_z_coarse_units'] = 'K kg/m^2/s'

    var1 = 'q_diff_flux_x_coarse'
    var_dict['q_diff_flux_x_coarse'] = 'QT_DIFF_F_COARSE_X'
    var_dict['q_diff_flux_x_coarse_desc'] = 'coarse-grained  zonal diffusive flux of non-precipitating water (qt) mixing raio'
    var_dict['q_diff_flux_x_coarse_units'] = 'g/kg m/s'

    var1 = 'q_diff_flux_y_coarse'
    var_dict['q_diff_flux_y_coarse'] = 'QT_DIFF_F_COARSE_Y'
    var_dict['q_diff_flux_y_coarse_desc'] = 'coarse-grained  meridional diffusive flux of non-precipitating water (qt) mixing raio'
    var_dict['q_diff_flux_y_coarse_units'] = 'g/kg m/s'

    var1 = 'q_diff_flux_z_coarse'
    var_dict['q_diff_flux_z_coarse'] = 'QT_DIFF_F_COARSE_Z'
    var_dict['q_diff_flux_z_coarse_desc'] = 'coarse-grained  vertical diffusive flux of non-precipitating water (qt) mixing raio'
    var_dict['q_diff_flux_z_coarse_units'] = 'g/kg kg/m^2/s'

    var1 = 'qp_diff_flux_x_coarse'
    var_dict['qp_diff_flux_x_coarse'] = 'QP_DIFF_F_COARSE_X'
    var_dict['qp_diff_flux_x_coarse_desc'] = 'coarse-grained  zonal diffusive flux of precipitating water (qp) mixing raio'
    var_dict['qp_diff_flux_x_coarse_units'] = 'g/kg m/s'

    var1 = 'qp_diff_flux_y_coarse'
    var_dict['qp_diff_flux_y_coarse'] = 'QP_DIFF_F_COARSE_Y'
    var_dict['qp_diff_flux_y_coarse_desc'] = 'coarse-grained  meridional diffusive flux of precipitating water (qp) mixing raio'
    var_dict['qp_diff_flux_y_coarse_units'] = 'g/kg m/s'

    var1 = 'qp_diff_flux_z_coarse'
    var_dict['qp_diff_flux_z_coarse'] = 'QP_DIFF_F_COARSE_Z'
    var_dict['qp_diff_flux_z_coarse_desc'] = 'coarse-grained  vertical diffusive flux of precipitating water (qp) mixing raio'
    var_dict['qp_diff_flux_z_coarse_units'] = 'g/kg kg/m^2/s'

    #subgrid diffusion:
    var1 = 'tfull_diff_flux_x_subgrid'
    var_dict['tfull_diff_flux_x_subgrid'] = 'TFULL_DIFF_FLUX_X'
    var_dict['tfull_diff_flux_x_subgrid_desc'] = 'subgrid zonal diffusive flux of liquid/ice water moist static energy'
    var_dict['tfull_diff_flux_x_subgrid_units'] = 'K m/s'

    var1 = 'tfull_diff_flux_y_subgrid'
    var_dict['tfull_diff_flux_y_subgrid'] = 'TFULL_DIFF_FLUX_Y'
    var_dict['tfull_diff_flux_y_subgrid_desc'] = 'subgrid meridional diffusive flux of liquid/ice water moist static energy'
    var_dict['tfull_diff_flux_y_subgrid_units'] = 'K m/s'

    var1 = 'tfull_diff_flux_z_subgrid'
    var_dict['tfull_diff_flux_z_subgrid'] = 'TFULL_DIFF_FLUX_Z'
    var_dict['tfull_diff_flux_z_subgrid_desc'] = 'subgrid vertical diffusive flux of liquid/ice water moist static energy'
    var_dict['tfull_diff_flux_z_subgrid_units'] = 'K kg/m^2/s'

    var1 = 't_diff_flux_x_subgrid'
    var_dict['t_diff_flux_x_subgrid'] = 'T_DIFF_FLUX_X'
    var_dict['t_diff_flux_x_subgrid_desc'] = 'subgrid zonal diffusive flux of modified energy variable'
    var_dict['t_diff_flux_x_subgrid_units'] = 'K*m/s'

    var1 = 't_diff_flux_y_subgrid'
    var_dict['t_diff_flux_y_subgrid'] = 'T_DIFF_FLUX_Y'
    var_dict['t_diff_flux_y_subgrid_desc'] = 'subgrid meridional diffusive flux of modified energy variable'
    var_dict['t_diff_flux_y_subgrid_units'] = 'K*m/s'

    var1 = 't_diff_flux_z_subgrid'
    var_dict['t_diff_flux_z_subgrid'] = 'T_DIFF_FLUX_Z'
    var_dict['t_diff_flux_z_subgrid_desc'] = 'subgrid vertical diffusive flux of modified energy variable'
    var_dict['t_diff_flux_z_subgrid_units'] = 'K kg/m^2/s'

    var1 = 'q_diff_flux_x_subgrid'
    var_dict['q_diff_flux_x_subgrid'] = 'QT_DIFF_FLUX_X'
    var_dict['q_diff_flux_x_subgrid_desc'] = 'subgrid zonal diffusive flux of non-precipitating water (qt) mixing raio'
    var_dict['q_diff_flux_x_subgrid_units'] = 'g/kg m/s'

    var1 = 'q_diff_flux_y_subgrid'
    var_dict['q_diff_flux_y_subgrid'] = 'QT_DIFF_FLUX_Y'
    var_dict['q_diff_flux_y_subgrid_desc'] = 'subgrid meridional diffusive flux of non-precipitating water (qt) mixing raio'
    var_dict['q_diff_flux_y_subgrid_units'] = 'g/kg m/s'

    var1 = 'q_diff_flux_z_subgrid'
    var_dict['q_diff_flux_z_subgrid'] = 'QT_DIFF_FLUX_Z'
    var_dict['q_diff_flux_z_subgrid_desc'] = 'subgrid vertical diffusive flux of non-precipitating water (qt) mixing raio'
    var_dict['q_diff_flux_z_subgrid_units'] = 'g/kg kg/m^2/s'

    expand_dict_var_desc_unit(var_dict, 'qp_diff_flux_x_subgrid', 'QP_DIFF_FLUX_X',
                              'subgrid zonal diffusive flux of non-precipitating water (qt) mixing raio',
                              'g/kg m/s')

    expand_dict_var_desc_unit(var_dict, 'qp_diff_flux_y_subgrid', 'QP_DIFF_FLUX_Y',
                              'subgrid meridional diffusive flux of non-precipitating water (qt) mixing raio',
                              'g/kg m/s')

    expand_dict_var_desc_unit(var_dict, 'qp_diff_flux_z_subgrid', 'QP_DIFF_FLUX_Z',
                              'subgrid vertical diffusive flux of non-precipitating water (qt) mixing raio',
                              'g/kg kg/m^2/s')


    #Microphysics
    expand_dict_var_desc_unit(var_dict, 'dqp_mic_coarse', 'DQP',
                              'coarse-grained qp tendency due to microphysics',
                              'g/kg/s')

    expand_dict_var_desc_unit(var_dict, 'dqp_t_tend_coarse', 'DQP_T_TEND_COARSE',
                              'coarse-grained t tendency due to microphysics',
                              'K/s')

    expand_dict_var_desc_unit(var_dict, 'dqp_mic_subgrid', 'DQP_RESOLVED',
                              'subgrid qp tendency due to microphysics',
                              'g/kg/s')

    expand_dict_var_desc_unit(var_dict, 'dqp_t_tend_subgrid', 'DQP_T_TEND_RESOLVED',
                              'subgrid t tendency due to microphysics',
                              'K/s')


    #Sedimentation
    expand_dict_var_desc_unit(var_dict, 'dq_sed_coarse', 'QT_TEND_CLOUD_COARSE',
                              'coarse-grained qt tendency due to sedimentation',
                              'g/kg/s')

    expand_dict_var_desc_unit(var_dict, 'dt_sed_coarse', 'LAT_HEAT_CLOUD_COARSE',
                              'coarse-grained latent heatung (tendency) due to sedimentation',
                              'K/s')

    expand_dict_var_desc_unit(var_dict, 'fz_coarse', 'CLOUD_FZ_FLUX_COARSE',
                              'coarse-grained water flux due to sedimentation',
                              'gr/kg m^2/s/kg')

    expand_dict_var_desc_unit(var_dict, 'fzt_coarse', 'CLOUD_FZT_FLUX_COARSE',
                              'coarse-grained heating flux due to sedimentation',
                              'K m^2/s/kg')


    expand_dict_var_desc_unit(var_dict, 'dq_sed_subgrid', 'QT_TEND_CLOUD_RES',
                              'subgrid qt tendency due to sedimentation',
                              'g/kg/s')

    expand_dict_var_desc_unit(var_dict, 'dt_sed_subgrid', 'LAT_HEAT_CLOUD_RES',
                              'subgrid latent heatung (tendency) due to sedimentation',
                              'K/s')


    expand_dict_var_desc_unit(var_dict, 'fz_subgrid', 'CLOUD_FZ_FLUX_RES',
                              'subgrid water flux due to sedimentation',
                              'gr/kg m^2/s/kg')

    expand_dict_var_desc_unit(var_dict, 'fzt_subgrid', 'CLOUD_FZT_FLUX_RES',
                              'subgrid heating flux due to sedimentation',
                              'K m^2/s/kg')


    #fall
    expand_dict_var_desc_unit(var_dict, 'qpfall_coarse', 'DQP_FALL_COARSE',
                              'coarse-grained qp tendency due to falling precip',
                              'g/kg/s')

    expand_dict_var_desc_unit(var_dict, 'lat_heat_fall_coarse', 'T_FALL_COARSE',
                              'coarse-grained latent heating (tendency) due to falling precip',
                              'K/s')

    expand_dict_var_desc_unit(var_dict, 'precip_flux_coarse', 'PRECIP',
                              'coarse-grained precipitation flux',
                              'kg/m^2/s') # For consistancy with matlab - not to multiply by 1000!


    expand_dict_var_desc_unit(var_dict, 'qpfall_subgrid', 'DQP_FALL_RES',
                              'subgrid qp tendency due to falling precip',
                              'g/kg/s')

    expand_dict_var_desc_unit(var_dict, 'lat_heat_fall_subgrid', 'T_FALL_RES',
                              'subgrid latent heating (tendency) due to falling precip',
                              'K/s')

    expand_dict_var_desc_unit(var_dict, 'precip_flux_subgrid', 'PRECIP_RESOLVED',
                              'subgrid precipitation flux',
                              'kg/m^2/s')  # For consistancy with matlab - not to multiply by 1000!

    #diffusivity

    expand_dict_var_desc_unit(var_dict, 'tkz_coarse', 'TKZ_COARSE',
                              'coarse grained diffusivity - for momentum diffusion',
                              'm^2/s')

    expand_dict_var_desc_unit(var_dict, 'tkh_z_coarse', 'TKHZ_COARSE',
                              'coarse grained conductivity',
                              'm^2/s')  # Before I calculated the diffusivity (needed to multiply by inverse Prendtl number)


    expand_dict_var_desc_unit(var_dict, 'tkh_z_resolved', 'TKHZ_RESOLVED',
                              'resolved conductivity',
                              'm^2/s')  # Before I calculated the diffusivity (needed to multiply by inverse Prendtl number)



    # momentum advection:

    expand_dict_var_desc_unit(var_dict, 'dudt_coarse', 'U_ADV_COARSE',
                              'coarse grained u tendency due to vertical advection',
                              'm/s^2')

    expand_dict_var_desc_unit(var_dict, 'dvdt_coarse', 'V_ADV_COARSE',
                              'coarse grained v tendency due to vertical advection',
                              'm/s^2')

    expand_dict_var_desc_unit(var_dict, 'dwdt_coarse', 'W_ADV_COARSE',
                              'coarse grained w tendency due to vertical advection',
                              'm/s^2')

    expand_dict_var_desc_unit(var_dict, 'dudt_subgrid', 'U_ADV_RESOLVED',
                              'subgrid u tendency due to vertical advection (cgrid)',
                              'm/s^2')

    expand_dict_var_desc_unit(var_dict, 'dvdt_subgrid', 'V_ADV_RESOLVED',
                              'subgrid v tendency due to vertical advection (cgrid)',
                              'm/s^2')

    expand_dict_var_desc_unit(var_dict, 'dwdt_subgrid', 'W_ADV_RESOLVED',
                              'subgrid w tendency due to vertical advection (cgrid)',
                              'm/s^2')

    expand_dict_var_desc_unit(var_dict, 'dudt_coarse_collocated', 'U_ADV_NORM_GRID_COARSE',
                              'coarse grained u tendency due to vertical advection (collocated grid)',
                              'm/s^2')

    expand_dict_var_desc_unit(var_dict, 'dvdt_coarse_collocated', 'V_ADV_NORM_GRID_COARSE',
                              'coarse grained v tendency due to vertical advection (collocated grid)',
                              'm/s^2')

    expand_dict_var_desc_unit(var_dict, 'dudt_subgrid_collocated', 'U_ADV_NORM_GRID_RESOLVED',
                              'subgrid u tendency due to vertical advection (collocated grid)',
                              'm/s^2')

    expand_dict_var_desc_unit(var_dict, 'dvdt_subgrid_collocated', 'V_ADV_NORM_GRID_RESOLVED',
                              'subgrid v tendency due to vertical advection (collocated grid)',
                              'm/s^2')

    expand_dict_var_desc_unit(var_dict, 'dudt_subgrid_u_grid', 'U_ADV_U_GRID_RESOLVED',
                              'subgrid u tendency due to vertical advection (u grid)',
                              'm/s^2')

    expand_dict_var_desc_unit(var_dict, 'dvdt_subgrid_v_grid', 'V_ADV_V_GRID_RESOLVED',
                              'subgrid v tendency due to vertical advection (v grid)',
                              'm/s^2')




    # momentum surface fluxes.

    # NN: (multiplied output by dz - also in f2py version)
    # irhoadz(k) = 1.0 / (rho(k) * adz(k) * dz) ! Usefulfactor
    # u_tendency_adv(k) = - (u_flux_adv(k + 1) - u_flux_adv(k)) * irhoadz(k)

    expand_dict_var_desc_unit(var_dict, 'fluxbu_coarse', 'SFLUX_U_COARSE',
                              'coarse-grained surface u flux (cgrid)',
                              'm/s^2 kg/m^3') # Verified units!

    expand_dict_var_desc_unit(var_dict, 'fluxbv_coarse', 'SFLUX_V_COARSE',
                              'coarse-grained surface v flux (cgrid)',
                              'm/s^2 kg/m^3')  # Verified units!

    expand_dict_var_desc_unit(var_dict, 'fluxbu_collocated_coarse', 'U_SURF_FLUX_NORM_GRID_COARSE',
                              'coarse-grained surfave u flux (collocated grid)',
                              'm/s^2 kg/m^3') # Verified units!


    expand_dict_var_desc_unit(var_dict, 'fluxbv_collocated_coarse', 'V_SURF_FLUX_NORM_GRID_COARSE',
                              'coarse-grained surfave v flux (collocated grid)',
                              'm/s^2 kg/m^3') # Verified units!

    expand_dict_var_desc_unit(var_dict, 'fluxbu_subgrid', 'SFLUX_U_RESOLVED',
                              'subgrid surface u flux (cgrid)',
                              'm/s^2 kg/m^3')  # Verified units!

    expand_dict_var_desc_unit(var_dict, 'fluxbv_subgrid', 'SFLUX_V_RESOLVED',
                              'subgrid surface v flux (cgrid)',
                              'm/s^2 kg/m^3')  # Verified units!

    expand_dict_var_desc_unit(var_dict, 'fluxbu_collocated_subgrid', 'U_SURF_FLUX_NORM_GRID_RESOLVED',
                              'subgrid surfave u flux (collocated grid)',
                              'm/s^2 kg/m^3') # Verified units!


    expand_dict_var_desc_unit(var_dict, 'fluxbv_collocated_subgrid', 'V_SURF_FLUX_NORM_GRID_RESOLVED',
                              'subgrid surfave v flux (collocated grid)',
                              'm/s^2 kg/m^3') # Verified units!

    expand_dict_var_desc_unit(var_dict, 'fluxbu_u_grid_subgrid', 'U_SURF_FLUX_U_GRID_RESOLVED',
                              'subgrid surfave u flux (u grid)',
                              'm/s^2 kg/m^3')  # Verified units!

    expand_dict_var_desc_unit(var_dict, 'fluxbv_v_grid_subgrid', 'V_SURF_FLUX_V_GRID_RESOLVED',
                              'subgrid surfave v flux (v grid)',
                              'm/s^2 kg/m^3')  # Verified units!


    expand_dict_var_desc_unit(var_dict, 'pp_coarse', 'PP',
                              'coarse-grained pressure perturbation',
                              'Pascal')  # Verified units!

    expand_dict_var_desc_unit(var_dict, 'u_coarse_collocated', 'U_NORM_GRID',
                              'coarse-grained u (collocated grid)',
                              'm/s')  # Verified units!


    expand_dict_var_desc_unit(var_dict, 'v_coarse_collocated', 'V_NORM_GRID',
                              'coarse-grained v (collocated grid)',
                              'm/s')  # Verified units!

    expand_dict_var_desc_unit(var_dict, 'mass_flux_coarse', 'UPDRAFT_01_TRESH',
                              'Mass flux with 0.1 threshold',
                              'kg/m^2/s')  # Verified units!

    expand_dict_var_desc_unit(var_dict, 'tabs_sigma', 'TABS_SIGMA',
                              'tabs on interpolated 74 sigma levels',
                              'K')  # Verified units!


    expand_dict_var_desc_unit(var_dict, 'qt_sigma', 'QT_SIGMA',
                              'qt on interpolated 74 sigma levels',
                              'g/kg')  # Verified units!

    expand_dict_var_desc_unit(var_dict, 'sigma_tot', 'SIGMA_TOT',
                              'the sigma coordinates I actually had data on',
                              '-')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'qp_sigma', 'QP_SIGMA',
                              'qp on interpolated 74 sigma levels',
                              'g/kg')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'tfull_flux_z_out_subgrid_sigma', 'TFULL_FLUX_Z_OUT_SUBGRID_SIGMA',
                              'tfull_flux_z_out_subgrid_sigma on interpolated 74 sigma levels',
                              'K kg/m^2/s')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 't_flux_z_out_subgrid_sigma', 'T_FLUX_Z_OUT_SUBGRID_SIGMA',
                              't_flux_z_out_subgrid_sigma on interpolated 74 sigma levels',
                              'K kg/m^2/s')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'q_flux_z_out_subgrid_sigma', 'Q_FLUX_Z_OUT_SUBGRID_SIGMA',
                              'q_flux_z_out_subgrid_sigma on interpolated 74 sigma levels',
                              'g/kg kg/m^2/s')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'qp_flux_z_out_subgrid_sigma', 'QP_FLUX_Z_OUT_SUBGRID_SIGMA',
                              'qp_flux_z_out_subgrid_sigma on interpolated 74 sigma levels',
                              'g/kg kg/m^2/s')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'qt_flux_z_out_subgrid_sigma', 'QT_FLUX_Z_OUT_SUBGRID_SIGMA',
                              'qt_flux_z_out_subgrid_sigma on interpolated 74 sigma levels',
                              'g/kg kg/m^2/s')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'rho_sigma', 'RHO_SIGMA',
                              'rho_sigma on interpolated 74 sigma levels',
                              'kg/m^3')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'keddysc_sigma', 'KEDDYSC_SIGMA',
                              'keddysc_sigma on interpolated 74 sigma levels',
                              'm2/s')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'qrad_sigma', 'QRAD_SIGMA',
                              'qrad_sigma on interpolated 74 sigma levels',
                              'K/s')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'prec_sigma', 'PREC_SIGMA',
                              'prec_sigma on interpolated 74 sigma levels',
                              'kg/m2/s')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'lprec_sigma', 'LPREC_SIGMA',
                              'lprec_sigma on interpolated 74 sigma levels',
                              'W/m2')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'sed_sigma', 'SED_SIGMA',
                              'sed_sigma on interpolated 74 sigma levels',
                              'kg/m2/s')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'lsed_sigma', 'LSED_SIGMA',
                              'lsed_sigma on interpolated 74 sigma levels',
                              'W/m2')  # Verified units!
        
    expand_dict_var_desc_unit(var_dict, 'rhoqpw_sigma', 'RHOQPW_SIGMA',
                              'rhoqpw_sigma on interpolated 74 sigma levels',
                              'kg/m2/s g/')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'rhoqps_sigma', 'RHOQPS_SIGMA',
                              'rhoqps_sigma on interpolated 74 sigma levels',
                              'kg/m2/s g/')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'qp_micro_sigma', 'QP_MICRO_SIGMA',
                              'qp_micro_sigma on interpolated 74 sigma levels',
                              'kg/kg/s')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 't_sigma', 'T_SIGMA',
                              't_sigma on interpolated 74 sigma levels',
                              'K')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'qv_sigma', 'QV_SIGMA',
                              'qv_sigma on interpolated 74 sigma levels',
                              'kg/kg')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'qc_sigma', 'QC_SIGMA',
                              'qc_sigma on interpolated 74 sigma levels',
                              'kg/kg')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'qi_sigma', 'QI_SIGMA',
                              'qi_sigma on interpolated 74 sigma levels',
                              'kg/kg')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'skt', 'SKT',
                              'skin temperature',
                              'K')  # Verified units!
    
    expand_dict_var_desc_unit(var_dict, 'sfc_reference_p', 'SFC_REFERENCE_P',
                              'The lowest pressure level',
                              'kg/kg')  # Verified units!
    
    return var_dict








    # def create_var_dict():
    #
    #     var_dict = dict()
    #     expand_dict_var_desc_unit(var_dict, 'p', 'p',
    #                               'reference pressure (changes every snapshot)',
    #                               'hPa')
    #
    #     expand_dict_var_desc_unit(var_dict, 'p', 'p',
    #                               'reference pressure (changes every snapshot)',
    #                               'hPa')
    #
    #     var_dict[var1] = 'p'
    #     var_dict[var1 + desc1] = 'reference pressure (changes every snapshot)'
    #     var_dict[var1  +desc1] = 'hPa'
    #
    #     var1 = 'rho'
    #     var_dict[var1] = 'rho'
    #     var_dict[var1 + desc1] = 'reference density'
    #     var_dict['rho_units'] = 'kg/m^3'
    #
    #     var1 = 'u_coarse'
    #     var_dict[var1] = 'U'
    #     var_dict[var1 + desc1] = 'coarse grained zonal wind (c-grid)'
    #     var_dict['u_units'] = 'm/s'
    #
    #     var1 = 'v_coarse'
    #     var_dict[var1] = 'V'
    #     var_dict['v_desc'] = 'coarse grained meridional wind (c-grid)'
    #     var_dict['v_units'] = 'm/s'
    #
    #     var1 = 'w_coarse'
    #     var_dict[var1] = 'W'
    #     var_dict['w_desc'] = 'coarse grained vertical wind'
    #     var_dict['w_units'] = 'm/s'
    #
    #     var1 = 'tabs'
    #     var_dict[var1] = 'TABS'  # I think that this has differences due to time stepping (I think that originally tabs was saved after time stepping)
    #     var_dict['tabs_desc'] = 'coarse grained temperature (beginning of time step)'
    #     var_dict['tabs_units'] = 'K'
    #
    #     var1 = 'tabs_resolved_init'
    #     var_dict[var1] = 'TABS_RESOLVED_INIT'
    #     var_dict['tabs_resolved_init_desc'] = 'resolved temperature (beginning of time step)'
    #     var_dict['tabs_resolved_init_units'] = 'K'
    #
    #     var1 = 't_coarse'
    #     var_dict[var1] = 'T'
    #     var_dict['t_desc'] = 'coarse-grained modified energy variable (see Yuval and OGorman 2020)'
    #     var_dict['t_units'] = 'K'
    #
    #     var1 = 'tfull_coarse'
    #     var_dict[var1] = 'TFULL_INIT'
    #     var_dict['tfull_desc'] = 'coarse-grained liquid/ice water moist static energy'
    #     var_dict['tfull_units'] = 'K'
    #
    #     var1 = 'qv_coarse'
    #     var_dict[var1] = 'Q'  # verified that Q = QV_COARSE_INIT in the matlab file...
    #     var_dict['qv_desc'] = 'coarse-grained water vapor mixing ratio'
    #     var_dict['qv_units'] = 'g/kg'
    #
    #     var1 = 'qn_coarse_end'
    #     var_dict[var1] = 'QN'
    #     var_dict['qn_coarse_end_desc'] = 'coarse-grained cloud water+ice mixing ratio at end of time step'
    #     var_dict['qn_coarse_end_units'] = 'g/kg'
    #
    #     var1 = 'qn_coarse'
    #     var_dict[var1] = 'QN_COARSE_INIT'
    #     var_dict['qn_coarse_desc'] = 'coarse-grained cloud water+ice mixing ratio at beginning of time step'
    #     var_dict['qn_coarse_units'] = 'g/kg'
    #
    #     var1 = 'qn_resolved_init'
    #     var_dict[var1] = 'QN_RESOLVED_INIT'
    #     var_dict['qn_resolved_init_desc'] = 'resolved cloud water+ice mixing ratio at beginning of time step'
    #     var_dict['qn_resolved_init_units'] = 'g/kg'
    #
    #     var1 = 'qp_coarse'
    #     var_dict['qp_coarse'] = 'QP_COARSE_INIT'
    #     var_dict['qp_coarse_desc'] = 'coarse-grained preciptating water mixing ratio'
    #     var_dict['qp_coarse_units'] = 'g/kg'
    #
    #     var1 = 'Qrad_coarse'
    #     var_dict['Qrad_coarse'] = 'QRAD'
    #     var_dict['Qrad_coarse_desc'] = 'coarse-grained heating due to radiation'
    #     var_dict['Qrad_coarse_units'] = 'K/day'
    #
    #     #Scallar advection
    #     var1 = 'tfull_flux_x_out_coarse'
    #     var_dict['tfull_flux_x_out_coarse'] = 'TFULL_FLUX_COARSE_X'
    #     var_dict['tfull_flux_x_out_coarse_desc'] = 'coarse-grained zonal advective flux of liquid/ice water moist static energy'
    #     var_dict['tfull_flux_x_out_coarse_units'] = 'K kg/m^2/s'
    #
    #     var1 = 'tfull_flux_y_out_coarse'
    #     var_dict['tfull_flux_y_out_coarse'] = 'TFULL_FLUX_COARSE_Y'
    #     var_dict['tfull_flux_y_out_coarse_desc'] = 'coarse-grained meridional advective flux of liquid/ice water moist static energy'
    #     var_dict['tfull_flux_y_out_coarse_units'] = 'K kg/m^2/s'
    #
    #     var1 = 'tfull_flux_z_out_coarse'
    #     var_dict['tfull_flux_z_out_coarse'] = 'TFULL_FLUX_COARSE_Z'
    #     var_dict['tfull_flux_z_out_coarse_desc'] = 'coarse-grained vertical advective flux of liquid/ice water moist static energy'
    #     var_dict['tfull_flux_z_out_coarse_units'] = 'K kg/m^2/s'
    #
    #     var1 = 't_flux_x_out_coarse'
    #     var_dict['t_flux_x_out_coarse'] = 'T_FLUX_COARSE_X'
    #     var_dict['t_flux_x_out_coarse_desc'] = 'coarse-grained zonal advective flux of modified energy variable'
    #     var_dict['t_flux_x_out_coarse_units'] = 'K kg/m^2/s'
    #
    #     var1 = 't_flux_y_out_coarse'
    #     var_dict['t_flux_y_out_coarse'] = 'T_FLUX_COARSE_Y'
    #     var_dict['t_flux_y_out_coarse_desc'] = 'coarse-grained meridional advective flux of modified energy variable'
    #     var_dict['t_flux_y_out_coarse_units'] = 'K kg/m^2/s'
    #
    #     var1 = 't_flux_z_out_coarse'
    #     var_dict['t_flux_z_out_coarse'] = 'T_FLUX_COARSE_Z'
    #     var_dict['t_flux_z_out_coarse_desc'] = 'coarse-grained vertical advective flux of modified energy variable'
    #     var_dict['t_flux_z_out_coarse_units'] = 'K kg/m^2/s'
    #
    #     var1 = 'q_flux_x_out_coarse'
    #     var_dict['q_flux_x_out_coarse'] = 'QT_FLUX_COARSE_X'
    #     var_dict['q_flux_x_out_coarse_desc'] = 'coarse-grained zonal advective flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_flux_x_out_coarse_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'q_flux_y_out_coarse'
    #     var_dict['q_flux_y_out_coarse'] = 'QT_FLUX_COARSE_Y'
    #     var_dict['q_flux_y_out_coarse_desc'] = 'coarse-grained meridional advective flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_flux_y_out_coarse_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'q_flux_z_out_coarse'
    #     var_dict['q_flux_z_out_coarse'] = 'QT_FLUX_COARSE_Z'
    #     var_dict['q_flux_z_out_coarse_desc'] = 'coarse-grained vertical advective flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_flux_z_out_coarse_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'qp_flux_x_out_coarse'
    #     var_dict['qp_flux_x_out_coarse'] = 'QP_FLUX_COARSE_X'
    #     var_dict['qp_flux_x_out_coarse_desc'] = 'coarse-grained zonal meridional flux of precipitating water (qp) mixing raio'
    #     var_dict['qp_flux_x_out_coarse_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'qp_flux_y_out_coarse'
    #     var_dict['qp_flux_y_out_coarse'] = 'QP_FLUX_COARSE_Y'
    #     var_dict['qp_flux_y_out_coarse_desc'] = 'coarse-grained vertical meridional flux of precipitating water (qp) mixing raio'
    #     var_dict['qp_flux_y_out_coarse_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'qp_flux_z_out_coarse'
    #     var_dict['qp_flux_z_out_coarse'] = 'QP_FLUX_COARSE_Z'
    #     var_dict['qp_flux_z_out_coarse_desc'] = 'coarse-grained vertical advective flux of precipitating water (qp) mixing raio'
    #     var_dict['qp_flux_z_out_coarse_units'] = 'g/kg kg/m^2/s'
    #
    #     # subgrid Scallar advection
    #     var1 = 'tfull_flux_x_out_subgrid'
    #     var_dict['tfull_flux_x_out_subgrid'] = 'TFULL_FLUX_X'
    #     var_dict['tfull_flux_x_out_subgrid_desc'] = 'subgrid zonal advective flux of liquid/ice water moist static energy'
    #     var_dict['tfull_flux_x_out_coarse_units'] = 'K kg/m^2/s'
    #
    #     var1 = 'tfull_flux_y_out_subgrid'
    #     var_dict['tfull_flux_y_out_subgrid'] = 'TFULL_FLUX_Y'
    #     var_dict['tfull_flux_y_out_subgrid_desc'] = 'subgrid meridional advective flux of liquid/ice water moist static energy'
    #     var_dict['tfull_flux_y_out_coarse_units'] = 'K kg/m^2/s'
    #
    #     var1 = 'tfull_flux_z_out_subgrid'
    #     var_dict['tfull_flux_z_out_subgrid'] = 'TFULL_FLUX_Z'
    #     var_dict['tfull_flux_z_out_subgrid_desc'] = 'subgrid vertical advective flux of liquid/ice water moist static energy'
    #     var_dict['tfull_flux_z_out_coarse_units'] = 'K kg/m^2/s'
    #
    #     var1 = 't_flux_x_out_subgrid'
    #     var_dict['t_flux_x_out_subgrid'] = 'T_FLUX_X'
    #     var_dict['t_flux_x_out_subgrid_desc'] = 'subgrid zonal advective flux of modified energy variable'
    #     var_dict['t_flux_x_out_subgrid_units'] = 'K kg/m^2/s'
    #
    #     var1 = 't_flux_y_out_subgrid'
    #     var_dict['t_flux_y_out_subgrid'] = 'T_FLUX_Y'
    #     var_dict['t_flux_y_out_subgrid_desc'] = 'subgrid meridional advective flux of modified energy variable'
    #     var_dict['t_flux_y_out_subgrid_units'] = 'K kg/m^2/s'
    #
    #     var1 = 't_flux_z_out_subgrid'
    #     var_dict['t_flux_z_out_subgrid'] = 'T_FLUX_Z'
    #     var_dict['t_flux_z_out_subgrid_desc'] = 'subgrid vertical advective flux of modified energy variable'
    #     var_dict['t_flux_z_out_subgrid_units'] = 'K kg/m^2/s'
    #
    #     var1 = 'q_flux_x_out_subgrid'
    #     var_dict['q_flux_x_out_subgrid'] = 'QT_FLUX_X'
    #     var_dict['q_flux_x_out_subgrid_desc'] = 'subgrid zonal advective flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_flux_x_out_subgrid_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'q_flux_y_out_subgrid'
    #     var_dict['q_flux_y_out_subgrid'] = 'QT_FLUX_Y'
    #     var_dict['q_flux_y_out_subgrid_desc'] = 'subgrid meridional advective flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_flux_y_out_subgrid_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'q_flux_z_out_subgrid'
    #     var_dict['q_flux_z_out_subgrid'] = 'QT_FLUX_Z'
    #     var_dict['q_flux_z_out_subgrid_desc'] = 'subgrid vertical advective flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_flux_z_out_subgrid_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'qp_flux_x_out_subgrid'
    #     var_dict['qp_flux_x_out_subgrid'] = 'QP_FLUX_X'
    #     var_dict['qp_flux_x_out_subgrid_desc'] = 'subgrid zonal advective flux of precipitating water (qp) mixing raio'
    #     var_dict['qp_flux_x_out_subgrid_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'qp_flux_y_out_subgrid'
    #     var_dict['qp_flux_y_out_subgrid'] = 'QP_FLUX_Y'
    #     var_dict['qp_flux_y_out_subgrid_desc'] = 'subgrid meridional advective flux of precipitating water (qp) mixing raio'
    #     var_dict['qp_flux_y_out_subgrid_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'qp_flux_z_out_subgrid'
    #     var_dict['qp_flux_z_out_subgrid'] = 'QP_FLUX_Z'
    #     var_dict['qp_flux_z_out_subgrid_desc'] = 'subgrid vertical advective flux of precipitating water (qp) mixing raio'
    #     var_dict['qp_flux_z_out_subgrid_units'] = 'g/kg kg/m^2/s'
    #
    #     #diffusion
    #     var1 = 't_diff_flux_x_coarse'
    #     var_dict['t_diff_flux_x_coarse'] = 'T_DIFF_F_COARSE_X'
    #     var_dict['t_flux_x_out_coarse_desc'] = 'coarse-grained zonal diffusive flux of modified energy variable'
    #     var_dict['t_flux_x_out_coarse_units'] = 'K*m/s'
    #
    #     var1 = 't_diff_flux_y_coarse'
    #     var_dict['t_diff_flux_y_coarse'] = 'T_DIFF_F_COARSE_Y'
    #     var_dict['t_diff_flux_y_coarse_desc'] = 'coarse-grained meridional diffusive flux of modified energy variable'
    #     var_dict['t_diff_flux_y_coarse_units'] = 'K*m/s'
    #
    #     var1 = 't_diff_flux_z_coarse'
    #     var_dict['t_diff_flux_z_coarse'] = 'T_DIFF_F_COARSE_Z'
    #     var_dict['t_diff_flux_z_coarse_desc'] = 'coarse-grained vertical diffusive flux of modified energy variable'
    #     var_dict['t_diff_flux_z_coarse_units'] = 'K kg/m^2/s'
    #
    #     var1 = 'tfull_diff_flux_x_coarse'
    #     var_dict['tfull_diff_flux_x_coarse'] = 'TFULL_DIFF_F_COARSE_X'
    #     var_dict['tfull_diff_flux_x_coarse_desc'] = 'coarse-grained zonal diffusive flux of liquid/ice water moist static energy'
    #     var_dict['tfull_diff_flux_x_coarse_units'] = 'K*m/s'
    #
    #     var1 = 'tfull_diff_flux_y_coarse'
    #     var_dict['tfull_diff_flux_y_coarse'] = 'TFULL_DIFF_F_COARSE_Y'
    #     var_dict['tfull_diff_flux_y_coarse_desc'] = 'coarse-grained meridional diffusive flux of liquid/ice water moist static energy'
    #     var_dict['tfull_diff_flux_y_coarse_units'] = 'K*m/s'
    #
    #     var1 = 'tfull_diff_flux_z_coarse'
    #     var_dict['tfull_diff_flux_z_coarse'] = 'TFULL_DIFF_F_COARSE_Z'
    #     var_dict['tfull_diff_flux_z_coarse_desc'] = 'coarse-grained vertical diffusive flux of liquid/ice water moist static energy'
    #     var_dict['tfull_diff_flux_z_coarse_units'] = 'K kg/m^2/s'
    #
    #     var1 = 'q_diff_flux_x_coarse'
    #     var_dict['q_diff_flux_x_coarse'] = 'QT_DIFF_F_COARSE_X'
    #     var_dict['q_diff_flux_x_coarse_desc'] = 'coarse-grained  zonal diffusive flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_diff_flux_x_coarse_units'] = 'g/kg m/s'
    #
    #     var1 = 'q_diff_flux_y_coarse'
    #     var_dict['q_diff_flux_y_coarse'] = 'QT_DIFF_F_COARSE_Y'
    #     var_dict['q_diff_flux_y_coarse_desc'] = 'coarse-grained  meridional diffusive flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_diff_flux_y_coarse_units'] = 'g/kg m/s'
    #
    #     var1 = 'q_diff_flux_z_coarse'
    #     var_dict['q_diff_flux_z_coarse'] = 'QT_DIFF_F_COARSE_Z'
    #     var_dict['q_diff_flux_z_coarse_desc'] = 'coarse-grained  vertical diffusive flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_diff_flux_z_coarse_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'qp_diff_flux_x_coarse'
    #     var_dict['qp_diff_flux_x_coarse'] = 'QP_DIFF_F_COARSE_X'
    #     var_dict['qp_diff_flux_x_coarse_desc'] = 'coarse-grained  zonal diffusive flux of precipitating water (qp) mixing raio'
    #     var_dict['qp_diff_flux_x_coarse_units'] = 'g/kg m/s'
    #
    #     var1 = 'qp_diff_flux_y_coarse'
    #     var_dict['qp_diff_flux_y_coarse'] = 'QP_DIFF_F_COARSE_Y'
    #     var_dict['qp_diff_flux_z_coarse_desc'] = 'coarse-grained  meridional diffusive flux of precipitating water (qp) mixing raio'
    #     var_dict['qp_diff_flux_z_coarse_units'] = 'g/kg m/s'
    #
    #     var1 = 'qp_diff_flux_z_coarse'
    #     var_dict['qp_diff_flux_z_coarse'] = 'QP_DIFF_F_COARSE_Z'
    #     var_dict['qp_diff_flux_z_coarse_desc'] = 'coarse-grained  vertical diffusive flux of precipitating water (qp) mixing raio'
    #     var_dict['qp_diff_flux_z_coarse_units'] = 'g/kg kg/m^2/s'
    #
    #     #subgrid diffusion:
    #     var1 = 'tfull_diff_flux_x_subgrid'
    #     var_dict['tfull_diff_flux_x_subgrid'] = 'TFULL_DIFF_FLUX_X'
    #     var_dict['tfull_diff_flux_x_subgrid_desc'] = 'subgrid zonal diffusive flux of liquid/ice water moist static energy'
    #     var_dict['tfull_diff_flux_x_subgrid_units'] = 'K m/s'
    #
    #     var1 = 'tfull_diff_flux_y_subgrid'
    #     var_dict['tfull_diff_flux_y_subgrid'] = 'TFULL_DIFF_FLUX_Y'
    #     var_dict['tfull_diff_flux_y_subgrid_desc'] = 'subgrid meridional diffusive flux of liquid/ice water moist static energy'
    #     var_dict['tfull_diff_flux_y_subgrid_units'] = 'K m/s'
    #
    #     var1 = 'tfull_diff_flux_z_subgrid'
    #     var_dict['tfull_diff_flux_z_subgrid'] = 'TFULL_DIFF_FLUX_Z'
    #     var_dict['tfull_diff_flux_z_subgrid_desc'] = 'subgrid vertical diffusive flux of liquid/ice water moist static energy'
    #     var_dict['tfull_diff_flux_z_subgrid_units'] = 'K kg/m^2/s'
    #
    #     var1 = 't_diff_flux_x_subgrid'
    #     var_dict['t_diff_flux_x_subgrid'] = 'T_DIFF_FLUX_X'
    #     var_dict['t_diff_flux_x_subgrid_desc'] = 'subgrid zonal diffusive flux of modified energy variable'
    #     var_dict['t_diff_flux_x_subgrid_units'] = 'K*m/s'
    #
    #     var1 = 't_diff_flux_y_subgrid'
    #     var_dict['t_diff_flux_y_subgrid'] = 'T_DIFF_FLUX_Y'
    #     var_dict['t_diff_flux_y_subgrid_desc'] = 'subgrid meridional diffusive flux of modified energy variable'
    #     var_dict['t_diff_flux_y_subgrid_units'] = 'K*m/s'
    #
    #     var1 = 't_diff_flux_z_subgrid'
    #     var_dict['t_diff_flux_z_subgrid'] = 'T_DIFF_FLUX_Z'
    #     var_dict['t_diff_flux_z_subgrid_desc'] = 'subgrid vertical diffusive flux of modified energy variable'
    #     var_dict['t_diff_flux_z_subgrid_units'] = 'K kg/m^2/s'
    #
    #     var1 = 'q_diff_flux_x_subgrid'
    #     var_dict['q_diff_flux_x_subgrid'] = 'QT_DIFF_FLUX_X'
    #     var_dict['q_diff_flux_x_subgrid_desc'] = 'subgrid zonal diffusive flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_diff_flux_x_subgrid_units'] = 'g/kg m/s'
    #
    #     var1 = 'q_diff_flux_y_subgrid'
    #     var_dict['q_diff_flux_y_subgrid'] = 'QT_DIFF_FLUX_Y'
    #     var_dict['q_diff_flux_y_subgrid_desc'] = 'subgrid meridional diffusive flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_diff_flux_y_subgrid_units'] = 'g/kg m/s'
    #
    #     var1 = 'q_diff_flux_z_subgrid'
    #     var_dict['q_diff_flux_z_subgrid'] = 'QT_DIFF_FLUX_Z'
    #     var_dict['q_diff_flux_z_subgrid_desc'] = 'subgrid vertical diffusive flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_diff_flux_z_subgrid_units'] = 'g/kg kg/m^2/s'
    #
    #     var1 = 'qp_diff_flux_x_subgrid'
    #     var_dict['qp_diff_flux_x_subgrid'] = 'QP_DIFF_FLUX_X'
    #
    #     var1 = 'qp_diff_flux_y_subgrid'
    #     var_dict['qp_diff_flux_y_subgrid'] = 'QP_DIFF_FLUX_Y'
    #
    #     var1 = 'qp_diff_flux_z_subgrid'
    #     var_dict['qp_diff_flux_z_subgrid'] = 'QP_DIFF_FLUX_Z'
    #     var_dict['q_diff_flux_z_coarse_desc'] = 'subgrid vertical diffusive flux of non-precipitating water (qt) mixing raio'
    #     var_dict['q_diff_flux_z_coarse_units'] = 'g/kg kg/m^2/s'
    #
    #
    #     #Microphysics
    #     var1 = 'dqp_mic_coarse'
    #     var_dict['dqp_mic_coarse'] = 'DQP'
    #
    #     var1 = 'dqp_t_tend_coarse'
    #     var_dict['dqp_t_tend_coarse'] = 'DQP_T_TEND_COARSE'
    #
    #     var1 = 'dqp_mic_subgrid'
    #     var_dict['dqp_mic_subgrid'] = 'DQP_RESOLVED'
    #
    #     var1 = 'dqp_t_tend_subgrid'
    #     var_dict['dqp_t_tend_subgrid'] = 'DQP_T_TEND_RESOLVED'
    #
    #     #Sedimentation
    #     var1 = 'dq_sed_coarse'
    #     var_dict['dq_sed_coarse'] = 'QT_TEND_CLOUD_COARSE'
    #
    #     var1 = 'dt_sed_coarse'
    #     var_dict['dt_sed_coarse'] = 'LAT_HEAT_CLOUD_COARSE'
    #
    #     var1 = 'fz_coarse'
    #     var_dict['fz_coarse'] = 'CLOUD_FZ_FLUX_COARSE'
    #
    #     var1 = 'fzt_coarse'
    #     var_dict['fzt_coarse'] = 'CLOUD_FZT_FLUX_COARSE'
    #
    #     var1 = 'dq_sed_subgrid'
    #     var_dict['dq_sed_subgrid'] = 'QT_TEND_CLOUD_RES'
    #
    #     var1 = 'dt_sed_subgrid'
    #     var_dict['dt_sed_subgrid'] = 'LAT_HEAT_CLOUD_RES'
    #
    #     var1 = 'fz_subgrid'
    #     var_dict['fz_subgrid'] = 'CLOUD_FZ_FLUX_RES'
    #
    #     var1 = 'fzt_subgrid'
    #     var_dict['fzt_subgrid'] = 'CLOUD_FZT_FLUX_RES'
    #
    #     #fall
    #     var1 = 'qpfall_coarse'
    #     var_dict['qpfall_coarse'] = 'DQP_FALL_COARSE'
    #
    #     var1 = 'lat_heat_fall_coarse'
    #     var_dict['lat_heat_fall_coarse'] = 'T_FALL_COARSE'
    #
    #     var1 = 'precip_flux_coarse'
    #     var_dict['precip_flux_coarse'] = 'PRECIP'
    #
    #     var1 = 'qpfall_subgrid'
    #     var_dict['qpfall_subgrid'] = 'DQP_FALL_RES'
    #
    #     var1 = 'lat_heat_fall_subgrid'
    #     var_dict['lat_heat_fall_subgrid'] = 'T_FALL_RES'
    #
    #     var1 = 'precip_flux_subgrid'
    #     var_dict['precip_flux_subgrid'] = 'PRECIP_RESOLVED' \
    #                                       ''
    #     #diffusivity
    #     var1 = 'tkh_z_coarse'
    #     var_dict['tkh_z_coarse'] = 'TKZ_COARSE'
    #
    #     # momentum advection:
    #     var1 = 'dudt_coarse'
    #     var_dict['dudt_coarse'] = 'U_ADV_COARSE'
    #
    #     var1 = 'dvdt_coarse'
    #     var_dict['dvdt_coarse'] = 'V_ADV_COARSE'
    #
    #     var1 = 'dwdt_coarse'
    #     var_dict['dwdt_coarse'] = 'W_ADV_COARSE'
    #
    #     var1 = 'dudt_subgrid'
    #     var_dict['dudt_subgrid'] = 'U_ADV_RESOLVED'
    #
    #     var1 = 'dvdt_subgrid'
    #     var_dict['dvdt_subgrid'] = 'V_ADV_RESOLVED'
    #
    #     var1 = 'dwdt_subgrid'
    #     var_dict['dwdt_subgrid'] = 'W_ADV_RESOLVED'
    #
    #     var1 = 'dudt_coarse_collocated'
    #     var_dict['dudt_coarse_collocated'] = 'U_ADV_NORM_GRID_COARSE'
    #
    #     var1 = 'dvdt_coarse_collocated'
    #     var_dict['dvdt_coarse_collocated'] = 'V_ADV_NORM_GRID_COARSE'
    #
    #     var1 = 'dudt_subgrid_collocated'
    #     var_dict['dudt_subgrid_collocated'] = 'U_ADV_NORM_GRID_RESOLVED'
    #
    #     var1 = 'dudt_subgrid_u_grid'
    #     var_dict['dudt_subgrid_u_grid'] = 'U_ADV_U_GRID_RESOLVED'
    #
    #     var1 = 'dvdt_subgrid_collocated'
    #     var_dict['dvdt_subgrid_collocated'] = 'V_ADV_NORM_GRID_RESOLVED'
    #
    #     var1 = 'dvdt_subgrid_v_grid'
    #     var_dict['dvdt_subgrid_v_grid'] = 'V_ADV_V_GRID_RESOLVED'
    #
    #     # momentum surface fluxs.
    #     var1 = 'fluxbu_coarse'
    #     var_dict['fluxbu_coarse'] = 'SFLUX_U_COARSE'
    #
    #     var1 = 'fluxbv_coarse'
    #     var_dict['fluxbv_coarse'] = 'SFLUX_V_COARSE'
    #
    #     var1 = 'fluxbu_collocated_coarse'
    #     var_dict['fluxbu_collocated_coarse'] = 'U_SURF_FLUX_NORM_GRID_COARSE'
    #
    #     var1 = 'fluxbv_collocated_coarse'
    #     var_dict['fluxbv_collocated_coarse'] = 'V_SURF_FLUX_NORM_GRID_COARSE'
    #
    #     var1 = 'fluxbu_subgrid'
    #     var_dict['fluxbu_subgrid'] = 'SFLUX_U_RESOLVED'
    #
    #     var1 = 'fluxbv_subgrid'
    #     var_dict['fluxbv_subgrid'] = 'SFLUX_V_RESOLVED'
    #
    #     var1 = 'fluxbu_collocated_subgrid'
    #     var_dict['fluxbu_collocated_subgrid'] = 'U_SURF_FLUX_NORM_GRID_RESOLVED'
    #
    #     var1 = 'fluxbv_collocated_subgrid'
    #     var_dict['fluxbv_collocated_subgrid'] = 'V_SURF_FLUX_NORM_GRID_RESOLVED'
    #
    #     var1 = 'fluxbu_u_grid_subgrid'
    #     var_dict['fluxbu_u_grid_subgrid'] = 'U_SURF_FLUX_U_GRID_RESOLVED'
    #
    #     var1 = 'fluxbv_v_grid_subgrid'
    #     var_dict['fluxbv_v_grid_subgrid'] = 'V_SURF_FLUX_V_GRID_RESOLVED'
    #
    #     var1 = 'pp_coarse'
    #     var_dict['pp_coarse'] = 'PP'
    #
    #     var1 = 'u_coarse_collocated'
    #     var_dict['u_coarse_collocated'] = 'U_NORM_GRID'
    #
    #     var1 = 'v_coarse_collocated'
    #     var_dict['v_coarse_collocated'] = 'V_NORM_GRID'
    #
    #     #mass flux
    #     var1 = 'mass_flux_coarse'
    #     var_dict['mass_flux_coarse'] = 'UPDRAFT_01_TRESH'
    #
    #
    #
    #     return var_dict


def coarse_grain_spherical(field, mu, ady, terra, nx, ny, nz_init, nz_end, coarseness=12, terra_sum_lim=0):
    # including the effects of sphericity on the field
    # terra_sum_lim - if I want to limit the number of points I am relying on (minimum)
    field_coarse = np.zeros([nx, ny, nz_end - nz_init])
    #     print(field_coarse.shape)
    k_ind = -1
    for k in range(nz_init, nz_end):  # Coarse grained coordinates!
        k_ind = k_ind + 1
        #         print('loop')
        #         print(k)
        #         print(k_ind)
        for jj in range(ny):  # Coarse grained coordinates!
            for ii in range(nx):  # Coarse grained coordinates!
                factor = 0
                tmp = 0
                terra_sum = 0
                for j in range(coarseness * jj, coarseness * jj + coarseness):
                    for i in range(coarseness * ii, coarseness * ii + coarseness):
                        www = mu[j] * ady[j] * terra[i, j, k]
                        factor = factor + www
                        tmp = tmp + field[i, j, k_ind] * www
                        terra_sum = terra_sum + terra[i, j, k]
                if terra_sum > terra_sum_lim:
                    field_coarse[ii, jj, k_ind] = tmp / (factor + 1.e-10)
                else:
                    field_coarse[ii, jj, k_ind] = 0
    return field_coarse


def coarse_grain_spherical_no_y(field, mu, ady, terra, nx, ny, nz_init, nz_end, coarseness=12, terra_sum_lim=0):
    # At higher latitudes Marat did not coarse grain in the y direction!
    field_coarse = np.zeros([nx, ny, nz_end - nz_init])
    #     print(field_coarse.shape)
    k_ind = -1
    for k in range(nz_init, nz_end):  # Coarse grained coordinates!
        k_ind = k_ind + 1
        #         print('loop')
        #         print(k)
        #         print(k_ind)
        for jj in range(ny):  # Coarse grained coordinates!
            for ii in range(nx):  # Coarse grained coordinates!
                factor = 0
                tmp = 0
                terra_sum = 0
                #                 for j in range(coarseness*jj,coarseness*jj+coarseness):
                for i in range(coarseness * ii, coarseness * ii + coarseness):
                    www = terra[i, jj, k]
                    factor = factor + www
                    tmp = tmp + field[i, jj, k_ind] * www
                    terra_sum = terra_sum + terra[i, jj, k]
            if terra_sum > terra_sum_lim:
                field_coarse[ii, jj, k_ind] = tmp / (factor + 1.e-10)
            else:
                field_coarse[ii, jj, k_ind] = 0
    return field_coarse








def create_data_array_gsam(field, nc_name, description, units,
                               x_coarse, y_coarse,z,  dataset,
                      filename = 'none' , moisture_flag = False, test_mode = False):
    if np.isnan(field).any():
        raise Exception('there is a nan in ' + nc_name)

    if moisture_flag:
        fact = 1000.0 # To change the units of moisture related variables
    else:
        fact = 1.0
    if len(field.shape) == 3:
        array = xr.DataArray(
            data=np.moveaxis(field, (0, 1, 2), (2, 1, 0)).astype(np.float32) * fact,
            dims=['z', 'lat', 'lon'],
            coords=dict(
                z=('z', z),
                lat=('lat', y_coarse),
                lon=('lon', x_coarse),
            ),
            attrs=dict(
                description=description,
                units=units,
            )
        )
    elif len(field.shape) == 2:
        array = xr.DataArray(
            data=np.moveaxis(field, (0, 1), (1, 0)).astype(np.float32) * fact,
            dims=['lat', 'lon'],
            coords=dict(
                y=('lat', y_coarse),
                x=('lon', x_coarse),
            ),
            attrs=dict(
                description=description,
                units=units,
            )
        )
    elif len(field.shape) == 1:
        array = xr.DataArray(
            data=field.astype(np.float32) * fact,
            dims=['z'],
            coords=dict(
                z=('z', z),
            ),
            attrs=dict(
                description=description,
                units=units,
            )
        )
    # if test_mode: # Compare to matlab version
    #     readnc_flip_axis(nc_name, field*fact, filename, nc_name)

    dataset[nc_name] = array


# I think that this is how Marat obtained indices of coarse graining in y... Note that this is not 1 or 12 there are numbers in between...
def calc_y_ind(ady_glob,ny_high,ny_coarse,coarse_fact =12):
    # Calculate the y indices Marat used for coarse graining
    j = int(ny_high / 2)
    l = 0
    ind11 = ny_coarse/2
    j_start = np.zeros(ny_coarse)
    j_end = np.zeros(ny_coarse)
    while j < ny_high - 1: # I think ideally this should go until ny_high. However this does not work properly (even when ady_glob is calculated to have ny_high dimension)
        j_start[ind11] = j
        #     print(j)
        l = l + 1
        sum1 = 0
        for i in range(coarse_fact):
            sum1 = sum1 + ady_glob[j]
            j = j + 1
            if sum1 > coarse_fact or i == coarse_fact-1:
                j_end[ind11] = j
                ind11 = ind11 + 1
                break
    # print(l)

    j = int(ny_high / 2) - 1
    l = 0
    ind11 = int(ny_coarse / 2)-1
    while j > -1:
        j_end[ind11] = j
        #     print(j)
        l = l + 1
        sum1 = 0
        for i in range(coarse_fact):
            sum1 = sum1 + ady_glob[j]
            j = j - 1
            if sum1 > coarse_fact or i == coarse_fact-1:
                j_start[ind11] = j
                ind11 = ind11 - 1
                break

    j_start = j_start.astype(int)
    j_end = j_end.astype(int)
    return j_start,j_end


def calc_y_ind2(ady_glob, ny_high, ny_coarse, coarse_fact=12):
    # This seems to work well exept in the region that the y coarse is smaller than 12 and larger than 1
    # ady_glob should be:
    # # #Option 2:
    # yv_gl_glob_2 = latv_gl[:] * deg2rad * rad_earth / earth_factor  # I use lat possibly need to shift by 0.5 grid
    # ady_glob_2 = np.zeros(lat.shape[0])
    # for j in range(lat.shape[0]):
    #     ady_glob_2[j] = (yv_gl_glob_2[j + 1] - yv_gl_glob_2[j]) / dy

    #     j = int(ny_high / 2)
    #     l = 0
    #     ind11 = int(ny_coarse/2)
    j_start = np.zeros(ny_coarse)
    j_end = np.zeros(ny_coarse)

    j = int(ny_high / 2)
    l = 0
    ind11 = int(ny_coarse / 2) - 1
    while j > 0:
        if ind11 < 0:
            print('There is a mistake in indices')
            break
        j_end[ind11] = j
        #     print(j)
        l = l + 1
        sum1 = 0
        for i in range(coarse_fact):
            sum1 = sum1 + ady_glob[j]
            j = j - 1
            if sum1 > coarse_fact or i == coarse_fact - 1:
                j_start[ind11] = j
                ind11 = ind11 - 1
                break

    print(l)

    j_start[int(ny_coarse / 2)] = j_start[int(ny_coarse / 2) - 1] + coarse_fact
    j_end[int(ny_coarse / 2)] = j_end[int(ny_coarse / 2) - 1] + coarse_fact
    ll = 0
    for ii in range(int(ny_coarse / 2) + 1, int(ny_coarse)):
        ll = ll + 1
        j_start[ii] = j_start[ii - ll] + (j_end[int(ny_coarse / 2) - 1] - j_end[int(ny_coarse / 2) - 1 - ll])
        j_end[ii] = j_end[ii - ll] + (j_start[int(ny_coarse / 2) - 1] - j_start[int(ny_coarse / 2) - 1 - ll])

    j_start = j_start.astype(int)
    j_end = j_end.astype(int)
    return j_start, j_end




def terra_num_of_points(terra, nx, ny, nz_init, nz_end, j_start, j_end, coarseness=12):
    # Need to include the effects of sphericity on the field I think!
    #This outputs the number of points I relied on the computation... and also relative to the amount of points I should have...
    field_coarse = np.zeros([nx, ny, nz_end - nz_init])
    field_coarse2 = np.zeros([nx, ny, nz_end - nz_init])
    #     print(field_coarse.shape)
    k_ind = -1
    for k in range(nz_init, nz_end):  # Coarse grained coordinates!
        k_ind = k_ind + 1
        #         print('loop')
        #         print(k)
        #         print(k_ind)
        for jj in range(ny):  # Coarse grained coordinates!
            for ii in range(nx):  # Coarse grained coordinates!
                factor = 0
                tmp = 0
                terra_sum = 0
                for j in range(j_start[jj], j_end[jj]):
                    for i in range(coarseness * ii, coarseness * ii + coarseness):
                        #                         www = terra[i,j,k]
                        factor = factor + 1
                        #                         tmp = tmp+field[i,j,k_ind]*www
                        terra_sum = terra_sum + terra[i, j, k_ind]
                #                 if terra_sum > terra_sum_lim:
                #                     field_coarse[ii,jj,k_ind]=tmp/(factor+1.e-10)
                #                 else:
                #                     field_coarse[ii,jj,k_ind]=0
                field_coarse[ii, jj, k_ind] = terra_sum
                field_coarse2[ii, jj, k_ind] = factor

    return field_coarse, field_coarse2


def coarse_grain_spherical_full(field, mu, ady, terra, nx, ny, nz_init, nz_end, j_start, j_end, coarseness=12,
                                terra_sum_lim=0):
    #Trying to coarse grain like in SAM - currently I have an issue where the y coarse is smaller than 12 and larger than 1.
    field_coarse = np.zeros([nx, ny, nz_end - nz_init])
    #     print(field_coarse.shape)
    k_ind = -1
    for k in range(nz_init, nz_end):  # Coarse grained coordinates!
        k_ind = k_ind + 1
        #         print('loop')
        #         print(k)
        #         print(k_ind)
        for jj in range(ny):  # Coarse grained coordinates!
            for ii in range(nx):  # Coarse grained coordinates!
                factor = 0
                tmp = 0
                terra_sum = 0
                for j in range(j_start[jj], j_end[jj]):
                    for i in range(coarseness * ii, coarseness * ii + coarseness):
                        www = mu[j] * ady[j] * terra[i, j, k]
                        factor = factor + www
                        tmp = tmp + field[i, j, k_ind] * www
                        terra_sum = terra_sum + terra[i, j, k]
                if terra_sum > terra_sum_lim:
                    field_coarse[ii, jj, k_ind] = tmp / (factor + 1.e-10)
                else:
                    field_coarse[ii, jj, k_ind] = 0
    return field_coarse


def coarse_grain_topog_median_mean(field, mu, ady, nx, ny, j_start, j_end, coarseness=12, terra_sum_lim=0):
    # Coarse grain 2D field (topography)
    field_coarse = np.zeros([nx, ny])
    field_coarse_median = np.zeros([nx, ny])
    #     print(field_coarse.shape)
    for jj in range(ny):  # Coarse grained coordinates!
        for ii in range(nx):  # Coarse grained coordinates!
            field_coarse_median[ii, jj] = np.median(
                field[coarseness * ii:coarseness * ii + coarseness, j_start[jj]:j_end[jj]])
            tmp = 0
            factor = 0
            for j in range(j_start[jj], j_end[jj]):
                for i in range(coarseness * ii, coarseness * ii + coarseness):
                    www = mu[j] * ady[j]
                    factor = factor + www
                    tmp = tmp + field[i, j] * www

            field_coarse[ii, jj] = tmp / (factor + 1.e-10)
    return field_coarse, field_coarse_median


def calc_y_ind_edge_processor(ady_glob, ny_high, ny_coarse, ny_coarse_proc=29, coarse_fact=12, processor_points=96):
    # Calculate the y indices Marat used for coarse graining
    j = ny_high - processor_points
    l = 0
    ind11 = 0
    j_start = np.zeros(ny_coarse_proc)
    j_end = np.zeros(ny_coarse_proc)
    sum1 = 0
    while j < ny_high:  # I think ideally this should go until ny_high. However this does not work properly (even when ady_glob is calculated to have ny_high dimension)
        j_start[ind11] = j
        #     print(j)
        l = l + 1
        sum1 = 0
        for i in range(coarse_fact):
            sum1 = sum1 + ady_glob[j]
            j = j + 1
            if sum1 > coarse_fact or i == coarse_fact - 1:
                j_end[ind11] = j
                ind11 = ind11 + 1
                break
    print(ind11)
    j_start_tot = np.zeros(ny_coarse)
    j_end_tot = np.zeros(ny_coarse)

    j_start_tot[ny_coarse - ny_coarse_proc:] = j_start
    j_end_tot[ny_coarse - ny_coarse_proc:] = j_end
    j_start_tot[0] = 0
    j_end_tot[0] = 1

    ll = 0
    for ii in range(1, ny_coarse_proc):
        ll = ll + 1
        j_start_tot[ii] = j_start_tot[ii - 1] + (j_end[-ll] - j_end[-1 - ll])
        j_end_tot[ii] = j_end_tot[ii - 1] + (j_start[-ll] - j_start[-ll - 1])
    #         print(j_end[-ll] - j_end[-1-ll])
    #     print(j_start_tot)

    for i in range(ny_coarse_proc, ny_coarse - ny_coarse_proc):
        j_start_tot[i] = j_start_tot[i - 1] + 12
        j_end_tot[i] = j_end_tot[i - 1] + 12

    j_start_tot = j_start_tot.astype(int)
    j_end_tot = j_end_tot.astype(int)
    return j_start_tot, j_end_tot#, j_start, j_end



def interpolation(data, p1, p_interp,axis_interp=0):
    #Python has such function- Need to check the edges... need to translate between z to sigma_reference
    f_out = interp1d(p1, data, axis=axis_interp,fill_value='extrapolate')
    return f_out(p_interp)
