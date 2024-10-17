'''
| SkinOptics
| Copyright (C) 2024 Victor Lima

    | This program is free software: you can redistribute it and/or modify
    | it under the terms of the GNU General Public License as published by
    | the Free Software Foundation, either version 3 of the License, or
    | (at your option) any later version.

    | This program is distributed in the hope that it will be useful,
    | but WITHOUT ANY WARRANTY; without even the implied warranty of
    | MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    | GNU General Public License for more details.

    | You should have received a copy of the GNU General Public License
    | along with this program.  If not, see <https://www.gnu.org/licenses/>.
     
| Victor Lima
| victorporto\@ifsc.usp.br
| victor.lima\@ufscar.br

| Release Date:
| October 2024
| Last Modification:
| October 2024

| References:

| [CCH91] Chardon, Cretois & Hourseau 1991.
| Skin colour typology and suntanning pathways.
| https://doi.org/10.1111/j.1467-2494.1991.tb00561.x

| [T*94] Takiwaki, Shirai, Kanno, Watanabe & Arase 1994.
| Quantification of erythema and pigmentation using a videomicroscope and a computer.
| https://doi.org/10.1111/j.1365-2133.1994.tb08462.x

| [F*96] Fullerton, Fischer, Lahti, Wilhelm, Takiwaki & Serup 1996.
| Guidetines for measurement of skin colour and erythema: A report from the Standardization Group of the European Society of Contact Dermatitis.
| https://doi.org/10.1111/j.1600-0536.1996.tb02258.x

| [S*96] Stokes, Anderson, Chandrasekar & Motta 1996.
| A Standard Default Color Space for the Internet - sRGB.
| https://www.w3.org/Graphics/Color/sRGB.html

| [IEC99] IEC 1999.
| Multimedia systems and equipment - Colour measurement and management - Part 2-1: Colour management - Default RGB colour space - sRGB.
| IEC 61966-2-1:1999

| [CIE04] CIE 2004.
| Colorimetry, 3rd edition.
| CIE 15:2004

| [D*06] Del Bino, Sok, Bessac & Bernerd 2006.
| Relationship between skin response to ultraviolet exposure and skin color type.
| https://doi.org/10.1111/j.1600-0749.2006.00338.x

| [S07] Schanda (editor) 2007.
| Colorimetry: Understanding the CIE System.
| http://dx.doi.org/10.1002/9780470175637

| [HP11] Hunt & Pointer 2011.
| Measuring Colour.
| https://doi.org/10.1002/9781119975595

| [DB13] Del Bino & Bernerd 2013.
| Variations in skin colour and the biological consequences of ultraviolet radiation exposure.
| https://doi.org/10.1111/bjd.12529

| [WSS13] Wyman, Sloan & Shirley 2013.
| Simple Analytic Approximations to the CIE XYZ Color Matching Functions.
| https://jcgt.org/published/0002/02/01/

| [CIE18a] CIE 2018.
| CIE standard illuminant A - 1 nm.
| https://doi.org/10.25039/CIE.DS.8jsxjrsn

| [CIE18b] CIE 2018.
| CIE standard illuminant D55.
| https://doi.org/10.25039/CIE.DS.qewfb3kp

| [CIE18c] CIE 2018.
| CIE standard illuminant D75.
| https://doi.org/10.25039/CIE.DS.9fvcmrk4

| [CIE19a] CIE 2019.
| CIE 1931 colour-matching functions, 2 degree observer.
| https://doi.org/10.25039/CIE.DS.xvudnb9b

| [CIE19b] CIE 2019.
| CIE 1964 colour-matching functions, 10 degree observer
| https://doi.org/10.25039/CIE.DS.sqksu2n5

| [L*20] Ly, Dyer, Feig, Chien & Del Bino 2020.
| Research Techniques Made Simple: Cutaneous Colorimetry: A Reliable Technique for Objective Skin Color Measurement.
| https://doi.org/10.1016/j.jid.2019.11.003

| [CIE22a] CIE 2022.
| CIE standard illuminant D50.
| https://doi.org/10.25039/CIE.DS.etgmuqt5

| [CIE22b] CIE 2022.
| CIE standard illuminant D65.
| https://doi.org/10.25039/CIE.DS.hjfjmt59
'''

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from skinoptics.utils import *
from skinoptics.dataframes import *

def rspd(lambda0, illuminant):
    r'''
    | The relative spectral power distribution S(:math:`\lambda`) of a chosen standard illuminant
    | as a function of wavelength.
    | Linear interpolation of data from CIE datasets [CIE18a] [CIE22a] [CIE18b] [CIE22b] [CIE18c].
    
    | wavelength range:
    | [300 nm, 830 nm] (at 1 nm intervals, for illuminant = 'A', 'D50' or 'D65')
    | or [300 nm, 780 nm] (at 5 nm intervals, for illuminant = 'D55' or 'D75')
    
    :param lambda0: wavelength [nm] (must be in range [300 nm, 830 nm] or [300 nm, 780 nm])
    :type lambda0: float or np.ndarray
    
    :param illuminant: the user can choose one of the following... 'A', 'D50', 'D55', 'D65' or 'D75'
    :type illuminant: str

    | 'A' refers to the CIE standard illuminant A
    | 'D50' refers to the CIE standard illuminant D50
    | 'D55' refers to the CIE standard illuminant D55
    | 'D65' refers to the CIE standard illuminant D65
    | 'D75' refers to the CIE standard illuminant D75
    
    :return: - **rspd** (*float or np.ndarray*) – relative spectral power distribution [-]
    '''
    
    if illuminant == 'A' or 'D50' or 'D65':
        if isinstance(lambda0, np.ndarray) == True:
            if np.any(lambda0 < 300) or np.any(lambda0 > 830):
                msg = 'At least one element in the input lambda0 is out of the range [300 nm, 830 nm].'
                raise Exception(msg)
        else:
            if lambda0 < 300 or lambda0 > 830:
                msg = 'The input lambda0 = {} nm is out of the range [300 nm, 830 nm].'.format(lambda0)
                raise Exception(msg)
    elif illuminant == 'D55' or 'D75':
        if isinstance(lambda0, np.ndarray) == True:
            if np.any(lambda0 < 300) or np.any(lambda0 > 780):
                msg = 'At least one element in the input lambda0 is out of the range [300 nm, 780 nm].'
                raise Exception(msg)
        else:
            if lambda0 < 300 or lambda0 > 780:
                msg = 'The input lambda0 = {} nm is out of the range [300 nm, 780 nm].'.format(lambda0)
                raise Exception(msg)
    
    if illuminant == 'A': 
        rspd = interp1d(np.array(rspds_A_D50_D65_dataframe)[:,0],
                        np.array(rspds_A_D50_D65_dataframe)[:,1])(lambda0)
    elif illuminant == 'D50':
        rspd = interp1d(np.array(rspds_A_D50_D65_dataframe)[:,0],
                        np.array(rspds_A_D50_D65_dataframe)[:,2])(lambda0)
    elif illuminant == 'D55':
        rspd = interp1d(np.array(rspds_D55_D75_dataframe)[:,0],
                        np.array(rspds_D55_D75_dataframe)[:,1])(lambda0)
    elif illuminant == 'D65':
        rspd = interp1d(np.array(rspds_A_D50_D65_dataframe)[:,0],
                        np.array(rspds_A_D50_D65_dataframe)[:,3])(lambda0)
    elif illuminant == 'D75':
        rspd = interp1d(np.array(rspds_D55_D75_dataframe)[:,0],
                        np.array(rspds_D55_D75_dataframe)[:,2])(lambda0)
    else:
        msg = 'The input illuminant = {} is not valid.'.format(illuminant)
        raise Exception(msg)
        
    return rspd 

def cmfs(lambda0, observer, cmfs_model = 'CIE'):
    r'''
    | The CIE color-matching functions :math:`\bar{x}(\lambda)`, :math:`\bar{y}(\lambda)` and :math:`\bar{z}(\lambda)` for a chosen standard observer
    | as a function of wavelength.
    
    | wavelength range: [360 nm, 830 nm] (at 1 nm intervals for cmfs_model = 'CIE')
    
    :param lambda0: wavelength [nm] (must be in range [360., 830.] for cmfs_model = 'CIE')
    :type lambda0: float or np.ndarray 
    
    :param observer: the user can choose one of the following... '2o' or '10o'
    :type observer: str
    
    :param cmfs_model: the user can choose one of the following... 'CIE', 'Wyman_singlelobe' or 'Wyman_multilobe' (default to 'CIE')
    :type cmfs_model: str

    | '2o' refers to the CIE 1931 2 degree standard observer
    | '10o' refers to the CIE 1964 10 degree standard observer

    | 'CIE' for the linear interpolation of data from CIE datasets [CIE19a] [CIE19b]
    | 'Wyman_singlelobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.1)
    | 'Wyman_multilobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.2)
    
    :return: - **xbar** (*float or np.ndarray*) – :math:`\bar{x}(\lambda`) color-matching function [-]
       - **ybar** (*float or np.ndarray*) – :math:`\bar{y}(\lambda`) color-matching function [-]
       - **zbar** (*float or np.ndarray*) – :math:`\bar{z}(\lambda`) color-matching function [-]
    '''
    
    if cmfs_model == 'CIE':
        if isinstance(lambda0, np.ndarray) == True:
            if np.any(lambda0 < 360) or np.any(lambda0 > 830):
                msg = 'At least one element in the input lambda0 is out of the range [360 nm, 830 nm].'
                raise Exception(msg)
        else:
            if lambda0 < 360 or lambda0 > 830:
                msg = 'The input lambda0 = {} nm is out of the range [360 nm, 830 nm].'.format(lambda0)
                raise Exception(msg)
        if observer == '2o':
            xbar = interp1d(np.array(cmfs_dataframe)[:,0],
                            np.array(cmfs_dataframe)[:,1])(lambda0)
            ybar = interp1d(np.array(cmfs_dataframe)[:,0],
                            np.array(cmfs_dataframe)[:,2])(lambda0)
            zbar = interp1d(np.array(cmfs_dataframe)[:,0],
                            np.array(cmfs_dataframe)[:,3])(lambda0)
        elif observer == '10o':
            xbar = interp1d(np.array(cmfs_dataframe)[:,0],
                            np.array(cmfs_dataframe)[:,4])(lambda0)
            ybar = interp1d(np.array(cmfs_dataframe)[:,0],
                            np.array(cmfs_dataframe)[:,5])(lambda0)
            zbar = interp1d(np.array(cmfs_dataframe)[:,0],
                            np.array(cmfs_dataframe)[:,6])(lambda0)
        else:
            msg = 'The input observer = {} is not valid.'.format(observer)
            raise Exception(msg)
    elif cmfs_model == 'Wyman_singlelobe':
        if observer == '2o':
            xbar = gaussian(lambda0, 1.065, 595.8, 33.33) \
            + gaussian(lambda0, 0.366, 446.8, 19.44)
            ybar = gaussian(np.log(lambda0), 1.014, np.log(556.3), 0.075)
            zbar = gaussian(np.log(lambda0), 1.839, np.log(449.8), 0.051)
        elif observer == '10o':
            xbar = mod_gaussian_Wyman(lambda0, 0.398, -570.1, 1014, 1250) \
            + mod_gaussian_Wyman(-lambda0, 1.132, -1338, 743.5, 234)
            ybar = gaussian(lambda0, 1.011, 556.1, 46.14)
            zbar = mod_gaussian_Wyman(lambda0, 2.06, 265.8, 180.4,32)
        else:
            msg = 'The input observer = {} is not valid.'.format(observer)
            raise Exception(msg)
    elif cmfs_model == 'Wyman_multilobe':
        if observer == '2o':
            coeffs = [[0.362, 1.056, -0.065, 0.821, 0.286, 0., 1.217, 0.681, 0.],
                      [442.0, 599.8, 501.1, 568.8, 530.9, 0., 437.0, 459.0, 0.],
                      [0.0624, 0.0264, 0.0490, 0.0213, 0.0613, 0., 0.0845, 0.0385, 0.],
                      [0.0374, 0.0323, 0.0382, 0.0247, 0.0322, 0., 0.0278, 0.0725, 0.]]
            if isinstance(lambda0, np.ndarray) == True:
                xbar, ybar, zbar = np.zeros((3,len(lambda0)))
                c = 0
                for j in range(len(lambda0)):
                    X, Y, Z = 0., 0., 0.  
                    for i in range(3):
                        X += piecewise_gaussian_Wyman(lambda0[j], coeffs[0][i], coeffs[1][i],
                                                      coeffs[2][i], coeffs[3][i])
                        Y += piecewise_gaussian_Wyman(lambda0[j], coeffs[0][i+3], coeffs[1][i+3],
                                                      coeffs[2][i+3], coeffs[3][i+3])
                        Z += piecewise_gaussian_Wyman(lambda0[j], coeffs[0][i+6], coeffs[1][i+6],
                                                      coeffs[2][i+6], coeffs[3][i+6])
                    xbar[j], ybar[j], zbar[j] = X, Y, Z
            elif isinstance(lambda0, (int, float)) == True: 
                X, Y, Z = 0., 0., 0. 
                for i in range(3):
                    X += piecewise_gaussian_Wyman(lambda0, coeffs[0][i], coeffs[1][i],
                                                  coeffs[2][i], coeffs[3][i])
                    Y += piecewise_gaussian_Wyman(lambda0, coeffs[0][i+3], coeffs[1][i+3],
                                                  coeffs[2][i+3], coeffs[3][i+3])
                    Z += piecewise_gaussian_Wyman(lambda0, coeffs[0][i+6], coeffs[1][i+6],
                                                  coeffs[2][i+6], coeffs[3][i+6])
                xbar, ybar, zbar = X, Y, Z
            else:
                msg = 'The input lambda0 must be int, float or np.ndarray.'
                raise Exception(msg)
        else:
            msg = 'The input observer = {} is not valid for cmfs_model = Wyman_multilobe.'.format(observer)
            raise Exception(msg) 
    else:
        msg = 'The input cmfs_model = {} is not valid.'.format(cmfs_model)
        raise Exception(msg) 
        
    return xbar, ybar, zbar

def xy_from_XYZ(X, Y, Z):
    r'''
    | Calculate CIE xy chromaticities from CIE XYZ coordinates.
    
    | :math:`x = \frac{X}{X + Y + Z}`
    | :math:`y = \frac{Y}{X + Y + Z}`

    :param X: X coordinate [-]
    :type X: float or np.ndarray
    
    :param Y: Y coordinate [-]
    :type Y: float or np.ndarray
    
    :param Z: Z coordinate [-]
    :type Z: float or np.ndarray
    
    :return: - **x** (*float or np.ndarray*) – x chromaticity [-]
       - **y** (*float or np.ndarray*) – y chromaticity [-]
    '''
    
    x = X/(X + Y + Z)
    y = Y/(X + Y + Z)
    
    return x, y

def XYZ_wp(illuminant, observer, cmfs_model = 'CIE', K = 1.):
    r'''
    The white point CIE XYZ coordinates for a chosen standard illuminant and standard observer.
    
    :param illuminant: the user can choose one of the following... 'A', 'D50', 'D55', 'D65' or 'D75'
    :type illuminant: str
    
    :param observer: the user can choose one of the following... '2o' or '10o'
    :type observer: str
    
    :param cmfs_model: the user can choose one of the following... 'CIE', 'Wyman_singlelobe' or 'Wyman_multilobe' (default to 'CIE')
    :type cmfs_model: str
    
    :param K: scaling factor (usually 1. or 100.) [-] (default to 1.)
    :type K: float

    | 'A' refers to the CIE standard illuminant A
    | 'D50' refers to the CIE standard illuminant D50
    | 'D55' refers to the CIE standard illuminant D55
    | 'D65' refers to the CIE standard illuminant D65
    | 'D75' refers to the CIE standard illuminant D75

    | '2o' refers to the CIE 1931 2 degree standard observer
    | '10o' refers to the CIE 1964 10 degree standard observer

    | 'CIE' for the linear interpolation of data from CIE datasets [CIE19a] [CIE19b]
    | 'Wyman_singlelobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.1)
    | 'Wyman_multilobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.2)
    
    | K = 1. for CIE XYZ coordinates in range [0, 1]
    | K = 100. for CIE XYZ coordinates in range [0, 100]
    
    :return: - **Xn** (*float*) – white point X coordinate [-]
       - **Yn** (*float*) – white point Y coordinate [-]
       - **Zn** (*float*) – white point Z coordinate [-]
    '''
    
    if illuminant == 'D55' or illuminant == 'D75':
        Xn, Yn, Zn = XYZ_from_spectrum(np.arange(360, 780, 1), np.ones(len(np.arange(360, 780, 1)))*100,
                                       lambda_max = 780, illuminant = illuminant, observer = observer, cmfs_model = cmfs_model, K = K)
    else:
        Xn, Yn, Zn = XYZ_from_spectrum(np.arange(360, 830, 1), np.ones(len(np.arange(360, 830, 1)))*100,
                                       illuminant = illuminant, observer = observer, cmfs_model = cmfs_model, K = K)
    
    return Xn, Yn, Zn

def xy_wp(illuminant, observer):
    r'''
    | The white point CIE xy chromaticities for a chosen standard illuminant and standard observer.
    | Calculated from the white point CIE XYZ coordinates (see function :meth:`skinoptics.colors.XYZ_wp`).
    
    :param illuminant: the user can choose one of the following... 'A', 'D50', 'D55', 'D65' or 'D75'
    :type illuminant: str
    
    :param observer: the user can choose one of the following... '2o' or '10o'
    :type observer: str

    | 'A' refers to the CIE standard illuminant A
    | 'D50' refers to the CIE standard illuminant D50
    | 'D55' refers to the CIE standard illuminant D55
    | 'D65' refers to the CIE standard illuminant D65
    | 'D75' refers to the CIE standard illuminant D75

    | '2o' refers to the CIE 1931 2 degree standard observer
    | '10o' refers to the CIE 1964 10 degree standard observer
    
    :return: - **xn** (*float*) – white point CIE x chromaticity [-]
       - **yn** (*float*) – white point CIE y chromaticity [-]
    '''
    
    Xn, Yn, Zn = XYZ_wp(illuminant = illuminant, observer = observer)
    xn, yn = xy_from_XYZ(Xn, Yn, Zn)
    
    return xn, yn

def transf_matrix_sRGB_linear_from_XYZ():
    r'''
    The transformation matrix employed to obtain linear sRGB coordinates from CIE XYZ coordinates.

    :math:`\mathcal{M} = 
    \begin{bmatrix}
    3.24062 & -1.5372 & -0.4986 \\
    -0.9689 & 1.8758 & 0.0415 \\
    0.0557 & -0.2040 & 1.0570
    \end{bmatrix}`

    :returns: - **M** (*np.ndarray*) – transformation matrix
    '''

    return np.array([[3.24062, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])

def nonlinear_corr_sRGB(u):
    r'''
    The nonlinear correction for sRGB coordinates.
    
    :math:`\gamma(u) =  
    \left \{ \begin{matrix}
    12.92 \mbox{ } u, & \mbox{if } u \le 0.0031308 \\
    1.055 \mbox{ } u^{1/2.4} - 0.055, & \mbox{if } u > 0.0031308 \\
    \end{matrix} \right.`
    
    :param u: linear R, G or B coordinate [-]
    :type u: float or np.ndarray
    
    :return: - **gamma** (*float or np.ndarray*) – nonlinear R, G or B coordinate [-]
    '''
    
    if isinstance(u, np.ndarray) == True:
        gamma = np.zeros(len(u))
        for i in range(len(u)):
            if u[i] <= 0.0031308:
                gamma[i] = 12.92*u[i]
            else:
                gamma[i] = 1.055*u[i]**(1./2.4) - 0.055
    elif isinstance(u, (int, float)) == True:
        if u <= 0.0031308:
            gamma = 12.92*u
        else:
            gamma = 1.055*u**(1./2.4) - 0.055
    else:
        msg = 'u must be int, float or np.ndarray.'
        raise Exception(msg)
        
    return gamma

def inv_nonlinear_corr_sRGB(u):
    r'''
    The inverse nonlinear correction for sRGB coordinates.
    
    :math:`\gamma^{-1}(u) =  
    \left \{ \begin{matrix}
    u/12.92, & \mbox{if } u \le 0.04045 \\
    [(u + 0.055)/1.055]^{2.4}, & \mbox{if } u > 0.04045 \\
    \end{matrix} \right.`
    
    :param u: nonlinear R, G or B coordinate [-]
    :type u: float or np.ndarray
    
    :return: - **inv_gamma** (*float or np.ndarray*) – linear R, G or B coordinate [-]
    '''
    
    if isinstance(u, np.ndarray) == True:
        inv_gamma = np.zeros(len(u))
        for i in range(len(u)):
            if u[i] <= 0.04045:
                inv_gamma[i] = u[i]/12.92
            else:
                inv_gamma[i] = ((u[i] + 0.055)/1.055)**(2.4)
    elif isinstance(u, (int, float)) == True:
        if u <= 0.04045:
            inv_gamma = u/12.92
        else:
            inv_gamma = ((u + 0.055)/1.055)**(2.4)
    else:
        msg = 'u must be int, float or np.ndarray.'
        raise Exception(msg)
        
    return inv_gamma

def sRGB_from_XYZ(X, Y, Z, K = 1., sRGB_scale = 'norm'):
    r'''
    | Calculate sRGB coordinates from CIE XYZ coordinates.
    | CIE XYZ coordinates must be for the standard illuminant D65 and the 2 degree standard observer.
    | For details please check Stokes et al. [S*96] and IEC [IEC99].
    
    :math:`\begin{bmatrix}
    R \\
    G \\
    B
    \end{bmatrix}
    =
    \begin{bmatrix}
    \gamma(R_{linear}) \\
    \gamma(G_{linear}) \\
    \gamma(B_{linear})
    \end{bmatrix}`

    in which

    :math:`\begin{bmatrix}
    R_{linear} \\
    G_{linear} \\
    B_{linear}
    \end{bmatrix}
    =
    \mathcal{M}
    \begin{bmatrix}
    X \\
    Y \\
    Z
    \end{bmatrix}`

    and

    :math:`\gamma(u) =  
    \left \{ \begin{matrix}
    12.92 \mbox{ } u, & \mbox{if } u \le 0.0031308 \\
    1.055 \mbox{ } u^{1/2.4} - 0.055, & \mbox{if } u > 0.0031308 \\
    \end{matrix} \right.`
    
    :param X: X coordinate [-]
    :type X: float or np.ndarray
    
    :param Y: Y coordinate [-]
    :type Y: float or np.ndarray
    
    :param Z: Z coordinate [-]
    :type Z: float or np.ndarray
    
    :param K: scaling factor (usually 1. or 100.) [-] (default to 1.)
    :type K: float
    
    :param sRGB_scale: the user can choose one of the following... 'norm' or '8bit'
    :type sRGB_scale:: str (default to 'norm')

    | K = 1. for CIE XYZ coordinates in range [0, 1]
    | K = 100. for CIE XYZ coordinates in range [0, 100]

    | 'norm' for sRGB coordinates in range [0,1] (normalized scale)
    | '8bit' for sRGB coordinates in range [0, 255] (8-bit scale)
    
    :return: - **R** (*float or np.ndarray*) – R coordinate [-]
       - **G** (*float or np.ndarray*) – G coordinate [-]
       - **B** (*float or np.ndarray*) – B coordinate [-]
    '''
    
    M = transf_matrix_sRGB_linear_from_XYZ()
    
    if isinstance(X, np.ndarray) == True and \
    isinstance(Y, np.ndarray) == True and \
    isinstance(Z, np.ndarray) == True:
        if len(X) - len(Y) != 0 or len(X) - len(Z) != 0:
            msg = 'X, Y and Z must have the same length.'
            raise Exception(msg)
        R, G, B = np.zeros((3, len(X)))
        for i in range(len(X)):
            R_linear, G_linear, B_linear = np.clip(np.matmul(M, np.array([X[i]/K, Y[i]/K, Z[i]/K])), 0, 1)
            R[i], G[i], B[i] = nonlinear_corr_sRGB(np.array([R_linear, G_linear, B_linear])) 
    elif isinstance(X, (int, float)) == True and \
    isinstance(Y, (int, float)) == True and \
    isinstance(Z, (int, float)): 
        R_linear, G_linear, B_linear = np.clip(np.matmul(M, np.array([X/K, Y/K, Z/K])), 0 , 1)
        R, G, B = nonlinear_corr_sRGB(np.array([R_linear, G_linear, B_linear])) 
    else:
        msg = 'X, Y and Z must be int, float or np.ndarray.'
        raise Exception(msg)
    
    if sRGB_scale == 'norm':
        pass
    elif sRGB_scale == '8bit':
        scaling = 255
        R, G, B = np.round(scaling*np.array([R, G, B]))
    else:
        msg = 'The input sRGB_scale = {} is not valid.'.format(sRGB_scale)
        raise Exception(msg)
        
    return R, G, B

def XYZ_from_sRGB(R, G, B, K = 1., sRGB_scale = 'norm'):
    r'''
    | Calculate CIE XYZ coordinates from sRGB coordinates.
    | The obtained CIE XYZ coordinates are respective to the standard illuminant D65 and the
    | 2 degree standard observer.
    | For details please check Stokes et al. [S*96] and IEC [IEC99].
    
    :math:`\begin{bmatrix}
    X \\
    Y \\
    Z
    \end{bmatrix}
    =
    \mathcal{M}^{-1}
    \begin{bmatrix}
    R_{linear} \\
    G_{linear} \\
    B_{linear}
    \end{bmatrix}`

    in which

    :math:`\begin{bmatrix}
    R_{linear} \\
    G_{linear} \\
    B_{linear}
    \end{bmatrix}
    =
    \begin{bmatrix}
    \gamma^{-1}(R) \\
    \gamma^{-1}(G) \\
    \gamma^{-1}(B)
    \end{bmatrix}`

    and

    :math:`\gamma^{-1}(u) =  
    \left \{ \begin{matrix}
    u/12.92, & \mbox{if } u \le 0.04045 \\
    [(u + 0.055)/1.055]^{2.4}, & \mbox{if } u > 0.04045 \\
    \end{matrix} \right.`
        
    :param R: R coordinate [-]
    :type R: float or np.ndarray
    
    :param G: G coordinate [-]
    :type G: float or np.ndarray
    
    :param B: B coordinate [-]
    :type B: float or np.ndarray
    
    :param K: scaling factor (usually 1. or 100.) [-] (default to 1.)
    :type K: float
    
    :param sRGB_scale: the user can choose one of the following... 'norm' or '8bit'
    :type sRGB_scale:: str (default to 'norm')

    | K = 1. for CIE XYZ coordinates in range [0, 1]
    | K = 100. for CIE XYZ coordinates in range [0, 100]

    | 'norm' for sRGB coordinates in range [0,1] (normalized scale)
    | '8bit' for sRGB coordinates in range [0, 255] (8-bit scale)
    
    :return: - **X** (*float or np.ndarray*) – X coordinate [-]
       - **Y** (*float or np.ndarray*) – Y coordinate [-]
       - **Z** (*float or np.ndarray*) – Z coordinate [-]
    '''

    if sRGB_scale == 'norm':
        pass
    elif sRGB_scale == '8bit':
        scaling = 255
        R, G, B = np.array([R, G, B])/scaling
    else:
        msg = 'The input sRGB_scale = {} is not valid.'.format(sRGB_scale) 
        raise Exception(msg)
    
    inv_M = np.round(np.linalg.inv(transf_matrix_sRGB_linear_from_XYZ()), 4)
    
    if isinstance(R, np.ndarray) == True and \
    isinstance(G, np.ndarray) == True and \
    isinstance(B, np.ndarray) == True:
        if len(R) - len(G) != 0 or len(R) - len(B) != 0:
            msg = 'R, G and B must have the same length.'
            raise Exception(msg)
        X, Y, Z = np.zeros((3, len(R)))
        for i in range(len(R)):
            R_linear, G_linear, B_linear  = inv_nonlinear_corr_sRGB(np.array([R[i], G[i], B[i]]))
            X[i], Y[i], Z[i] = np.matmul(inv_M, np.array([R_linear, G_linear, B_linear]))
    elif isinstance(R, (int, float)) == True and \
    isinstance(G, (int, float)) == True and \
    isinstance(B, (int, float)):
        R_linear, G_linear, B_linear  = inv_nonlinear_corr_sRGB(np.array([R, G, B]))
        X, Y, Z = np.matmul(inv_M, np.array([R_linear, G_linear, B_linear]))
    else:
        msg = 'R, G and B must be int, float or np.ndarray.'
        raise Exception(msg)
    
    return X*K, Y*K, Z*K

def f_Lab_from_XYZ(u):
    r'''
    | The function :math:`f(u)` used to calculate CIE L*a*b* coordinates from CIE XYZ coordinates
    | (see function :meth:`skinoptics.colors.Lab_from_XYZ`).
    
    :math:`f(u) = \left\{ 
    \begin{matrix}
    \sqrt[3]{u}, & \mbox{if }  u > \left(\frac{6}{29}\right)^3 \\
    \frac{1}{3}\left(\frac{29}{6}\right)^2 u + \frac{4}{29}, & \mbox{if }  u \le \left(\frac{6}{29}\right)^3
    \end{matrix}\right.`
    
    :param u: X/Xn, Y/Yn or Z/Zn ratio[-]
    :type u: float or np.ndarray
    
    :return: - **f** (*float or np.ndarray*) – evaluated function [-]
    '''
    
    delta = 6./29.
    if isinstance(u, np.ndarray) == True:
        f = np.zeros(len(u))
        for i in range(len(u)):
            if u[i] > delta**3.:
                f[i] = np.cbrt(u[i])
            else:
                f[i] = u[i]/3./delta**2. + 4./29.
    elif isinstance(u, (int, float)) == True:
        if u > delta**3.:
            f = np.cbrt(u)
        else:
            f = u/3./delta**2. + 4./29.
    else:
        msg = 'u must be int, float or np.ndarray.'
        raise Exception(msg)
        
    return f
    
def inv_f_Lab_from_XYZ(u):
    r'''
    The :math:`f^{-1}(u)` function, i.e. the inverse of the :math:`f(u)` function :meth:`skinoptics.colors.f_Lab_from_XYZ`.
    
    :math:`f^{-1}(u) = \left\{ 
    \begin{matrix}
    u^3, & \mbox{if }  u > \frac{6}{29} \\
    3 \mbox{ } \left(\frac{6}{29}\right)^2\left(u - \frac{4}{29} \right), & \mbox{if } u \le \frac{6}{29}
    \end{matrix}\right.`
        
    :param u: function variable [-]
    :type u: float or np.ndarray
    
    :return: - **f** (*float or np.ndarray*) – evaluated function [-]
    '''
    
    delta = 6./29.  
    if isinstance(u, np.ndarray) == True:
        inv_f = np.zeros(len(u))
        for i in range(len(u)):
            if u[i] > delta:
                inv_f[i] = u[i]**3
            else:
                inv_f[i] = 3.*delta**2.*(u[i] - 4./29.)
    elif isinstance(u, (int, float)) == True:
        if u > delta:
            inv_f = u**3
        else:
            inv_f = 3.*delta**2.*(u - 4./29.)
    else:
        msg = 'u must be int, float or np.ndarray.'
        raise Exception(msg)
        
    return inv_f

def Lab_from_XYZ(X, Y, Z, illuminant = 'D65', observer = '10o', K = 1.):
    r'''
    | Calculate CIE L*a*b* coordinates from CIE XYZ coordinates.
    | CIE XYZ and CIE L*a*b* coordinates must be for the same standard illuminant and standard observer.
    | For detailts please check CIE [CIE04], Schanda 2006 [S06] and Hunt & Pointer 2011 [HP11].
    
    | :math:`L^* = 116 \mbox{ } f(Y/Y_n) - 16`
    | :math:`a^* = 500 \mbox{ } [f(X/X_n) - f(Y/Y_n)]`
    | :math:`b^* = 200 \mbox{ } [f(Y/Y_n) - f(Z/Z_n)]`

    in which (:math:`X_n`, :math:`Y_n`, :math:`Z_n`) is the white point and 

    :math:`f(u) = \left\{ 
    \begin{matrix}
    \sqrt[3]{u}, & \mbox{if }  u > \left(\frac{6}{29}\right)^3 \\
    \frac{1}{3}\left(\frac{29}{6}\right)^2 u + \frac{4}{29}, & \mbox{if }  u \le \left(\frac{6}{29}\right)^3
    \end{matrix}\right.`
        
    :param X: X coordinate [-]
    :type X: float or np.ndarray
    
    :param Y: Y coordinate [-]
    :type Y: float or np.ndarray

    :param Z: Z coordinate [-]
    :type Z: float or np.ndarray
    
    :param illuminant: the user can choose one of the following... 'A', 'D50', 'D55', 'D65' or 'D75'
    :type illuminant: str
    
    :param observer: the user can choose one of the following... '2o' or '10o'
    :type observer: str

    :param K: scaling factor (usually 1. or 100.) [-] (default to 1.)
    :type K: float

    | 'A' refers to the CIE standard illuminant A
    | 'D50' refers to the CIE standard illuminant D50
    | 'D55' refers to the CIE standard illuminant D55
    | 'D65' refers to the CIE standard illuminant D65
    | 'D75' refers to the CIE standard illuminant D75

    | '2o' refers to the CIE 1931 2 degree standard observer
    | '10o' refers to the CIE 1964 10 degree standard observer
    
    | K = 1. for CIE XYZ coordinates in range [0, 1]
    | K = 100. for CIE XYZ coordinates in range [0, 100]
    
    :return: - **L** (*float or np.ndarray*) – L* coordinate [-]
       - **a** (*float or np.ndarray*) – a* coordinate [-]
       - **b** (*float or np.ndarray*) – b* coordinate [-]
    '''
    
    Xn, Yn, Zn = XYZ_wp(illuminant = illuminant, observer = observer, K = K)
    f = f_Lab_from_XYZ
    
    if isinstance(X, np.ndarray) == True and \
    isinstance(Y, np.ndarray) == True and \
    isinstance(Z, np.ndarray) == True:
        if len(X) - len(Y) != 0 or len(X) - len(Z) != 0:
            msg = 'X, Y and Z must have the same length.'
            raise Exception(msg)
        L, a, b = np.zeros((3, len(X)))
        for i in range(len(X)):
            L[i] = 116.*f(Y[i]/Yn) - 16.
            a[i] = 500.*(f(X[i]/Xn) - f(Y[i]/Yn))
            b[i] = 200.*(f(Y[i]/Yn) - f(Z[i]/Zn))
    elif isinstance(X, (int, float)) == True and \
    isinstance(Y, (int, float)) == True and \
    isinstance(Z, (int, float)):
        L = 116.*f(Y/Yn) - 16.
        a = 500.*(f(X/Xn) - f(Y/Yn))
        b = 200.*(f(Y/Yn) - f(Z/Zn))
    else:
        msg = 'X, Y and Z must be int, float or np.ndarray.'
        raise Exception(msg)
    
    return L, a, b
        
def XYZ_from_Lab(L, a, b, illuminant = 'D65', observer = '10o', K = 1.):
    r'''
    | Calculate CIE XYZ coordinates from CIE L*a*b* coordinates.
    | CIE XYZ and CIE L*a*b* coordinates must be for the same standard illuminant and standard observer.
    | For detailts please check CIE [CIE04], Schanda 2006 [S06] and Hunt & Pointer 2011 [HP11].
    
    | :math:`X = f^{-1}[(L^* + 16)/116 + a^*/500] \mbox{ } X_n`
    | :math:`Y = f^{-1}[(L^* + 16)/116] \mbox{ } Y_n`
    | :math:`Z = f^{-1}[(L^* + 16)/116 - b^*/200] \mbox{ } Z_n`

    in which (:math:`X_n`, :math:`Y_n`, :math:`Z_n`) is the white point and 

    :math:`f^{-1}(u) = \left\{ 
    \begin{matrix}
    u^3, & \mbox{if }  u > \frac{6}{29} \\
    3 \mbox{ } \left(\frac{6}{29}\right)^2\left(u - \frac{4}{29} \right), & \mbox{if } u \le \frac{6}{29}
    \end{matrix}\right.`
        
    :param L: L* coordinate [-] (must be in range [0, 100])
    :type L: float or np.ndarray
    
    :param a: a* coordinate [-]
    :type a: float or np.ndarray
    
    :param b: b* coordinate [-]
    :type b: float or np.ndarray
    
    :param illuminant: the user can choose one of the following... 'A', 'D50', 'D55', 'D65' or 'D75'
    :type illuminant: str
    
    :param observer: the user can choose one of the following... '2o' or '10o'
    :type observer: str

    :param K: scaling factor (usually 1. or 100.) [-] (default to 1.)
    :type K: float

    | 'A' refers to the CIE standard illuminant A
    | 'D50' refers to the CIE standard illuminant D50
    | 'D55' refers to the CIE standard illuminant D55
    | 'D65' refers to the CIE standard illuminant D65
    | 'D75' refers to the CIE standard illuminant D75

    | '2o' refers to the CIE 1931 2 degree standard observer
    | '10o' refers to the CIE 1964 10 degree standard observer
    
    | K = 1. for CIE XYZ coordinates in range [0, 1]
    | K = 100. for CIE XYZ coordinates in range [0, 100]
    
    :return: - **X** (*float or np.ndarray*) – X* coordinate [-]
       - **Y** (*float or np.ndarray*) – Y* coordinate [-]
       - **Z** (*float or np.ndarray*) – Z* coordinate [-]
    '''
    
    if isinstance(L, np.ndarray) == True:
        if np.any(L < 0) or np.any(L > 100):
            msg = 'At least one element in the input L is out of the range [0, 100].'
            raise Exception(msg)
    else:
        if L < 0 or L > 100:
            msg = 'The input L = {} is out of the range [0, 100].'.format(L)
            raise Exception(msg)
                
    Xn, Yn, Zn = XYZ_wp(illuminant = illuminant, observer = observer, K = K)
    inv_f = inv_f_Lab_from_XYZ
    
    if isinstance(L, np.ndarray) == True and \
    isinstance(a, np.ndarray) == True and \
    isinstance(b, np.ndarray) == True:
        if len(L) - len(a) != 0 or len(L) - len(b) != 0:
            msg = 'L, a and b must have the same length.'
            raise Exception(msg)
        X, Y, Z = np.zeros((3, len(L)))
        for i in range(len(L)):
            X[i] = Xn*inv_f((L[i] + 16.)/116. + a[i]/500.)
            Y[i] = Yn*inv_f((L[i] + 16.)/116.)
            Z[i] = Zn*inv_f((L[i] + 16.)/116. - b[i]/200.)
    elif isinstance(L, (int, float)) == True and \
    isinstance(a, (int, float)) == True and \
    isinstance(b, (int, float)):
        X = Xn*inv_f((L + 16.)/116. + a/500.)
        Y = Yn*inv_f((L + 16.)/116.)
        Z = Zn*inv_f((L + 16.)/116. - b/200.)
    else:
        msg = 'L, a and b must be float or np.ndarray.'
        raise Exception(msg)

    return X, Y, Z

def chroma(a, b):
    r'''
    Calculate the chroma C* from a* and b* coordinates.
    
    :math:`C^* = \sqrt{a^{*2} + b^{*2}}`
    
    :param a: a* coordinate [-]
    :type a: float or np.ndarray
    
    :param b: b* coordinate [-]
    :type b: float or np.ndarray
    
    :return: - **chroma** (*float or np.ndarray*) – chroma [-]
    '''
        
    return np.sqrt(a**2. + b**2.)

def hue(a, b):
    r'''
    Calculate the hue angle h* from a* and b* coordinates.
    
    :math:`h^* = \mbox{arctan2 } (b^*, a^*) \times \frac{180}{\pi}`
    
    :param a: a* coordinate [-]
    :type a: float or np.ndarray
    
    :param b: b* coordinate [-]
    :type b: float or np.ndarray
    
    :return: - **hue** (*float or np.ndarray*) – hue angle [degrees] (in range [0, 360])
    '''
    
    hue = np.arctan2(b,a)*180./np.pi

    hue_shape = hue.shape
    hue_flatten = hue.flatten()
    
    if isinstance(hue, np.ndarray) == True:
        for i in hue_flatten:
            if i < 0:
                i += 360
        hue = hue_flatten.reshape(hue_shape)
    elif isinstance(hue, (int, float)) == True:
        if hue < 0:
            hue += 360
    
    return hue

def ITA(L, b, L0 = 50.):
    r'''
    | Calculate the Individual Typology Angle (ITA) from L* and b* coordinates.
    | For details please check Chardon, Cretois & Hourseau 1991 [CCH91], Del Bino et al. 2006 [D*06],
    | Del Bino & Bernerd 2013 [DB13] and Ly et al. [L*20].
    
    :math:`\mbox{ITA} = \arctan\left(\frac{L^*-L_0^*}{b^*}\right) \times \frac{180}{\pi}`
    
    :param L: L* coordinate [-]
    :type L: float or np.ndarray
    
    :param b: b* coordinate [-]
    :type b: float or np.ndarray
    
    :param L0: L0 coordinate [-] (default to 50.)
    :type L0: float
    
    :return: - **ITA** (*float or np.ndarray*) – Individual Typology Angle [degrees]
    '''
    
    return np.arctan((L - L0)/b)*180./np.pi

def ITA_class(ITA):
    r'''
    | Skin color classification based on the Individual Typology Angle :meth:`skinoptics.colors.ITA`.
    | For details please check Chardon, Cretois & Hourseau 1991 [CCH91], Del Bino et al. 2006 [D*06],
    | Del Bino & Bernerd 2013 [DB13] and Ly et al. [L*20].
    
    +---------------------------+-----------------------------------------------+
    | skin color classification | ITA range                                     |
    +===========================+===============================================+
    | very light                | ITA :math:`> 55^\circ`                        |
    +---------------------------+-----------------------------------------------+
    | light                     | :math:`41^\circ <` ITA :math:`\le 55^\circ`   |
    +---------------------------+-----------------------------------------------+
    | intermediate              | :math:`28^\circ <` ITA :math:`\le 41^\circ`   |
    +---------------------------+-----------------------------------------------+
    | tan                       | :math:`10^\circ <` ITA :math:`\le 28^\circ`   |
    +---------------------------+-----------------------------------------------+
    | brown                     | :math:`-30^\circ <` ITA :math:`\le 10^\circ`  |
    +---------------------------+-----------------------------------------------+
    | dark                      | ITA :math:`\le -30^\circ`                     |
    +---------------------------+-----------------------------------------------+ 
    
    :param ITA: Individual Typology Angle [degrees] (must be greater than -90 and less than 90)
    :type ITA: float or np.ndarray
    
    :return: - **ITA_class** (*str or np.ndarray*) – skin color classification based on the Individual Typology Angle
    '''
    
    if isinstance(ITA, np.ndarray) == True:
        if np.any(ITA < -90) or np.any(ITA > 90):
            msg = 'At least one element in the input ITA is out of the range [-90, 90].'
            raise Exception(msg)
    else:
        if ITA < -90 or ITA > 90:
            msg = 'The input ITA = {} is out of the range [-90, 90].'.format(ITA)
            raise Exception(msg)
    
    if isinstance(ITA, np.ndarray) == True:
        ITA_class_list = ['']*len(ITA)
        for i in range(len(ITA)):
            if ITA[i] > 55:
                ITA_class_list[i] = 'very light'
            elif ITA[i] > 41 and ITA[i] <= 55:
                ITA_class_list[i] = 'light'
            elif ITA[i] > 28 and ITA[i] <= 41:
                ITA_class_list[i] = 'intermediate'
            elif ITA[i] > 10 and ITA[i] <= 28:
                ITA_class_list[i] = 'tan'
            elif ITA[i] > -30 and ITA[i] <= 10:
                ITA_class_list[i] = 'brown'
            else:
                ITA_class_list[i] = 'dark'
        ITA_class = np.array(ITA_class_list)
    else:
        if ITA > 55:
            ITA_class = 'very light'
        elif ITA > 41 and ITA <= 55:
            ITA_class = 'light'
        elif ITA > 28 and ITA <= 41:
            ITA_class = 'intermediate'
        elif ITA > 10 and ITA <= 28:
            ITA_class = 'tan'
        elif ITA > -30 and ITA <= 10:
            ITA_class = 'brown'
        else:
            ITA_class = 'dark'
    
    return ITA_class

def Delta_L(L0, L1):
    r'''
    Calculate the lightness difference :math:`\Delta L^*` between a reference color lightness :math:`L^*_0`
    and a test color lightness :math:`L^*_1`.
    
    :math:`\Delta L^* = L^*_1 - L^*_0`
    
    :param L0: reference color L* coordinate [-]
    :type L0: float or np.ndarray
    
    :param L1: test color L* coordinate [-]
    :type L1: float or np.ndarray
    
    :return: - **delta_L** (*float or np.ndarray*) – lightness difference [-]
    '''
    
    return L1 - L0

def Delta_a(a0, a1):
    r'''
    Calculate the difference :math:`\Delta a^*` between a reference color :math:`a^*_0` coordinate
    and a test color :math:`a^*_1` coordinate.
    
    :math:`\Delta a^* = a^*_1 - a^*_0`
    
    :param a0: reference color a* coordinate [-]
    :type a0: float or np.ndarray
    
    :param a1: test color a* coordinate [-]
    :type a1: float or np.ndarray
    
    :return: - **delta_a** (*float or np.ndarray*) – a* difference [-]
    '''
    
    return a1 - a0

def Delta_b(b0, b1):
    r'''
    Calculate the difference :math:`\Delta b^*` between a reference color :math:`b^*_0` coordinate
    and a test color :math:`b^*_1` coordinate.
    
    :math:`\Delta b^* = b^*_1 - b^*_0`
    
    :param b0: reference color b* coordinate [-]
    :type b0: float or np.ndarray
    
    :param b1: test color b* coordinate [-]
    :type b1: float or np.ndarray
    
    :return: - **delta_b** (*float or np.ndarray*) – b* difference [-]
    '''
    
    return b1 - b0

def Delta_E(L0, a0, b0, L1, a1, b1):
    r'''
    Calculate the color difference :math:`\Delta E^*` between between
    a reference color (:math:`L^*_0`, :math:`a^*_0`, :math:`b^*_0`) and
    a test color (:math:`L^*_1`, :math:`a^*_1`, :math:`b^*_1`).
    
    :math:`\Delta E^* = \sqrt{(L^*_1 - L^*_0)^2 + (a^*_1 - a^*_0)^2 + (b^*_1 - b^*_0)^2}`
    
    :param L0: reference color L* coordinate [-]
    :type L0: float or np.ndarray
    
    :param a0: reference color a* coordinate [-]
    :type a0: float or np.ndarray

    :param b0: reference color b* coordinate [-]
    :type b0: float or np.ndarray
    
    :param L1: test color L* coordinate [-]
    :type L1: float or np.ndarray
    
    :param a1: test color a* coordinate [-]
    :type a1: float or np.ndarray

    :param b1: test color b* coordinate [-]
    :type b1: float or np.ndarray
    
    :return: - **delta_E** (*float or np.ndarray*) – color difference [-]
    '''
    
    return np.sqrt(Delta_L(L0 = L0, L1 = L1)**2 + Delta_a(a0 = a0, a1 = a1)**2 + Delta_b(b0 = b0, b1 = b1)**2)

def EI(R_green, R_red):
    r'''
    | Calculate the Erythema Index (EI) from the reflectances on chosen green 
    | (usually approx. 568 nm) and red bands (usually approx. 655 nm).
    | For details please check Takiwaki et al. 1994 [T*94] and Fullerton et al. 1996 [F*96].
    
    :math:`\mbox{EI} = 100 \mbox{ } [\mbox{log}_{10}(R_\mbox{red}) - \mbox{log}_{10}(R_\mbox{green})]`
    
    :param R_green: reflectance on a chosen green band [%]
    :type R_green: float or np.ndarray
    
    :param R_red: reflectance on a chosen red band [%]
    :type R_red: float or np.ndarray
    
    :return: - **EI** (*float or np.ndarray*) – Erythema Index [-]
    '''
    
    return 100*(np.log10(R_red/100) - np.log10(R_green/100))

def MI(R_red):
    r'''
    | Calculate the Melanin Index (MI) from the reflectance on a chosen red band
    | (usually approx. 655 nm).
    | For details please check Takiwaki et al. 1994 [T*94] and Fullerton et al. 1996 [F*96].
    
    :math:`\mbox{MI} = 100 \mbox{ } [-\mbox{log}_{10}(R_\mbox{red})]`
    
    :param R_red: reflectance on a chosen red band [%]
    :type R_red: float or np.ndarray
    
    :return: - **MI** (*float or np.ndarray* – Melanin Index [-]
    '''
    
    return 100*(-np.log10(R_red/100))

def XYZ_from_spectrum(all_lambda, spectrum, lambda_min = 360., lambda_max = 830., lambda_step = 1.,
                      illuminant = 'D65', observer = '10o', cmfs_model = 'CIE', K = 1., interp1d_kind = 'cubic'):
    r'''
    | Calculate the CIE XYZ coordinates from the reflectance spectrum :math:`R(\lambda)` or the
    | transmittance spectrum :math:`T(\lambda)` for a chosen standard illuminant and standard observer.
    | Integration using the composite trapezoid rule from 360 nm to 830 nm (as default).
    | If the wavelength array does not cover the whole region, a constant extrapolation is perfomed.
    | For details please check CIE [CIE04] (see their section 7).
    
    | :math:`X = \frac{K}{N} \int_\lambda \mbox{ } R(\lambda) \mbox{ } S(\lambda) \mbox{ } \bar{x}(\lambda) \mbox{ } d\lambda`
    | :math:`Y = \frac{K}{N} \int_\lambda \mbox{ } R(\lambda) \mbox{ } S(\lambda) \mbox{ } \bar{y}(\lambda) \mbox{ } d\lambda`
    | :math:`Z = \frac{K}{N} \int_\lambda \mbox{ } R(\lambda) \mbox{ } S(\lambda) \mbox{ } \bar{z}(\lambda) \mbox{ } d\lambda`

    in which 

    | :math:`N = \int_\lambda \mbox{ } S(\lambda) \mbox{ } \bar{y}(\lambda) \mbox{ } d\lambda`
    
    The reflectance spectrum :math:`R(\lambda)` is replaced by the transmittance spectrum
    :math:`T(\lambda)` when dealing with color in some cases.
        
    :param all_lambda: wavelength array
    :type all_lambda: np.ndarray
    
    :param spectrum: reflectance or transmittance spectrum respective to the wavelength array [%]
    :type spectrum: np.ndarray
    
    :param lambda_min: lower limit of summation/integration (minimum wavelength to take into account) [nm] (default to 360.)
    :type lambda_min: float
    
    :param lambda_max: upper limit of summation/integration (maximum wavelength to take into account) [nm] (default to 830.)
    :type lambda_max: float
    
    :param lambda_step: summation interval (wavelength step) [nm] (default to 1.)
    :type lambda_step: float
    
     :param illuminant: the user can choose one of the following... 'A', 'D50', 'D55', 'D65' or 'D75'
    :type illuminant: str
    
    :param observer: the user can choose one of the following... '2o' or '10o'
    :type observer: str
    
    :param cmfs_model: the user can choose one of the following... 'CIE', 'Wyman_singlelobe' or 'Wyman_multilobe' (default to 'CIE')
    :type cmfs_model: str
    
    :param K: scaling factor (usually 1. or 100.) [-] (default to 1.)
    :type K: float

    :param interp1d_kind: kind argument of scipy.interpolation.interp1d (default to 'cubic' [CIE04] (see their section 7.2.1.1))
    :type interp1d_kind: str

    | 'A' refers to the CIE standard illuminant A
    | 'D50' refers to the CIE standard illuminant D50
    | 'D55' refers to the CIE standard illuminant D55
    | 'D65' refers to the CIE standard illuminant D65
    | 'D75' refers to the CIE standard illuminant D75

    | '2o' refers to the CIE 1931 2 degree standard observer
    | '10o' refers to the CIE 1964 10 degree standard observer

    | 'CIE' for the linear interpolation of data from CIE datasets [CIE19a] [CIE19b]
    | 'Wyman_singlelobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.1)
    | 'Wyman_multilobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.2)
    
    | K = 1. for CIE XYZ coordinates in range [0, 1]
    | K = 100. for CIE XYZ coordinates in range [0, 100]
    
    :return: - **X** (*float*) – X coordinate [-]
             - **Y** (*float*) – Y coordinate [-]
             - **Z** (*float*) – Z coordinate [-]
    '''
    
    x = np.arange(lambda_min, lambda_max + lambda_step, lambda_step)
    R_or_T_lambda = interp1d(all_lambda, spectrum/100, kind = interp1d_kind,
                             bounds_error = False, fill_value = (spectrum[0]/100, 
                                                                  spectrum[-1]/100))(x)
    S_lambda = rspd(x, illuminant = illuminant)
    xbar_lambda, ybar_lambda, zbar_lambda = cmfs(x, observer = observer, cmfs_model = cmfs_model)
    
    y0 = S_lambda*ybar_lambda
    N = trapezoid(y0, x = x, dx = lambda_step)
    
    R_or_T_lambda_times_S_lambda = R_or_T_lambda*S_lambda
    
    y1 = R_or_T_lambda_times_S_lambda*xbar_lambda
    X = K/N*trapezoid(y1, x = x, dx = lambda_step)
    
    y2 = R_or_T_lambda_times_S_lambda*ybar_lambda
    Y = K/N*trapezoid(y2, x = x, dx = lambda_step)
    
    y3 = R_or_T_lambda_times_S_lambda*zbar_lambda
    Z = K/N*trapezoid(y3, x = x, dx = lambda_step)
    
    return X, Y, Z

def sRGB_from_spectrum(all_lambda, spectrum, lambda_min = 360, lambda_max = 830, lambda_step = 1,
                       cmfs_model = 'CIE', interp1d_kind = 'cubic', sRGB_scale = 'norm'):
    r'''
    | Calculate the sRGB coordinates from the reflectance or the transmittance spectrum.
    | First calculate CIE XYZ coordinates (respective to the standard illuminant D65 and
    | the 2 degree standard observer) from the spectrum and then calculate sRGB coordinates
    | from CIE XYZ coordinates (see functions :meth:`skinoptics.colors.sRGB_from_XYZ` and
    | :meth:`skinoptics.colors.XYZ_from_spectrum`).

    :param all_lambda: wavelength array
    :type all_lambda: np.ndarray
    
    :param spectrum: reflectance or transmittance spectrum respective to the wavelength array [%]
    :type spectrum: np.ndarray
    
    :param lambda_min: lower limit of summation/integration (minimum wavelength to take into account) [nm] (default to 360.)
    :type lambda_min: float
    
    :param lambda_max: upper limit of summation/integration (maximum wavelength to take into account) [nm] (default to 830.)
    :type lambda_max: float
    
    :param lambda_step: summation interval (wavelength step) [nm] (default to 1.)
    :type lambda_step: float
    
    :param cmfs_model: the user can choose one of the following... 'CIE', 'Wyman_singlelobe' or 'Wyman_multilobe' (default to 'CIE')
    :type cmfs_model: str

    :param interp1d_kind: kind argument of scipy.interpolation.interp1d (default to 'cubic' [CIE04] (see their section 7.2.1.1))
    :type interp1d_kind: str

    :param sRGB_scale: the user can choose one of the following... 'norm' or '8bit' (default to 'norm')
    :type sRGB_scale: str

    | 'CIE' for the linear interpolation of data from CIE datasets [CIE19a] [CIE19b]
    | 'Wyman_singlelobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.1)
    | 'Wyman_multilobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.2)    

    | 'norm' for sRGB coordinates in range [0,1] (normalized scale)
    | '8bit' for sRGB coordinates in range [0, 255] (8-bit scale)
    
    :return: - **R** (*float*) – R coordinate [-]
       - **G** (*float*) – G coordinate [-]
       - **B** (*float*) – B coordinate [-]
    '''
    
    return sRGB_from_XYZ(*XYZ_from_spectrum(all_lambda = all_lambda, spectrum = spectrum,
                                            lambda_min = lambda_min, lambda_max = lambda_max,
                                            lambda_step = lambda_step,
                                            illuminant = 'D65', observer = '2o', cmfs_model = cmfs_model,
                                            K = 1., interp1d_kind = interp1d_kind),
                         K = 1., sRGB_scale = sRGB_scale)

def Lab_from_spectrum(all_lambda, spectrum, lambda_min = 360, lambda_max = 830, lambda_step = 1,
                     illuminant = 'D65', observer = '10o', cmfs_model = 'CIE', interp1d_kind = 'cubic'):
    r'''
    | Calculate the CIE L*a*b* coordinates from the reflectance or the transmittance spectrum.
    | First calculate CIE XYZ coordinates from the spectrum for a chosen standard illuminant
    | and standard observer and then calculate CIE L*a*b* coordinates from CIE XYZ coordinates
    | (see functions :meth:`skinoptics.colors.Lab_from_XYZ` and :meth:`skinoptics.colors.XYZ_from_spectrum`).
        
    :param all_lambda: wavelength array
    :type all_lambda: np.ndarray
    
    :param spectrum: reflectance or transmittance spectrum respective to the wavelength array [%]
    :type spectrum: np.ndarray
    
    :param lambda_min: lower limit of summation/integration (minimum wavelength to take into account) [nm] (default to 360.)
    :type lambda_min: float
    
    :param lambda_max: upper limit of summation/integration (maximum wavelength to take into account) [nm] (default to 830.)
    :type lambda_max: float
    
    :param lambda_step: summation interval (wavelength step) [nm] (default to 1.)
    :type lambda_step: float
    
     :param illuminant: the user can choose one of the following... 'A', 'D50', 'D55', 'D65' or 'D75'
    :type illuminant: str
    
    :param observer: the user can choose one of the following... '2o' or '10o'
    :type observer: str
    
    :param cmfs_model: the user can choose one of the following... 'CIE', 'Wyman_singlelobe' or 'Wyman_multilobe' (default to 'CIE')
    :type cmfs_model: str

    :param interp1d_kind: kind argument of scipy.interpolation.interp1d (default to 'cubic' [CIE04] (see their section 7.2.1.1))
    :type interp1d_kind: str

    | 'A' refers to the CIE standard illuminant A
    | 'D50' refers to the CIE standard illuminant D50
    | 'D55' refers to the CIE standard illuminant D55
    | 'D65' refers to the CIE standard illuminant D65
    | 'D75' refers to the CIE standard illuminant D75

    | '2o' refers to the CIE 1931 2 degree standard observer
    | '10o' refers to the CIE 1964 10 degree standard observer

    | 'CIE' for the linear interpolation of data from CIE datasets [CIE19a] [CIE19b]
    | 'Wyman_singlelobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.1)
    | 'Wyman_multilobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.2)
    
    :return: - **L** (*float*) – L* coordinate [-]
       - **a** (*float*) – a* coordinate [-]
       - **b** (*float*) – b* coordinate [-]
    '''

    return Lab_from_XYZ(*XYZ_from_spectrum(all_lambda = all_lambda, spectrum = spectrum,
                                           lambda_min = lambda_min, lambda_max = lambda_max,
                                           lambda_step = lambda_step,
                                           illuminant = illuminant, observer = observer, cmfs_model = cmfs_model,
                                           K = 1., interp1d_kind = interp1d_kind),
                        illuminant = illuminant, observer = observer, K = 1.)

def sRGB_from_lambda0(lambda0, cmfs_model = 'CIE', sRGB_scale = 'norm'):
    r'''
    | Calculate the sRGB coordinates respective to the color of a monochromatic light
    | (single wavelength).
    
    wavelength range: [360 nm, 830 nm]
    
    :param lambda0: wavelength of the monochromatic light [nm]
    :type lambda0: float or np.ndarray
    
    :param cmfs_model: the user can choose one of the following... 'CIE', 'Wyman_singlelobe' or 'Wyman_multilobe' (default to 'CIE')
    :type cmfs_model: str

    :param sRGB_scale: the user can choose one of the following... 'norm' or '8bit' (default to 'norm')
    :type sRGB_scale: str

    | 'CIE' for the linear interpolation of data from CIE datasets [CIE19a] [CIE19b]
    | 'Wyman_singlelobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.1)
    | 'Wyman_multilobe' for the functions from Wyman, Sloan & Shirley 2013 [WSS13] (section 2.2)    

    | 'norm' for sRGB coordinates in range [0,1] (normalized scale)
    | '8bit' for sRGB coordinates in range [0, 255] (8-bit scale)

    :return: - **R** (*float or np.ndarray*) – R coordinate [-]
       - **G** (*float or np.ndarray*) – G coordinate [-]
       - **B** (*float or np.ndarray*) – B coordinate [-]
    '''
    
    return sRGB_from_XYZ(*cmfs(lambda0, observer = '2o', cmfs_model = cmfs_model),
                         K = 1., sRGB_scale = sRGB_scale)
