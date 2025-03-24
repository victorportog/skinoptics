'''
| SkinOptics
| Copyright (C) 2024-2025 Victor Lima

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
| victorportog.github.io

| Release date:
| October 2024
| Last modification:
| October 2024

| References:

| [SJT95] Saidi, Jacques & Tittel 1995.
| Mie and Rayleigh modeling of visible-light scattering in neonatal skin.
| https://doi.org/10.1364/AO.34.007410

| [S*06] Salomatina, Jiang, Novak & Yaroslavsky 2006.
| Optical properties of normal and cancerous human skin in the visible and near-infrared spectral range.
| https://doi.org/10.1117/1.2398928

| [J13] Jacques 2013.
| Optical properties of biological tissues: a review.
| https://doi.org/10.1088/0031-9155/58/14/5007

| [N19] Niemz 2019.
| Laser-Tissue Interactions: Fundamentals and Applications (4th edition).
| https://doi.org/10.1007/978-3-030-11917-1
'''

import numpy as np
from scipy.interpolate import interp1d

from skinoptics.utils import *
from skinoptics.dataframes import *

def albedo(mua, mus):
    r'''
    | Calculate the optical albedo from the absorption coefficient and the scattering coefficient.
    | For details please check section 2.4 from Niemz 2019 [N19].
    
    :math:`a = \frac{\mu_s}{\mu_a + \mu_s}`
    
    :param mua: absorption coefficient [mm^-1]
    :type mua: float or np.ndarray
    
    :param mus: scattering coefficient [mm^-1]
    :type mus: float or np.ndarray
    
    :return: - **albedo** (*float or np.ndarray*) – optical albedo [-]
    '''
    
    return mus/(mua + mus)

def mus_from_rmus(rmus, g):
    r'''
    | Calculate the scattering coefficient from the reduced scattering coefficient and the 
    | anisotropy factor.
    
    :math:`\mu_s(\lambda) = \frac{\mu_s'(\lambda)}{1-g}`
    
    :param rmus: reduced scattering coefficient [mm^-1]
    :type rmus: float or np.ndarray
    
    :param g: anisotropy factor [-] (must be in the range [-1, 1])
    :type g: float
    
    :return: - **mus** (*float or np.ndarray*) – scattering coefficient [mm^-1]
    '''
    
    return rmus/(1 - g)

def rmus_from_mus(mus, g):
    r'''
    | Calculate the reduced scattering coefficient from the scattering coefficient and the 
    | anisotropy factor.
    
    :math:`\mu_s'(\lambda) = (1-g) \mbox{ } \mu_s(\lambda)`
    
    :param mus: scattering coefficient [mm^-1]
    :type mus: float or np.ndarray
    
    :param g: anisotropy factor [-] (must be in the range [-1, 1])
    :type g: float
    
    :return: - **rmus** (*float or np.ndarray*) – reduced scattering coefficient [mm^-1]
    '''
    
    return mus*(1 - g)

def rmus_EP_Salomatina(lambda0):
    r'''
    | The reduced scattering coefficient of human EPIDERMIS as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **rmus** (*float or np.ndarray*) – reduced scattering coefficient [mm^-1]
    '''
            
    return interp1d(np.array(EP_Salomatina_dataframe)[:,0], 
                    np.array(EP_Salomatina_dataframe)[:,3],
                    bounds_error = False,
                    fill_value = (np.array(EP_Salomatina_dataframe)[0,3],
                                  np.array(EP_Salomatina_dataframe)[-1,3]))(lambda0)

def rmus_DE_Salomatina(lambda0):
    r'''
    | The reduced scattering coefficient of human DERMIS as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2/>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **rmus** (*float or np.ndarray*) – reduced scattering coefficient [mm^-1]
    '''
    
    return interp1d(np.array(DE_Salomatina_dataframe)[:,0], 
                    np.array(DE_Salomatina_dataframe)[:,3],
                    bounds_error = False,
                    fill_value = (np.array(DE_Salomatina_dataframe)[0,3],
                                  np.array(DE_Salomatina_dataframe)[-1,3]))(lambda0)

def rmus_HY_Salomatina(lambda0):
    r'''
    | The reduced scattering coefficient of human HYPODERMIS as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2/>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **rmus** (*float or np.ndarray*) – reduced scattering coefficient [mm^-1]
    '''
            
    return interp1d(np.array(HY_Salomatina_dataframe)[:,0], 
                    np.array(HY_Salomatina_dataframe)[:,3],
                    bounds_error = False,
                    fill_value = (np.array(HY_Salomatina_dataframe)[0,3],
                                  np.array(HY_Salomatina_dataframe)[-1,3]))(lambda0)

def rmus_iBCC_Salomatina(lambda0):
    r'''
    | The reduced scattering coefficient of INFILTRATIVE BASAL CELL CARCINOMA (iBCC)
    | as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **rmus** (*float or np.ndarray*) – reduced scattering coefficient [mm^-1]
    '''
            
    return interp1d(np.array(iBCC_Salomatina_dataframe)[:,0], 
                    np.array(iBCC_Salomatina_dataframe)[:,3],
                    bounds_error = False,
                    fill_value = (np.array(iBCC_Salomatina_dataframe)[0,3],
                                  np.array(iBCC_Salomatina_dataframe)[-1,3]))(lambda0)

def rmus_nBCC_Salomatina(lambda0):
    r'''
    | The reduced scattering coefficient of NODULAR BASAL CELL CARCINOMA (nBCC)
    | as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **rmus** (*float or np.ndarray*) – reduced scattering coefficient [mm^-1]
    '''
    
    return interp1d(np.array(nBCC_Salomatina_dataframe)[:,0], 
                    np.array(nBCC_Salomatina_dataframe)[:,3],
                    bounds_error = False,
                    fill_value = (np.array(nBCC_Salomatina_dataframe)[0,3],
                                  np.array(nBCC_Salomatina_dataframe)[-1,3]))(lambda0)

def rmus_SCC_Salomatina(lambda0):
    r'''
    | The reduced scattering coefficient of SQUAMOUS CELL CARCINOMA (SCC)
    | as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **rmus** (*float or np.ndarray*) – reduced scattering coefficient [mm^-1]
    '''
            
    return interp1d(np.array(SCC_Salomatina_dataframe)[:,0],
                    np.array(SCC_Salomatina_dataframe)[:,3],
                    bounds_error = False,
                    fill_value = (np.array(SCC_Salomatina_dataframe)[0,3],
                                  np.array(SCC_Salomatina_dataframe)[-1,3]))(lambda0)

def std_rmus_EP_Salomatina(lambda0):
    r'''
    | The standard deviation respective to :meth:`skinoptics.scattering_coefficient.rmus_EP_Salomatina`.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **std_rmus** (*float or np.ndarray*) – standard deviation of the reduced scattering coefficient [mm^-1]
    '''
    
    return interp1d(np.array(EP_Salomatina_dataframe)[:,0], 
                    np.array(EP_Salomatina_dataframe)[:,4],
                    bounds_error = False,
                    fill_value = (np.array(EP_Salomatina_dataframe)[0,4],
                                  np.array(EP_Salomatina_dataframe)[-1,4]))(lambda0)

def std_rmus_DE_Salomatina(lambda0):
    r'''
    | The standard deviation respective to :meth:`skinoptics.scattering_coefficient.rmus_DE_Salomatina`.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **std_rmus** (*float or np.ndarray*) – standard deviation of the reduced scattering coefficient [mm^-1]
    '''
    
    return interp1d(np.array(DE_Salomatina_dataframe)[:,0], 
                    np.array(DE_Salomatina_dataframe)[:,4],
                    bounds_error = False,
                    fill_value = (np.array(DE_Salomatina_dataframe)[0,4],
                                  np.array(DE_Salomatina_dataframe)[-1,4]))(lambda0)

def std_rmus_HY_Salomatina(lambda0):
    r'''
    | The standard deviation respective to :meth:`skinoptics.scattering_coefficient.rmus_HY_Salomatina`.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **std_rmus** (*float or np.ndarray*) – standard deviation of the reduced scattering coefficient [mm^-1]
    '''
    
    return interp1d(np.array(HY_Salomatina_dataframe)[:,0], 
                    np.array(HY_Salomatina_dataframe)[:,4],
                    bounds_error = False,
                    fill_value = (np.array(HY_Salomatina_dataframe)[0,4],
                                  np.array(HY_Salomatina_dataframe)[-1,4]))(lambda0)

def rmus_Ray(lambda0, A):
    r'''
    | The reduced scattering coefficient as a function of wavelength, Rayleigh scattering only.
    | For details please check Jacques 2013 [J13].
    
    :math:`\mu_s'(\lambda) = A \mbox{ } \lambda^{-4}`
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray

    :param A: parameter :math:`A` [mm^-1 nm^4] (must be nonnegative)
    :type A: float
    
    :return: - **rmus** (*float or np.ndarray*) – reduced scattering coefficient [mm^-1]
    '''
    
    if A < 0:
        msg = 'The input A = {} is not valid.'.format(A)
        raise Exception(msg)
    
    return A*np.power(lambda0, -4, dtype = 'float64')
    
def rmus_Mie(lambda0, B, b):
    r'''
    | The reduced scattering coefficient as a function of wavelength, Mie scattering only.
    | For details please check Jacques 2013 [J13].
    
    :math:`\mu_s'(\lambda) = B \mbox{ } \lambda^{-b}`
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :param B: parameter :math:`B` [mm^-1 nm^b] (must be nonnegative)
    :type B: float
    
    :param b: Mie scattering power [-] (must be nonnegative)
    :type b: float

    :return: - **rmus** (*float or np.ndarray*) – reduced scattering coefficient [mm^-1]
    '''
    
    if B < 0:
        msg = 'The input B = {} is not valid.'.format(B)
        raise Exception(msg)
    if b < 0:
        msg = 'The input b = {} is not valid.'.format(b)
        raise Exception(msg)
    
    return B*np.power(lambda0, -b, dtype = 'float64')

def rmus_Jacques(lambda0, a, f_Ray, b_Mie):
    r'''
    | The reduced scattering coefficient as a function of wavelength, assuming contributions from
    | both Rayleigh and Mie scattering.
    | For details please check Jacques 2013 [J13].
    
    :math:`\mu_s'(\lambda) = a\left[f_{Ray} \mbox{ } \left(\frac{\lambda}{500}\right)^{-4} + (1-f_{Ray})\left(\frac{\lambda}{500}\right)^{-b_{Mie}} \right]`
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :param a: parameter :math:`a` [mm^-1]
    :type a: float
    
    :param f_Ray: fraction of Rayleigh scattering contribution [-] (must be in the range [0, 1])
    :type f_Ray: float
    
    :param b_Mie: Mie scattering power [-] (must be nonnegative)
    :type b_Mie: float

    :return: - **rmus** (*float or np.ndarray*) – reduced scattering coefficient [mm^-1]
    '''
    
    if f_Ray < 0 or f_Ray > 1:
        msg = 'The input f_Ray = {} is not valid.'.format(f_Ray)
        raise Exception(msg)
    if b_Mie < 0:
        msg = 'The input b_Mie = {} is not valid.'.format(b_Mie)
        raise Exception(msg)
        
    return a*(rmus_Ray(lambda0 = lambda0/500., A = f_Ray) +
              rmus_Mie(lambda0 = lambda0/500., B = (1 - f_Ray), b = b_Mie))

def rmus_Saidi(lambda0, A_Mie, B_Ray):
    r'''
    | The reduced scattering coefficient of NEONATAL SKIN as a function of wavelength.
    | Saidi, Jacques & Tittel 1995 [SJT95] fit to their own experimental data.
    
    :math:`\mu_s'(\lambda) = A_{Mie} \mbox{ (} 9.843 \times 10^{-7} \times \lambda^2 - 1.745 \times 10^{-3} \times \lambda + 1) + B_{Ray} \mbox{ } \lambda^{-4}`
    
    | wavelength range: [450 nm, 750 nm]
    | gestational ages between 19 and 52 weeks
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :param A_Mie: parameter :math:`A_{Mie}` [mm^-1] (must be nonnegative)
    :type A_Mie: float
    
    :param B_Ray: parameter :math:`B_{Ray}` [mm^-1 nm^4] (must be nonnegative)
    :type B_Ray: float
    
    :return: - **rmus** (*float or np.ndarray*) – reduced scattering coefficient [mm^-1]
    '''
    
    if A_Mie < 0:
        msg = 'The input A_Mie = {} is not valid.'.format(A_Mie)
        raise Exception(msg)
    if B_Ray < 0:
        msg = 'The input B_Ray = {} is not valid.'.format(B_Ray)
        raise Exception(msg)
              
    return A_Mie*quadratic(x = lambda0, a = 9.843E-7, b = -1.745E-3, c = 1) + \
           rmus_Ray(lambda0 = lambda0, A = B_Ray)
