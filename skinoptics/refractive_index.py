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
| March 2025

| References:

| [HQ73] Hale & Querry 1973.
| Optical Constants of Water in the 200-nm to 200-μm Wavelength Region.
| https://doi.org/10.1364/AO.12.000555

| [S81] Segelstein's M.S. Thesis 1981.
| The complex refractive index of water.
| https://mospace.umsystem.edu/xmlui/handle/10355/11599

| [LLX00] Li, Lin & Xie 2000.
| Refractive index of human whole blood with different types in the visible and near-infrared ranges.
| https://doi.org/10.1117/12.388073

| [D*06] Ding, Lu, Wooden, Kragel & Hu 2006.
| Refractive indices of human skin tissues at eight wavelengths and estimated dispersion relations
| between 300 and 1600 nm.
| https://doi.org/10.1088/0031-9155/51/6/008

| [FM06] Friebel & Meinke 2006.
| Model function to calculate the refractive index of native hemoglobin in the wavelength range of
| 250–1100 nm dependent on concentration.
| https://doi.org/10.1364/AO.45.002838

| [YLT18] Yanina, Lazareva & Tuchin 2018.
| Refractive index of adipose tissue and lipid droplet measured in wide spectral and temperature ranges.
| https://doi.org/10.1364/AO.57.004839

| [M*21] Matiatou, Giannios, Koutsoumpos, Toutouzas, Zografos & Moutzouris 2021.
| Data on the refractive index of freshly-excised human tissues in the visible and near-infrared
| spectral range.
| https://doi.org/10.1016/j.rinp.2021.103833
'''

import numpy as np
from scipy.interpolate import interp1d

from skinoptics.utils import *
from skinoptics.dataframes import *

def n_Cauchy(lambda0, A, B, C, D):
    r'''
    The Cauchy's equation.
    
    :math:`n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4} + \frac{D}{\lambda^6}`
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :param A: coefficient :math:`A` [-]
    :type A: float
    
    :param B: coefficient :math:`B` [nm^2]
    :type B: float
    
    :param C: coefficient :math:`C` [nm^4]
    :type C: float
    
    :param D: coefficient :math:`D` [nm^6]
    :type D: float
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
    
    return A + B/np.power(lambda0, 2., dtype = 'float64') + C/np.power(lambda0, 4., dtype = 'float64') + D/np.power(lambda0, 6., dtype = 'float64')

def n_Cornu(lambda0, A, B, C):
    r'''
    The Cornu's equation.
    
    :math:`n(\lambda) = A + \frac{B}{(\lambda - C)}`

    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :param A: coefficient :math:`A` [-]
    :type A: float
    
    :param B: coefficient :math:`B` [nm]
    :type B: float
    
    :param C: coefficient :math:`C` [nm]
    :type C: float
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
    
    return A + B/(lambda0 - C)

def n_Conrady(lambda0, A, B, C):
    r'''
    The Conrady's equation.
    
    :math:`n(\lambda) = A + \frac{B}{\lambda} + \frac{C}{\lambda^{3.5}}`

    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :param A: coefficient :math:`A` [-]
    :type A: float
    
    :param B: coefficient :math:`B` [nm]
    :type B: float
    
    :param C: coefficient :math:`C` [nm^3.5]
    :type C: float
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
    
    return A + B/lambda0 + C/np.power(lambda0, 3.5, dtype = 'float64')

def n_Sellmeier(lambda0, A1, B1, A2, B2):
    r'''
    The Sellmeier's equation.
    
    :math:`n(\lambda) = \sqrt{1 + \frac{A_1 \mbox{ } \lambda^2}{\lambda^2 - B_1} + \frac{A_2\mbox{ } \lambda^2}{\lambda^2 - B_2}}`

    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :param A1: coefficient :math:`A_1` [-]
    :type A1: float
    
    :param B1: coefficient :math:`B_1` [nm^2]
    :type B1: float
    
    :param A2: coefficient :math:`A_2` [-]
    :type A2: float
    
    :param B2: coefficient :math:`B_2` [nm^2]
    :type B2: float
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
        
    lambda0_sqrd = np.power(lambda0, 2., dtype = 'float64')

    return np.sqrt(1 + A1*lambda0_sqrd/(lambda0_sqrd - B1) + A2*lambda0_sqrd/(lambda0_sqrd - B2))

def n_wat_Hale(lambda0):
    r'''
    | The refractive index of WATER as a function of wavelength.
    | Linear interpolation of data from Hale & Querry 1973 [HQ73].
    
    | wavelength range: [200 nm, 200 μm]
    | temperature: 25 ºC
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
        
    return interp1d(np.array(n_and_k_wat_Hale_dataframe)[:,0],
                    np.array(n_and_k_wat_Hale_dataframe)[:,2],
                    bounds_error = False,
                    fill_value = (np.array(n_and_k_wat_Hale_dataframe)[0,2],
                                  np.array(n_and_k_wat_Hale_dataframe)[-1,2]))(lambda0)

def n_wat_Segelstein(lambda0):
    r'''
    | The refractive index of WATER as a function of wavelength.
    | Linear interpolation of data from D. J. Segelstein's M.S. Thesis 1981 [S81] collected
    | by S. Prahl and publicly available at <https://omlc.org/spectra/water/abs/index.html>.
    
    | wavelength range: [10 nm, 10 m]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
            
    return interp1d(np.array(n_and_k_wat_Segelstein_dataframe)[:,0]*1E3, 
                    np.array(n_and_k_wat_Segelstein_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(n_and_k_wat_Segelstein_dataframe)[0,1],
                                  np.array(n_and_k_wat_Segelstein_dataframe)[-1,1]))(lambda0)

def n_EP_Ding(lambda0, model = 'Cauchy'):
    r'''
    | The refractive index of human EPIDERMIS as a function of wavelength.
    | Ding et al. 2006 [D*06]'s fits to their own experimental data.
    | Complementary data publicly available at <bmlaser.physics.ecu.edu/literature/lit.htm>.
    
    | wavelength range: [325 nm, 1557 nm]
    | temperature: 22 ºC
    | body location: abdomen and arm
    | volunteers info: 12 female patients (10 caucasians and 2 african americans),
    | phototypes I-III and V, 27-63 years old
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :param model: the user can choose one of the following... 'Cauchy', 'Cornu' or 'Conrady' (default to 'Cauchy')
    :type moderl: str
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
        
    if model == 'Cauchy':
        return n_Cauchy(lambda0, 1.4134, 7907.9596, -389.9784, 0)
    elif model == 'Cornu':
        return n_Cornu(lambda0, 1.4057, 13.3628, 162.7100)
    elif model == 'Conrady':
        return n_Conrady(lambda0, 1.3963, 25.0563, 6237909.3004)
    else:
        msg = 'The input model = {} is not valid.'.format(model)
        raise Exception(msg)

def n_DE_Ding(lambda0, model = 'Cauchy'):
    r'''
    | The refractive index of human DERMIS as a function of wavelength.
    | Ding et al. 2006 [D*06]'s fits to their own experimental data.
    | Complementary data publicly available at <bmlaser.physics.ecu.edu/literature/lit.htm>.
    
    | wavelength range: [325 nm, 1557 nm]
    | temperature: 22 ºC
    | body location: abdomen and arm
    | volunteers info: 12 female patients (10 caucasians and 2 african americans),
    | phototypes I-III and V, 27-63 years old
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray

    :param model: the user can choose one of the following... 'Cauchy', 'Cornu' or 'Conrady' (default to 'Cauchy')
    :type moderl: str
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
    
    if model == 'Cauchy':
        return n_Cauchy(lambda0, 1.3696, 3916.8026, -2558.7704, 0)
    elif model == 'Cornu':
        return n_Cornu(lambda0, 1.2573, 453.8263, -2874.5367)
    elif model == 'Conrady':
        return n_Conrady(lambda0, 1.3549, 17.8990, -3593764.4133)
    else:
        msg = 'The input model = {} is not valid.'.format(model)
        raise Exception(msg)
        
def n_HY_Matiatou(lambda0):
    r'''
    | The refractive index of human HYPODERMIS as a function of wavelength.
    | Matiatou et al. 2021 [M*21]'s fit to their own experimental data.
    
    :math:`n(\lambda) = 1.44909 + \frac{5099.42}{\lambda^2}`
    
    | wavelength range: [450 nm, 1551 nm]
    | temperature: 25 ºC
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
    
    return n_Cauchy(lambda0, 1.44909, 5099.42, 0, 0)

def n_AT_Matiatou(lambda0):
    r'''
    | The refractive index of human ADIPOSE TISSUE as a function of wavelength.
    | Matiatou et al. 2021 [M*21]'s fit to their own experimental data.
    
    :math:`n(\lambda) = 1.44933 + \frac{4908.37}{\lambda^2}`
    
    | wavelength range: [450 nm, 1551 nm]
    | temperature: 25 ºC
    | body location: abdomen
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
        
    return n_Cauchy(lambda0, 1.44933, 4908.37, 0, 0)
        
def n_AT_Yanina(lambda0):
    r'''
    | The refractive index of human ADIPOSE TISSUE as a function of wavelength.
    | Yanina, Lazareva & Tuchin 2018 [YLT18]'s fit to their own experimental data.
    
    :math:`n(\lambda) = \sqrt{1 + \frac{1.1236 \mbox{ } \lambda^2}{\lambda^2-10556.6963} + \frac{0.2725 \mbox{ } \lambda^2}{\lambda^2-1.8867\times 10^7}}`
    
    | wavelength range: [480 nm, 1550 nm]
    | temperature: 23 ºC
    | body location: abdomen
    | volunteers info: 10 biopsies, 5 men, 40 – 50 years old, 70 – 80 kg
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
            
    return n_Sellmeier(lambda0, A1 = 1.1236, B1 = 10556.6963, A2 = 0.2725, B2 = 1.8867E7)

def n_blo_Li(lambda0):
    r'''
    | The refractive index of human BLOOD as a function of wavelength.
    | Li, Lin & Xie 2000 [LLX00]'s fit to their own experimental data.
    
    :math:`n(\lambda) = 1.357 + \frac{6.9 \times 10^3}{\lambda^2} + \frac{7.6 \times 10^8}{\lambda^4}`
    
    | wavelength range: [370 nm, 850 nm]
    | temperature: 27 - 28 ºC
    | blood types: A, B and O
    | volunteers info: 9 healthy volunteers, chinese, male and female
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **n** (*float or np.ndarray*) – refractive index [-]
    '''
            
    return n_Cauchy(lambda0, 1.357, 6.9E3, 7.6E8, 0)

def beta_oxy_Friebel(lambda0):
    r'''
    | The specific refractive increment of OXYHEMOGLOBIN solutions as a function of wavelength.
    | Linear interpolation of experimental data from Friebel & Meinke 2006 [FM06].
    
    | wavelength range: [250 nm, 1100 nm]
    
    :param lambda0: wavelength [nm] (must be in the range [250., 1100.])
    :type lambda0: float or np.ndarray
    
    :return: - **beta** (*float or np.ndarray*) – specific refractive increment [dL/g]
    '''
    
    if isinstance(lambda0, np.ndarray) == True:
        if np.any(lambda0 < 250) or np.any(lambda0 > 1100):
            msg = 'At least one element in the input lambda0 is out of the range [250 nm, 1100 nm].'
            raise Exception(msg)
    else:
        if lambda0 < 250 or lambda0 > 1100:
            msg = 'The input lambda0 = {} nm is out of the range [250 nm, 1100 nm].'.format(lambda0)
            raise Exception(msg)
            
    return interp1d(np.array(beta_oxy_Friebel_dataframe)[:,0],
                    np.array(beta_oxy_Friebel_dataframe)[:,1])(lambda0)

def n_oxy_Friebel(lambda0, Cmass_oxy, n_wat_model = 'Segelstein'):
    '''
    | The refractive index of OXYHEMOGLOBIN solutions as a function of wavelength and 
    | oxyhemoglobin concentration.
    | Calculated from the specific refractive increment :meth:`skinoptics.refractive_index.beta_oxy_Friebel`
    | and the refractive index of water.
    
    :math:`n_{oxy}(\lambda, C_{oxy}) = n_{wat}(\lambda) \mbox{ } [\\beta (\lambda) \mbox{ } C_{oxy} + 1]`
    
    | wavelength range: [250 nm, 1100 nm]
    | concentration range: [0 g/dL, 28.7 g/dL]
    
    :param lambda0: wavelength [nm] (must be in the range [250., 1100.])
    :type lambda0: float or np.ndarray
    
    :param Cmass_oxy: oxyhemoglobin mass concentration [g/dL]
    :type Cmass_oxy: float
    
    :param n_wat_model: the user can choose one of the following... 'Hale' or 'Segelstein' (default to 'Segelstein')
    :type n_wat_model: str

    | 'Hale' refers to :meth:`skinoptics.refractive_index.n_wat_Hale`
    | 'Segelstein' refers to :meth:`skinoptics.refractive_index.n_wat_Segelstein`
    
    :return: - **n** (*float or array-like*) – refractive index [-]
    '''
    
    if Cmass_oxy < 0:
        msg = 'The input Cmass_oxy = {} g/dL is not valid.'.format(Cmass_oxy)
        raise Exception(msg)
    if Cmass_oxy > 28.7:
        msg = 'The input Cmass_oxy = {} g/dL is out of the concentration range [0 g/dL, 28.7 g/dL].'.format(Cmass_oxy)
        warnings.warn(msg) 
    
    if n_wat_model == 'Hale':
        n_wat_lambda0 = n_wat_Hale(lambda0)
    elif n_wat_model == 'Segelstein':
        n_wat_lambda0 = n_wat_Segelstein(lambda0)
    else:
        msg = 'The input n_wat_model = {} is not valid.'.format(n_wat_model)
        raise Exception(msg)
    
    return n_wat_lambda0*(beta_oxy_Friebel(lambda0)*Cmass_oxy + 1)
