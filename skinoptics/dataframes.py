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
| March 2025

| Example:
| Lab_Alaluf2002_dataframe (respective to datasets/colors/Lab_Alaluf2002.txt)

+------------------------------------------+------------------+------------------+------------------+
| group(body_location)                     |   L*(D65,10o)[-] |   a*(D65,10o)[-] |   b*(D65,10o)[-] |
+==========================================+==================+==================+==================+
| european(photoprotected_volar_upper_arm) |            65    |             7.1  |            13.3  |
+------------------------------------------+------------------+------------------+------------------+
| chinese(photoprotected_volar_upper_arm)  |            62.1  |             8.4  |            16.3  |
+------------------------------------------+------------------+------------------+------------------+
| mexican(photoprotected_volar_upper_arm)  |            61.9  |             8.3  |            15.3  |
+------------------------------------------+------------------+------------------+------------------+
| indian(photoprotected_volar_upper_arm)   |            53.7  |            10.3  |            17.9  |
+------------------------------------------+------------------+------------------+------------------+
| african(photoprotected_volar_upper_arm)  |            49.2  |            10.2  |            18.4  |
+------------------------------------------+------------------+------------------+------------------+
| european(photoexposed_dorsal_forearm)    |            53.94 |            12.57 |            19.23 |
+------------------------------------------+------------------+------------------+------------------+
| chinese(photoexposed_dorsal_forearm)     |            51.38 |            12.69 |            19.18 |
+------------------------------------------+------------------+------------------+------------------+
| mexican(photoexposed_dorsal_forearm)     |            50.42 |            12.69 |            19.58 |
+------------------------------------------+------------------+------------------+------------------+
| indian(photoexposed_dorsal_forearm)      |            43.71 |            12.89 |            17.15 |
+------------------------------------------+------------------+------------------+------------------+
| african(photoexposed_dorsal_forearm)     |            38.14 |            12.66 |            15.04 |
+------------------------------------------+------------------+------------------+------------------+
'''

import os
import pandas as pd

folder0 = os.path.dirname(os.path.abspath(__file__))
folder1 = os.path.join(folder0, 'datasets', 'optical_properties')

ext_and_molarext_eum_Sarna_dataframe = pd.read_csv(os.path.join(folder1, 'ext_and_molarext_eum_Sarna.txt'), sep = ' ') 
ext_and_molarext_phe_Sarna_dataframe = pd.read_csv(os.path.join(folder1, 'ext_and_molarext_phe_Sarna.txt'), sep = ' ')
ext_and_molarext_oxy_and_deo_Prahl_dataframe = pd.read_csv(os.path.join(folder1, 'ext_and_molarext_oxy_and_deo_Prahl.txt'), sep = ' ')
molarext_bil_Li_dataframe = pd.read_csv(os.path.join(folder1, 'molarext_bil_Li.txt'), sep = ' ')

mua_wat_Hale_dataframe = pd.read_csv(os.path.join(folder1, 'mua_wat_Hale.txt'), sep = ' ')
mua_wat_Segelstein_dataframe = pd.read_csv(os.path.join(folder1, 'mua_wat_Segelstein.txt'), sep = ' ')
mua_fat_vanVeen_dataframe = pd.read_csv(os.path.join(folder1, 'mua_fat_vanVeen.txt'), sep = ' ')

EP_Salomatina_dataframe = pd.read_csv(os.path.join(folder1, 'EP_Salomatina.txt'), sep = ' ')
DE_Salomatina_dataframe = pd.read_csv(os.path.join(folder1, 'DE_Salomatina.txt'), sep = ' ')
HY_Salomatina_dataframe = pd.read_csv(os.path.join(folder1, 'HY_Salomatina.txt'), sep = ' ')
iBCC_Salomatina_dataframe = pd.read_csv(os.path.join(folder1, 'iBCC_Salomatina.txt'), sep = ' ')
nBCC_Salomatina_dataframe = pd.read_csv(os.path.join(folder1, 'nBCC_Salomatina.txt'), sep = ' ')
SCC_Salomatina_dataframe = pd.read_csv(os.path.join(folder1, 'SCC_Salomatina.txt'), sep = ' ')
EP_Shimojo_dataframe = pd.read_csv(os.path.join(folder1, 'EP_Shimojo.txt'), sep = ' ')
DE_Shimojo_dataframe = pd.read_csv(os.path.join(folder1, 'DE_Shimojo.txt'), sep = ' ')
HY_Shimojo_dataframe = pd.read_csv(os.path.join(folder1, 'HY_Shimojo.txt'), sep = ' ')

n_and_k_EP_Ding_dataframe = pd.read_csv(os.path.join(folder1, 'n_and_k_EP_Ding.txt'), sep = ' ')
n_and_k_DE_Ding_dataframe = pd.read_csv(os.path.join(folder1, 'n_and_k_DE_Ding.txt'), sep = ' ')
n_HY_Matiatou_dataframe = pd.read_csv(os.path.join(folder1, 'n_HY_Matiatou.txt'), sep = ' ')
n_AT_Matiatou_dataframe = pd.read_csv(os.path.join(folder1, 'n_AT_Matiatou.txt'), sep = ' ')
beta_oxy_Friebel_dataframe = pd.read_csv(os.path.join(folder1, 'beta_oxy_Friebel.txt'), sep = ' ')

n_and_k_wat_Hale_dataframe = pd.read_csv(os.path.join(folder1, 'n_and_k_wat_Hale.txt'), sep = ' ')
n_and_k_wat_Segelstein_dataframe = pd.read_csv(os.path.join(folder1, 'n_and_k_wat_Segelstein.txt'), sep = ' ')
oxy_and_deo_Bosschaart_dataframe = pd.read_csv(os.path.join(folder1, 'oxy_and_deo_Bosschaart.txt'), sep = ' ')

folder2 = os.path.join(folder0, 'datasets', 'colors')

rspds_A_D50_D65_dataframe = pd.read_csv(os.path.join(folder2, 'rspds_A_D50_D65.txt'), sep = ' ')
rspds_D55_D75_dataframe = pd.read_csv(os.path.join(folder2, 'rspds_D55_D75.txt'), sep = ' ')
cmfs_dataframe = pd.read_csv(os.path.join(folder2, 'cmfs.txt'), sep = ' ')

Lab_Alaluf2002_dataframe = pd.read_csv(os.path.join(folder2, 'Lab_Alaluf2002.txt'), sep = ' ')
Lab_Xiao2017_dataframe = pd.read_csv(os.path.join(folder2, 'Lab_Xiao2017.txt'), sep = ' ')
