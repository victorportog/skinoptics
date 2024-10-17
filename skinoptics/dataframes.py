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

| Example:
| wps_dataframe (respective to datasets/colors/wps.txt)

+--------------+------------+---------+---------+---------+
| illuminant   | observer   |   Xn[-] |   Yn[-] |   Zn[-] |
+==============+============+=========+=========+=========+
| A            | 2o         |  1.0985 |       1 |  0.3558 |
+--------------+------------+---------+---------+---------+
| D50          | 2o         |  0.9641 |       1 |  0.8250 |
+--------------+------------+---------+---------+---------+
| D55          | 2o         |  0.9568 |       1 |  0.9214 |
+--------------+------------+---------+---------+---------+
| D65          | 2o         |  0.9504 |       1 |  1.0888 |
+--------------+------------+---------+---------+---------+
| D75          | 2o         |  0.9497 |       1 |  1.2257 |
+--------------+------------+---------+---------+---------+
| A            | 10o        |  1.1114 |       1 |  0.3520 |
+--------------+------------+---------+---------+---------+
| D50          | 10o        |  0.9671 |       1 |  0.8141 |
+--------------+------------+---------+---------+---------+
| D55          | 10o        |  0.9580 |       1 |  0.9093 |
+--------------+------------+---------+---------+---------+
| D65          | 10o        |  0.9481 |       1 |  1.0733 |
+--------------+------------+---------+---------+---------+
| D75          | 10o        |  0.9442 |       1 |  1.2060 |
+--------------+------------+---------+---------+---------+
'''

import os
import pandas as pd

folder0 = os.path.dirname(os.path.abspath(__file__))[:-10]
folder1 = folder0 + 'datasets\\optical_properties\\'

ext_and_molarext_eum_Sarna_dataframe = pd.read_csv(folder1 + 'ext_and_molarext_eum_Sarna.txt', sep = ' ') 
ext_and_molarext_phe_Sarna_dataframe = pd.read_csv(folder1 + 'ext_and_molarext_phe_Sarna.txt', sep = ' ')
ext_and_molarext_oxy_and_deo_Prahl_dataframe = pd.read_csv(folder1 + 'ext_and_molarext_oxy_and_deo_Prahl.txt', sep = ' ')
molarext_bil_Li_dataframe = pd.read_csv(folder1 + 'molarext_bil_Li.txt', sep = ' ')

mua_wat_Hale_dataframe = pd.read_csv(folder1 + 'mua_wat_Hale.txt', sep = ' ')
mua_wat_Segelstein_dataframe = pd.read_csv(folder1 + 'mua_wat_Segelstein.txt', sep = ' ')
mua_fat_vanVeen_dataframe = pd.read_csv(folder1 + 'mua_fat_vanVeen.txt', sep = ' ')

EP_Salomatina_dataframe = pd.read_csv(folder1 + 'EP_Salomatina.txt', sep = ' ')
DE_Salomatina_dataframe = pd.read_csv(folder1 + 'DE_Salomatina.txt', sep = ' ')
HY_Salomatina_dataframe = pd.read_csv(folder1 + 'HY_Salomatina.txt', sep = ' ')
iBCC_Salomatina_dataframe = pd.read_csv(folder1 + 'iBCC_Salomatina.txt', sep = ' ')
nBCC_Salomatina_dataframe = pd.read_csv(folder1 + 'nBCC_Salomatina.txt', sep = ' ')
SCC_Salomatina_dataframe = pd.read_csv(folder1 + 'SCC_Salomatina.txt', sep = ' ')
EP_Shimojo_dataframe = pd.read_csv(folder1 + 'EP_Shimojo.txt', sep = ' ')
DE_Shimojo_dataframe = pd.read_csv(folder1 + 'DE_Shimojo.txt', sep = ' ')
HY_Shimojo_dataframe = pd.read_csv(folder1 + 'HY_Shimojo.txt', sep = ' ')

n_and_k_EP_Ding_dataframe = pd.read_csv(folder1 + 'n_and_k_EP_Ding.txt', sep = ' ')
n_and_k_DE_Ding_dataframe = pd.read_csv(folder1 + 'n_and_k_DE_Ding.txt', sep = ' ')
n_HY_Matiatou_dataframe = pd.read_csv(folder1 + 'n_HY_Matiatou.txt', sep = ' ')
n_AT_Matiatou_dataframe = pd.read_csv(folder1 + 'n_AT_Matiatou.txt', sep = ' ')
beta_oxy_Friebel_dataframe = pd.read_csv(folder1 + 'beta_oxy_Friebel.txt', sep = ' ')

n_and_k_wat_Hale_dataframe = pd.read_csv(folder1 + 'n_and_k_wat_Hale.txt', sep = ' ')
n_and_k_wat_Segelstein_dataframe = pd.read_csv(folder1 + 'n_and_k_wat_Segelstein.txt', sep = ' ')
oxy_and_deo_Bosschaart_dataframe = pd.read_csv(folder1 + 'oxy_and_deo_Bosschaart.txt', sep = ' ')

folder2 = folder0 + 'datasets\\colors\\'

rspds_A_D50_D65_dataframe = pd.read_csv(folder2 + 'rspds_A_D50_D65.txt', sep = ' ')
rspds_D55_D75_dataframe = pd.read_csv(folder2 + 'rspds_D55_D75.txt', sep = ' ')
cmfs_dataframe = pd.read_csv(folder2 + 'cmfs.txt', sep = ' ')
wps_dataframe = pd.read_csv(folder2 + 'wps.txt', sep = ' ')

Lab_Alaluf2002_dataframe = pd.read_csv(folder2 + 'Lab_Alaluf2002.txt', sep = ' ')
Lab_Xiao2017_dataframe = pd.read_csv(folder2 + 'Lab_Xiao2017.txt', sep = ' ')
