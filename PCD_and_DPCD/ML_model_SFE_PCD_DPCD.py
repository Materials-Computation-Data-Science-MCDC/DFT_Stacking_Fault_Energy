#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 08:56:45 2022

@author: Gaurav Arora, Ph.D  Email:gauravarora.1100@gmail.com
Google scholar: https://urldefense.com/v3/__https://scholar.google.com/citations?user=odZBkmMAAAAJ__;!!PTd7Sdtyuw!Sx-tZV496f4Istg6rT_7YM80nb3RUm-zSERKvHU9d_T0fzi69VwaBP5SsaLTlouyqizeUFPmPF7d6G3-og$  
LinkdIn: https://urldefense.com/v3/__https://www.linkedin.com/in/gaurav-arora-1100/__;!!PTd7Sdtyuw!Sx-tZV496f4Istg6rT_7YM80nb3RUm-zSERKvHU9d_T0fzi69VwaBP5SsaLTlouyqizeUFPmPF4aCiBc1w$  

Please note that this code can be used to train and test on different dataset depending upon the composition. 
User has to change the data[i] which is used while making descriptors and target on line 210 and 211. 
For more detail refer to published article: https://urldefense.com/v3/__https://aip.scitation.org/doi/10.1063/5.0122675__;!!PTd7Sdtyuw!Sx-tZV496f4Istg6rT_7YM80nb3RUm-zSERKvHU9d_T0fzi69VwaBP5SsaLTlouyqizeUFPmPF7KaGJWug$  


"""


#Importing the required modules
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

#Defining variables used for normalizing the data
fac = 100
fac_tar = 1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##Importing the data for different compositions
name_of_excel_sheet="SFE_PCD_DPCD_data.xlsx"
#For single dopant
data_single = pd.read_excel(str(name_of_excel_sheet), sheet_name = 'single_dopant1')
data_single = data_single.drop(['system'], axis = 1)
data_single = data_single.iloc[0:27,:]

PCD_dopant_per_single = data_single.iloc[:,0:320]/fac
PCD_dopant_def_single = data_single.iloc[:,320:640]/fac
DPCD_dopant_single = data_single.iloc[:,1280:1600]
VEC_dopant_single = data_single.iloc[:,1920:1980]
supercell_vol_dopant_single = data_single.iloc[:,1981:1982]/fac
bader_charge_dopant_single = data_single.iloc[:,1982:1994]

PCD_HM_per = data_single.iloc[:,640:960]/fac
PCD_HM_def = data_single.iloc[:,960:1280]/fac
DPCD_HM = data_single.iloc[:,1600:1920]/fac
HM_vol = data_single.iloc[:,1980:1981]/fac

SFE_single_dopant = np.array(data_single.iloc[:,1994:1995])/fac_tar

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#For system where two same dopants are placed at first nearest neighbor
data_double_same_near = pd.read_excel(str(name_of_excel_sheet), sheet_name = 'double_dopants_same_near1')
data_double_same_near = data_double_same_near.drop(['system'], axis = 1)
data_double_same_near = data_double_same_near.iloc[0:26, :]

PCD_dopant_per_double_same_near = data_double_same_near.iloc[:,0:320]/fac
PCD_dopant_def_double_same_near = data_double_same_near.iloc[:,320:640]/fac
DPCD_dopant_double_same_near = data_double_same_near.iloc[:,640:960]
VEC_dopant_double_same_near = data_double_same_near.iloc[:,960:1020]
supercell_vol_dopant_double_near = data_double_same_near.iloc[:,1021:1022]/fac
bader_charge_dopant_double_near = data_double_same_near.iloc[:,1022:1034]

SFE_double_dopant_same_near = np.array(data_double_same_near.iloc[:,1034:1035])/fac_tar

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#For system where two same dopants are placed at third nearest neighbor
data_double_same_far = pd.read_excel(str(name_of_excel_sheet), sheet_name = 'double_dopants_same_far1')
data_double_same_far = data_double_same_far.drop(['system'], axis = 1)

PCD_dopant_per_double_same_far = data_double_same_far.iloc[:,0:320]/fac
PCD_dopant_def_double_same_far = data_double_same_far.iloc[:,320:640]/fac
DPCD_dopant_double_same_far = data_double_same_far.iloc[:,640:960]
VEC_dopant_double_same_far = data_double_same_far.iloc[:,960:1020]
supercell_vol_dopant_double_far = data_double_same_far.iloc[:,1021:1022]/fac
bader_charge_dopant_double_far = data_double_same_far.iloc[:,1022:1034]

SFE_double_dopant_same_far = np.array(data_double_same_far.iloc[:,1034:1035])/fac_tar

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#For system where two different dopants are placed at first nearest neighbor
data_double_diff_near = pd.read_excel(str(name_of_excel_sheet), sheet_name = 'diff_dopants_near1')
data_double_diff_near = data_double_diff_near.drop(['system'], axis = 1)

PCD_dopant_per_double_diff_near = data_double_diff_near.iloc[:,0:320]/fac
PCD_dopant_def_double_diff_near = data_double_diff_near.iloc[:,320:640]/fac
DPCD_dopant_double_diff_near = data_double_diff_near.iloc[:,640:960]
VEC_dopant_double_diff_near = data_double_diff_near.iloc[:,960:1020]
supercell_vol_dopant_double_diff_near = data_double_diff_near.iloc[:,1021:1022]/fac
bader_charge_dopant_double_diff_near = data_double_diff_near.iloc[:,1022:1034]

SFE_double_dopant_diff_near = np.array(data_double_diff_near.iloc[:,1034:1035])/fac_tar

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#For system where two different dopants are placed at third nearest neighbor
data_double_diff_far = pd.read_excel(str(name_of_excel_sheet), sheet_name = 'diff_dopants_far1')
data_double_diff_far = data_double_diff_far.drop(['system'], axis = 1)

PCD_dopant_per_double_diff_far = data_double_diff_far.iloc[:,0:320]/fac
PCD_dopant_def_double_diff_far = data_double_diff_far.iloc[:,320:640]/fac
DPCD_dopant_double_diff_far = data_double_diff_far.iloc[:,640:960]
VEC_dopant_double_diff_far = data_double_diff_far.iloc[:,960:1020]
supercell_vol_dopant_double_diff_far = data_double_diff_far.iloc[:,1021:1022]/fac
bader_charge_dopant_double_diff_far = data_double_diff_far.iloc[:,1022:1034]

SFE_double_dopant_diff_far = np.array(data_double_diff_far.iloc[:,1034:1035])/fac_tar

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#For system where three different dopants are placed at random sites
data_quat = pd.read_excel(str(name_of_excel_sheet), sheet_name = 'quatenary_1')
data_quat = data_quat.drop(['system'], axis = 1)

PCD_dopant_per_quat = data_quat.iloc[:,0:320]/fac
PCD_dopant_def_quat = data_quat.iloc[:,320:640]/fac
DPCD_dopant_quat = data_quat.iloc[:,640:960]
VEC_dopant_quat = data_quat.iloc[:,960:1020]
supercell_vol_dopant_quat = data_quat.iloc[:,1021:1022]/fac
bader_charge_dopant_quat = data_quat.iloc[:,1022:1034]

SFE_quat = np.array(data_quat.iloc[:,1034:1035])/fac_tar


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# For systems having Co as one of the dopant 
data_only_Co = pd.read_excel(str(name_of_excel_sheet), sheet_name = 'only_Co_results')
data_only_Co = data_only_Co.drop(['system'], axis = 1)

PCD_dopant_per_only_Co = data_only_Co.iloc[:,0:320]/fac
PCD_dopant_def_only_Co = data_only_Co.iloc[:,320:640]/fac
DPCD_dopant_only_Co = data_only_Co.iloc[:,640:960]
VEC_dopant_only_Co = data_only_Co.iloc[:,960:1020]
supercell_vol_dopant_only_Co = data_only_Co.iloc[:,1021:1022]/fac
bader_charge_dopant_only_Co = data_only_Co.iloc[:,1022:1034]

SFE_only_Co = np.array(data_only_Co.iloc[:,1034:1035])/fac_tar

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#For systems where 6 dopants are used
data_high_conc = pd.read_excel(str(name_of_excel_sheet), sheet_name = 'higher_conc_1')
data_high_conc = data_high_conc.drop(['system'], axis = 1)

PCD_dopant_per_high_conc = data_high_conc.iloc[:,0:320]/fac
PCD_dopant_def_high_conc = data_high_conc.iloc[:,320:640]/fac
DPCD_dopant_high_conc = data_high_conc.iloc[:,640:960]
VEC_dopant_high_conc = data_high_conc.iloc[:,960:1020]
supercell_vol_dopant_high_conc = data_high_conc.iloc[:,1021:1022]/fac
bader_charge_dopant_high_conc = data_high_conc.iloc[:,1022:1034]

SFE_high_conc = np.array(data_high_conc.iloc[:,1034:1035])/fac_tar
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Concatenating the data 
data1 = pd.concat([PCD_dopant_per_single, PCD_dopant_def_single, DPCD_dopant_single, PCD_HM_per, 
                   PCD_HM_def, DPCD_HM, VEC_dopant_single, supercell_vol_dopant_single, HM_vol, 
                   bader_charge_dopant_single], axis = 1)

data2 = pd.concat([PCD_dopant_per_double_same_near, PCD_dopant_def_double_same_near, 
                   DPCD_dopant_double_same_near, PCD_HM_per.iloc[0:len(data_double_same_near),:], 
                   PCD_HM_def.iloc[0:len(data_double_same_near),:], DPCD_HM.iloc[0:len(data_double_same_near),:], 
                   VEC_dopant_double_same_near, supercell_vol_dopant_double_near, 
                  HM_vol.iloc[0:len(data_double_same_near),:], bader_charge_dopant_double_near], axis = 1)

data3 = pd.concat([PCD_dopant_per_double_same_far, PCD_dopant_def_double_same_far, DPCD_dopant_double_same_far,
                  PCD_HM_per.iloc[0:len(data_double_same_far),:], PCD_HM_def.iloc[0:len(data_double_same_far),:], 
                  DPCD_HM.iloc[0:len(data_double_same_far),:], VEC_dopant_double_same_far,
                  supercell_vol_dopant_double_far, 
                  HM_vol.iloc[0:len(data_double_same_far),:], bader_charge_dopant_double_far], axis = 1)

data4 = pd.concat([PCD_dopant_per_double_diff_near, PCD_dopant_def_double_diff_near, DPCD_dopant_double_diff_near,
              PCD_HM_per.iloc[0:len(data_double_diff_near),:], PCD_HM_def.iloc[0:len(data_double_diff_near),:], 
              DPCD_HM.iloc[0:len(data_double_diff_near),:], VEC_dopant_double_diff_near, 
              supercell_vol_dopant_double_diff_near, 
                  HM_vol.iloc[0:len(data_double_diff_near),:],
                   bader_charge_dopant_double_diff_near], axis = 1)

data5 = pd.concat([PCD_dopant_per_double_diff_far, PCD_dopant_def_double_diff_far, DPCD_dopant_double_diff_far,
              PCD_HM_per.iloc[0:len(data_double_diff_far),:], PCD_HM_def.iloc[0:len(data_double_diff_far),:], 
              DPCD_HM.iloc[0:len(data_double_diff_far),:], VEC_dopant_double_diff_far, 
              supercell_vol_dopant_double_diff_far, 
                  HM_vol.iloc[0:len(data_double_diff_far),:],
                   bader_charge_dopant_double_diff_far], axis = 1)


data6 = pd.concat([PCD_dopant_per_only_Co, PCD_dopant_def_only_Co, DPCD_dopant_only_Co,
              PCD_HM_per.iloc[0:len(data_only_Co),:], PCD_HM_def.iloc[0:len(data_only_Co),:], DPCD_HM.iloc[0:len(data_only_Co),:],
              VEC_dopant_only_Co, 
              supercell_vol_dopant_only_Co, 
                  HM_vol.iloc[0:len(data_only_Co),:],
                    bader_charge_dopant_only_Co], axis = 1)

data7 = pd.concat([PCD_dopant_per_quat, PCD_dopant_def_quat, DPCD_dopant_quat,
              PCD_HM_per.iloc[0:len(data_quat),:], PCD_HM_def.iloc[0:len(data_quat),:], 
              DPCD_HM.iloc[0:len(data_quat),:], VEC_dopant_quat, 
              supercell_vol_dopant_quat, 
                  HM_vol.iloc[0:len(data_quat),:],
                   bader_charge_dopant_quat], axis = 1)

data8 = pd.concat([PCD_dopant_per_high_conc, PCD_dopant_def_high_conc, DPCD_dopant_high_conc,
              PCD_HM_per.iloc[0:len(data_high_conc),:], PCD_HM_def.iloc[0:len(data_high_conc),:], 
              DPCD_HM.iloc[0:len(data_high_conc),:], VEC_dopant_high_conc, 
              supercell_vol_dopant_high_conc, 
                  HM_vol.iloc[0:len(data_high_conc),:],
                   bader_charge_dopant_high_conc], axis = 1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Defining the descriptors and target data for training a machine learning model
desc = np.array(pd.concat([data1, data2, data3, data4, data5, data6], axis = 0))
tar = np.concatenate((SFE_single_dopant, 
                      SFE_double_dopant_same_near,
                      SFE_double_dopant_same_far,
                      SFE_double_dopant_diff_near,
                      SFE_double_dopant_diff_far,
                      SFE_only_Co), axis = 0)

#Spliting the data into training and testing 
train_desc, test_desc, train_tar, test_tar = train_test_split(desc, tar, test_size = 0.2,
                                                              random_state = 43)

#Loading the support vector regression model and training 
model = SVR(kernel="poly", C = 500, gamma="auto", degree = 3, epsilon=0.1, coef0=1)
model = model.fit(train_desc, train_tar)

#Predicting on the training data
pred_SFE_train = model.predict(train_desc)
pred_SFE_train = pred_SFE_train.reshape(len(pred_SFE_train), )
mse_train = mse(pred_SFE_train, train_tar)
corr_train = r2(train_tar,pred_SFE_train)
print('rmse_train', np.sqrt(mse_train))
print('coeff of deter. train', corr_train)

#Predicting on the testing data
pred_SFE_test = model.predict(test_desc)
pred_SFE_test = pred_SFE_test.reshape(len(pred_SFE_test), )
mse_test = mse(pred_SFE_test, test_tar)
corr_test = r2(test_tar,pred_SFE_test)
print('rmse_test', np.sqrt(mse_test))
print('coeff of deter. test', corr_test)

#Predicting on systems for higher concentration (not used while training)
predi_data = np.array(pd.concat([data7], axis = 0))
predi_tar = np.concatenate((SFE_quat), axis = 0)
prediction = model.predict(predi_data)
rmse_pred = np.sqrt(mse(prediction,predi_tar))
corr_diff = r2(predi_tar,prediction)
print('rmse high_conc', rmse_pred)
print('coeff of deter. high_conc', corr_diff)

#Plotting the necessary results
fig,ax = plt.subplots(figsize = (5,5))
ax.scatter( train_tar, pred_SFE_train,label = 'training')
ax.scatter( test_tar,pred_SFE_test, label = 'test')
ax.scatter( predi_tar, prediction,label = 'higher conc')
ax.set_ylabel('Predicted SFE (%)')
ax.set_xlabel('True SFE (%)')
ax.legend()