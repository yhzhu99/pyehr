#!/usr/bin/env python
# coding: utf-8

# # Preprocess the TJH Dataset

# ## Import packages

# In[ ]:


import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from datasets.preprocess.tools import forward_fill_pipeline, normalize_dataframe

data_dir = "./datasets/tjh/"
Path(os.path.join(data_dir, 'processed')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(data_dir, 'statistics')).mkdir(parents=True, exist_ok=True)

SEED = 42

# ## Read data from files

# In[ ]:


df = pd.read_excel(os.path.join(data_dir, 'raw', 'time_series_375_prerpocess_en.xlsx'))

# ## Preprocess Data

# ### Rename columns

# In[ ]:


df = df.rename(columns={"PATIENT_ID": "PatientID", "outcome": "Outcome", "gender": "Sex", "age": "Age", "RE_DATE": "RecordTime", "Admission time": "AdmissionTime", "Discharge time": "DischargeTime"})

# ### Fill PatientID column

# In[ ]:


df['PatientID'] = df['PatientID'].ffill()

# ### Format data values

# In[ ]:


# gender transformation: 1--male, 0--female
df['Sex'] = df['Sex'].replace(2, 0)

# only reserve y-m-d precision for `RE_DATE` and `Discharge time` columns
df['RecordTime'] = df['RecordTime'].dt.strftime('%Y-%m-%d')
df['DischargeTime'] = df['DischargeTime'].dt.strftime('%Y-%m-%d')
df['AdmissionTime'] = df['AdmissionTime'].dt.strftime('%Y-%m-%d')

# ### Exclude patients with missing labels

# In[ ]:


df = df.dropna(subset = ['PatientID', 'RecordTime', 'DischargeTime'], how='any')

# ### Calculate the Length-of-Stay (LOS) label

# In[ ]:


df['LOS'] = (pd.to_datetime(df['DischargeTime']) - pd.to_datetime(df['RecordTime'])).dt.days

# Notice: Set negative LOS values to 0
df['LOS'] = df['LOS'].apply(lambda x: 0 if x < 0 else x)

# ### Drop columns whose values are all the same or all NaN

# In[ ]:


# Drop '2019-nCoV nucleic acid detection' column 
df = df.drop(columns=['2019-nCoV nucleic acid detection'])

# ### Record feature names

# In[ ]:


basic_records = ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime']
target_features = ['Outcome', 'LOS']
demographic_features = ['Sex', 'Age']
labtest_features = ['Hypersensitive cardiac troponinI', 'hemoglobin', 'Serum chloride', 'Prothrombin time', 'procalcitonin', 'eosinophils(%)', 'Interleukin 2 receptor', 'Alkaline phosphatase', 'albumin', 'basophil(%)', 'Interleukin 10', 'Total bilirubin', 'Platelet count', 'monocytes(%)', 'antithrombin', 'Interleukin 8', 'indirect bilirubin', 'Red blood cell distribution width ', 'neutrophils(%)', 'total protein', 'Quantification of Treponema pallidum antibodies', 'Prothrombin activity', 'HBsAg', 'mean corpuscular volume', 'hematocrit', 'White blood cell count', 'Tumor necrosis factorα', 'mean corpuscular hemoglobin concentration', 'fibrinogen', 'Interleukin 1β', 'Urea', 'lymphocyte count', 'PH value', 'Red blood cell count', 'Eosinophil count', 'Corrected calcium', 'Serum potassium', 'glucose', 'neutrophils count', 'Direct bilirubin', 'Mean platelet volume', 'ferritin', 'RBC distribution width SD', 'Thrombin time', '(%)lymphocyte', 'HCV antibody quantification', 'D-D dimer', 'Total cholesterol', 'aspartate aminotransferase', 'Uric acid', 'HCO3-', 'calcium', 'Amino-terminal brain natriuretic peptide precursor(NT-proBNP)', 'Lactate dehydrogenase', 'platelet large cell ratio ', 'Interleukin 6', 'Fibrin degradation products', 'monocytes count', 'PLT distribution width', 'globulin', 'γ-glutamyl transpeptidase', 'International standard ratio', 'basophil count(#)', 'mean corpuscular hemoglobin ', 'Activation of partial thromboplastin time', 'Hypersensitive c-reactive protein', 'HIV antibody quantification', 'serum sodium', 'thrombocytocrit', 'ESR', 'glutamic-pyruvic transaminase', 'eGFR', 'creatinine']

require_impute_features = labtest_features
normalize_features = ['Age'] + labtest_features + ['LOS']

# ### Set negative values to NaN

# In[ ]:


# Set negative values to NaN
df[df[demographic_features + labtest_features] < 0] = np.nan

# ### Merge by date

# In[ ]:


# Merge by PatientID and RecordTime
df = df.groupby(['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime'], dropna=True, as_index = False).mean()

# ### Change the order of columns

# In[ ]:


df = df[ basic_records + target_features + demographic_features + labtest_features ]

# ### Export data to files

# In[ ]:


df.to_csv(os.path.join(data_dir, 'processed', 'tjh_dataset_formatted.csv'), index=False)

# ## Stratified split dataset into train, validation and test sets
# 
# - Also include (Imputation & Normalization & Outlier Filtering) steps
# - The train, validation and test sets are saved in the `./processed` folder
# 

# ### 10-fold dataset setting
# 
# - use 8:1:1 10-fold

# In[ ]:


num_folds = 10

# Group the dataframe by patient ID
grouped = df.groupby('PatientID')

# Split the patient IDs into train/val/test sets
patients = np.array(list(grouped.groups.keys()))
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

for fold, (train_val_index, test_index) in enumerate(kf.split(patients, df.groupby('PatientID')['Outcome'].first())):
    # Get the train/val/test patient IDs for the current fold
    train_val_patients, test_patients = patients[train_val_index], patients[test_index]

    # Split the train_val_patients into train/val sets
    train_patients, val_patients = train_test_split(train_val_patients, test_size=1/(num_folds-1), random_state=SEED, stratify=df[df['PatientID'].isin(train_val_patients)].groupby('PatientID')['Outcome'].first())

    # Create train, val, and test dataframes for the current fold
    train_df = df[df['PatientID'].isin(train_patients)]
    val_df = df[df['PatientID'].isin(val_patients)]
    test_df = df[df['PatientID'].isin(test_patients)]
    
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Save the train, val, and test dataframes for the current fold to csv files
    
    fold_dir = os.path.join(data_dir, 'processed', f'fold_{fold}')
    Path(fold_dir).mkdir(parents=True, exist_ok=True)
    # train_df.to_csv(os.path.join(fold_dir, "train_raw.csv"), index=False)
    # val_df.to_csv(os.path.join(fold_dir, "val_raw.csv"), index=False)
    # test_df.to_csv(os.path.join(fold_dir, "test_raw.csv"), index=False)

    # Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range

    # Normalize data
    train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)
    
    # Drop rows if all features are recorded NaN
    train_df = train_df.dropna(axis=0, how='all', subset=normalize_features)
    val_df = val_df.dropna(axis=0, how='all', subset=normalize_features)
    test_df = test_df.dropna(axis=0, how='all', subset=normalize_features)

    # # Save the train, val, and test dataframes for the current fold to csv files
    # train_df.to_csv(os.path.join(fold_dir, "train_after_zscore.csv"), index=False)
    # val_df.to_csv(os.path.join(fold_dir, "val_after_zscore.csv"), index=False)
    # test_df.to_csv(os.path.join(fold_dir, "test_after_zscore.csv"), index=False)

    # Forward Imputation after grouped by PatientID
    # Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
    train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
    val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
    test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

    # Save the imputed dataset to pickle file
    pd.to_pickle(train_x, os.path.join(fold_dir, "train_x.pkl"))
    pd.to_pickle(train_y, os.path.join(fold_dir, "train_y.pkl"))
    pd.to_pickle(train_pid, os.path.join(fold_dir, "train_pid.pkl"))
    pd.to_pickle(val_x, os.path.join(fold_dir, "val_x.pkl"))
    pd.to_pickle(val_y, os.path.join(fold_dir, "val_y.pkl"))
    pd.to_pickle(val_pid, os.path.join(fold_dir, "val_pid.pkl"))
    pd.to_pickle(test_x, os.path.join(fold_dir, "test_x.pkl"))
    pd.to_pickle(test_y, os.path.join(fold_dir, "test_y.pkl"))
    pd.to_pickle(test_pid, os.path.join(fold_dir, "test_pid.pkl"))
    pd.to_pickle(los_info, os.path.join(fold_dir, "los_info.pkl"))

# ### Hold-out dataset setting (Stratified)
# 
# - For TJH dataset, use 7:1:2 splitting strategy, 70% training, 10% validation, 20% testing
# 

# In[ ]:


# Group the dataframe by patient ID
grouped = df.groupby('PatientID')

# Get the patient IDs and outcomes
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

# Get the train_val/test patient IDs
train_val_patients, test_patients = train_test_split(patients, test_size=20/100, random_state=SEED, stratify=patients_outcome)

# Get the train/val patient IDs
train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
train_patients, val_patients = train_test_split(train_val_patients, test_size=10/80, random_state=SEED, stratify=train_val_patients_outcome)
# Create train, val, test dataframes for the current fold
train_df = df[df['PatientID'].isin(train_patients)]
val_df = df[df['PatientID'].isin(val_patients)]
test_df = df[df['PatientID'].isin(test_patients)]
save_dir = os.path.join(data_dir, 'processed', 'fold_10') # forward fill
Path(save_dir).mkdir(parents=True, exist_ok=True)

# # Save the train, val, and test dataframes for the current fold to csv files
# train_df.to_csv(os.path.join(save_dir, "train_raw.csv"), index=False)
# val_df.to_csv(os.path.join(save_dir, "val_raw.csv"), index=False)
# test_df.to_csv(os.path.join(save_dir, "test_raw.csv"), index=False)
# Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

# # Save the zscored dataframes to csv files
# train_df.to_csv(os.path.join(save_dir, "train_after_zscore.csv"), index=False)
# val_df.to_csv(os.path.join(save_dir, "val_after_zscore.csv"), index=False)
# test_df.to_csv(os.path.join(save_dir, "test_after_zscore.csv"), index=False)

# Forward Imputation after grouped by PatientID
# Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

# Save the imputed dataset to pickle file
pd.to_pickle(train_x, os.path.join(save_dir, "train_x.pkl"))
pd.to_pickle(train_y, os.path.join(save_dir, "train_y.pkl"))
pd.to_pickle(train_pid, os.path.join(save_dir, "train_pid.pkl"))
pd.to_pickle(val_x, os.path.join(save_dir, "val_x.pkl"))
pd.to_pickle(val_y, os.path.join(save_dir, "val_y.pkl"))
pd.to_pickle(val_pid, os.path.join(save_dir, "val_pid.pkl"))
pd.to_pickle(test_x, os.path.join(save_dir, "test_x.pkl"))
pd.to_pickle(test_y, os.path.join(save_dir, "test_y.pkl"))
pd.to_pickle(test_pid, os.path.join(save_dir, "test_pid.pkl"))
pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl")) # LOS statistics (calculated from the train set)

# ### Hold-out dataset setting (According to admission time)
# 
# - For TJH dataset, use 7:1:2 splitting strategy, 70% training, 10% validation, 20% testing
# 
# **time order**
# 

# In[ ]:


# Group the dataframe by PatientID
grouped = df.groupby('PatientID')

# Get the first AdmissionTime for each patient
patients_admission_time = {patient_id: grouped.get_group(patient_id)['AdmissionTime'].iloc[0] for patient_id in grouped.groups.keys()}

# Sort patient IDs by first AdmissionTime
sorted_patients = sorted(patients_admission_time, key=patients_admission_time.get)

# Calculate the indices for splitting
train_idx = int(len(sorted_patients) * 0.6)
val_idx = int(len(sorted_patients) * 0.8)

# Split the patient IDs into train, validation, and test sets
train_patients = sorted_patients[:train_idx]
val_patients = sorted_patients[train_idx:val_idx]
test_patients = sorted_patients[val_idx:]

# Create train, val, test, dataframes for the current fold
train_df = df[df['PatientID'].isin(train_patients)]
val_df = df[df['PatientID'].isin(val_patients)]
test_df = df[df['PatientID'].isin(test_patients)]
save_dir = os.path.join(data_dir, 'processed', 'fold_11') # forward fill
Path(save_dir).mkdir(parents=True, exist_ok=True)

train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

# Forward Imputation after grouped by PatientID
# Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

# Save the imputed dataset to pickle file
pd.to_pickle(train_x, os.path.join(save_dir, "train_x.pkl"))
pd.to_pickle(train_y, os.path.join(save_dir, "train_y.pkl"))
pd.to_pickle(train_pid, os.path.join(save_dir, "train_pid.pkl"))
pd.to_pickle(val_x, os.path.join(save_dir, "val_x.pkl"))
pd.to_pickle(val_y, os.path.join(save_dir, "val_y.pkl"))
pd.to_pickle(val_pid, os.path.join(save_dir, "val_pid.pkl"))
pd.to_pickle(test_x, os.path.join(save_dir, "test_x.pkl"))
pd.to_pickle(test_y, os.path.join(save_dir, "test_y.pkl"))
pd.to_pickle(test_pid, os.path.join(save_dir, "test_pid.pkl"))
pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl")) # LOS statistics (calculated from the train set)
