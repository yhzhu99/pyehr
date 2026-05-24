#!/usr/bin/env python
# coding: utf-8

# # Preprocess the CDSL Dataset

# ## Import packages

# In[1]:


import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
import datetime
import re
from sklearn.model_selection import train_test_split, StratifiedKFold

from datasets.preprocess.tools import forward_fill_pipeline, normalize_dataframe

data_dir = "./datasets/cdsl/"
Path(os.path.join(data_dir, 'processed')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(data_dir, 'statistics')).mkdir(parents=True, exist_ok=True)

SEED = 42

# ## Preprocess Demographic Data

# In[ ]:


demographic = pd.read_csv(os.path.join(data_dir, 'raw', '19_04_2021/COVID_DSL_01.CSV'), encoding='ISO-8859-1', sep='|')
print(len(demographic))
demographic.head()

# In[ ]:


med = pd.read_csv(os.path.join(data_dir, 'raw', '19_04_2021/COVID_DSL_04.CSV'), encoding='ISO-8859-1', sep='|')
print(len(med))
med.head()

# In[ ]:


len(med['ID_ATC7'].unique())

# ### Exclude patients with missing labels

# In[ ]:


print(len(demographic))
demographic = demographic.dropna(axis=0, how='any', subset=['IDINGRESO', 'F_INGRESO_ING', 'F_ALTA_ING', 'MOTIVO_ALTA_ING'])
print(len(demographic))

# In[ ]:


def outcome2num(x):
    if x == 'Fallecimiento':
        return 1
    else:
        return 0

def to_one_hot(x, feature):
    if x == feature:
        return 1
    else:
        return 0

# In[ ]:


# select necessary columns from demographic
demographic = demographic[
        [
            'IDINGRESO', 
            'EDAD',
            'SEX',
            'F_INGRESO_ING', 
            'F_ALTA_ING', 
            'MOTIVO_ALTA_ING', 
            'ESPECIALIDAD_URGENCIA', 
            'DIAG_URG'
        ]
    ]

# rename column
demographic = demographic.rename(columns={
    'IDINGRESO': 'PATIENT_ID',
    'EDAD': 'AGE',
    'SEX': 'SEX',
    'F_INGRESO_ING': 'ADMISSION_DATE',
    'F_ALTA_ING': 'DEPARTURE_DATE',
    'MOTIVO_ALTA_ING': 'OUTCOME',
    'ESPECIALIDAD_URGENCIA': 'DEPARTMENT_OF_EMERGENCY',
    'DIAG_URG': 'DIAGNOSIS_AT_EMERGENCY_VISIT'
})

# SEX: Male: 1; Female: 0
demographic['SEX'] = demographic['SEX'].replace({'MALE': 1, 'FEMALE': 0})

# outcome: Fallecimiento(dead): 1; others: 0
demographic['OUTCOME'] = demographic['OUTCOME'].map(outcome2num)


# In[ ]:


# only reserve useful columns in demographic table
demographic = demographic[
        [
            'PATIENT_ID',
            'AGE',
            'SEX',
            'ADMISSION_DATE',
            'DEPARTURE_DATE',
            'OUTCOME',
            # 'DIFFICULTY_BREATHING',
            # 'SUSPECT_COVID',
            # 'FEVER',
            # 'EMERGENCY'
        ]
    ]

# In[ ]:


demographic.describe().to_csv(os.path.join(data_dir, 'statistics', 'demographic_overview.csv'), index=False)
demographic.describe()

# In[ ]:


demographic.to_csv(os.path.join(data_dir, 'processed', 'demographic.csv'), index=False)
demographic.head()

# ## Preprocess Vital Signal Data

# In[ ]:


vital_signs = pd.read_csv(os.path.join(data_dir, 'raw', '19_04_2021/COVID_DSL_02.CSV'), encoding='ISO-8859-1', sep='|')
print(len(vital_signs))
vital_signs.head()

# In[ ]:


vital_signs = vital_signs.rename(columns={
    'IDINGRESO': 'PATIENT_ID',
    'CONSTANTS_ING_DATE': 'RECORD_DATE',
    'CONSTANTS_ING_TIME': 'RECORD_TIME',
    'FC_HR_ING': 'HEART_RATE',
    'GLU_GLY_ING': 'BLOOD_GLUCOSE',
    'SAT_02_ING': 'OXYGEN_SATURATION',
    'TA_MAX_ING': 'MAX_BLOOD_PRESSURE',
    'TA_MIN_ING': 'MIN_BLOOD_PRESSURE',
    'TEMP_ING': 'TEMPERATURE'
})
vital_signs['RECORD_TIME'] = vital_signs['RECORD_DATE'] + ' ' + vital_signs['RECORD_TIME']
vital_signs['RECORD_TIME'] = vital_signs['RECORD_TIME'].map(lambda x: str(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M')))
vital_signs = vital_signs.drop(['RECORD_DATE', 'SAT_02_ING_OBS', 'BLOOD_GLUCOSE'], axis=1)

# In[ ]:


vital_signs.describe()

# In[ ]:


vital_signs.head()

# In[ ]:


def format_temperature(x):
    if type(x) == str:
        return float(x.replace(',', '.'))
    else:
        return float(x)

def format_oxygen(x):
    x = float(x)
    if x > 100:
        return np.nan
    else:
        return x

def format_heart_rate(x):
    x = int(x)
    if x > 220:
        return np.nan
    else:
        return x

vital_signs['TEMPERATURE'] = vital_signs['TEMPERATURE'].map(lambda x: format_temperature(x))
vital_signs['OXYGEN_SATURATION'] = vital_signs['OXYGEN_SATURATION'].map(lambda x: format_oxygen(x))
vital_signs['HEART_RATE'] = vital_signs['HEART_RATE'].map(lambda x: format_heart_rate(x))

# In[ ]:


vital_signs = vital_signs.replace(0, np.nan)

# In[ ]:


vital_signs = vital_signs.groupby(['PATIENT_ID', 'RECORD_TIME'], dropna=True, as_index = False).mean()
vital_signs.head()

# In[ ]:


vital_signs.describe()

# In[ ]:


vital_signs.describe().to_csv(os.path.join(data_dir, 'statistics', 'vital_signs_overview.csv'), index=False)
vital_signs.describe()

# In[ ]:


vital_signs.to_csv(os.path.join(data_dir, 'processed', 'visual_signs.csv'), index=False)
vital_signs.head()

# ## Preprocess Lab Tests Data

# In[ ]:


lab_tests = pd.read_csv(os.path.join(data_dir, 'raw', '19_04_2021/COVID_DSL_06_v2.CSV'), encoding='ISO-8859-1', sep=';')
lab_tests = lab_tests.rename(columns={'IDINGRESO': 'PATIENT_ID'})
print(len(lab_tests))

# only reserve useful columns
lab_tests = lab_tests[
        [
            'PATIENT_ID',
            'LAB_NUMBER',
            'LAB_DATE',
            'TIME_LAB',
            'ITEM_LAB',
            'VAL_RESULT'
            # UD_RESULT: unit
            # REF_VALUES: reference values
        ]
    ]

lab_tests.head()

# In[ ]:


lab_tests = lab_tests.groupby(['PATIENT_ID', 'LAB_NUMBER', 'LAB_DATE', 'TIME_LAB', 'ITEM_LAB'], dropna=True, as_index = False).first()
lab_tests = lab_tests.set_index(['PATIENT_ID', 'LAB_NUMBER', 'LAB_DATE', 'TIME_LAB', 'ITEM_LAB'], drop = True).unstack('ITEM_LAB')['VAL_RESULT'].reset_index()

lab_tests = lab_tests.drop([
    'CFLAG -- ALARMA HEMOGRAMA', 
    'CORONA -- PCR CORONAVIRUS 2019nCoV', 
    'CRIOGLO -- CRIOGLOBULINAS',
    'EGCOVID -- ESTUDIO GENETICO COVID-19',
    'FRO1 -- ',
    'FRO1 -- FROTIS EN SANGRE PERIFERICA',
    'FRO2 -- ',
    'FRO2 -- FROTIS EN SANGRE PERIFERICA',
    'FRO3 -- ',
    'FRO3 -- FROTIS EN SANGRE PERIFERICA',
    'FRO_COMEN -- ',
    'FRO_COMEN -- FROTIS EN SANGRE PERIFERICA',
    'G-CORONAV (RT-PCR) -- Tipo de muestra: ASPIRADO BRONCOALVEOLAR',
    'G-CORONAV (RT-PCR) -- Tipo de muestra: EXUDADO',
    'GRRH -- GRUPO SANGUÖNEO Y FACTOR Rh',
    'HEML -- RECUENTO CELULAR LIQUIDO',
    'HEML -- Recuento Hemat¡es',
    'IFSUERO -- INMUNOFIJACION EN SUERO',
    'OBS_BIOMOL -- OBSERVACIONES GENETICA MOLECULAR',
    'OBS_BIOO -- Observaciones Bioqu¡mica Orina',
    'OBS_CB -- Observaciones Coagulaci¢n',
    'OBS_GASES -- Observaciones Gasometr¡a Arterial',
    'OBS_GASV -- Observaciones Gasometr¡a Venosa',
    'OBS_GEN2 -- OBSERVACIONES GENETICA',
    'OBS_HOR -- Observaciones Hormonas',
    'OBS_MICRO -- Observaciones Microbiolog¡a',
    'OBS_NULA2 -- Observaciones Bioqu¡mica',
    'OBS_NULA3 -- Observaciones Hematolog¡a',
    'OBS_PESP -- Observaciones Pruebas especiales',
    'OBS_SERO -- Observaciones Serolog¡a',
    'OBS_SIS -- Observaciones Orina',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: ASPIRADO BRONCOALVEOLAR',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: BAS',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: ESPUTO',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: EXUDADO',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: LAVADO BRONCOALVEOLAR',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: LAVADO NASOFARÖNGEO',
    'PTGOR -- PROTEINOGRAMA ORINA',
    'RESUL_IFT -- ESTUDIO DE INMUNOFENOTIPO',
    'RESUL_IFT -- Resultado',
    'Resultado -- Resultado',
    'SED1 -- ',
    'SED1 -- SEDIMENTO',
    'SED2 -- ',
    'SED2 -- SEDIMENTO',
    'SED3 -- ',
    'SED3 -- SEDIMENTO',
    'TIPOL -- TIPO DE LIQUIDO',
    'Tecnica -- T\x82cnica',
    'TpMues -- Tipo de muestra',
    'VHCBLOT -- INMUNOBLOT VIRUS HEPATITIS C',
    'VIR_TM -- VIRUS TIPO DE MUESTRA',
    'LEGIORI -- AG. LEGIONELA PNEUMOPHILA EN ORINA',
    'NEUMOORI -- AG NEUMOCOCO EN ORINA',
    'VIHAC -- VIH AC'
    ], axis=1)


lab_tests.head()

# In[ ]:


lab_tests = lab_tests.replace('Sin resultado.', np.nan)
lab_tests = lab_tests.replace('Sin resultado', np.nan)
lab_tests = lab_tests.replace('----', np.nan).replace('---', np.nan)
lab_tests = lab_tests.replace('> ', '').replace('< ', '')

def change_format(x):
    if x is None:
        return np.nan
    elif type(x) == str:
        if x.startswith('Negativo ('):
            return x.replace('Negativo (', '-')[:-1]
        elif x.startswith('Positivo ('):
            return x.replace('Positivo (', '')[:-1]
        elif x.startswith('Zona limite ('):
            return x.replace('Zona limite (', '')[:-1]
        elif x.startswith('>'):
            return x.replace('> ', '').replace('>', '')
        elif x.startswith('<'):
            return x.replace('< ', '').replace('<', '')
        elif x.endswith(' mg/dl'):
            return x.replace(' mg/dl', '')
        elif x.endswith('/æl'):
            return x.replace('/æl', '')
        elif x.endswith(' copias/mL'):
            return x.replace(' copias/mL', '')
        elif x == 'Numerosos':
            return 1.5
        elif x == 'Aislados':
            return 0.5
        elif x == 'Se detecta' or x == 'Se observan' or x == 'Normal' or x == 'Positivo':
            return 1
        elif x == 'No se detecta' or x == 'No se observan' or x == 'Negativo':
            return 0
        elif x == 'Indeterminado':
            return np.nan
        else:
            num = re.findall(r"[-+]?\d+\.\d+", x)
            if len(num) == 0:
                return np.nan
            else:
                return num[0]
    else:
        return x

feature_value_dict = dict()

for k in tqdm(lab_tests.keys()[4:]):
    lab_tests[k] = lab_tests[k].map(lambda x: change_format(change_format(x)))
    feature_value_dict[k] = lab_tests[k].unique()

# In[ ]:


def nan_and_not_nan(x):
    if x == x:
        return 1
    else: # nan
        return 0

def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def is_all_float(x):
    for i in x:
        if i == i and (i != None):
            if not is_float(i):
                return False
    return True

def to_float(x):
    if x != None:
        return float(x)
    else:
        return np.nan

other_feature_dict = dict()

for feature in tqdm(feature_value_dict.keys()):
    values = feature_value_dict[feature]
    if is_all_float(values):
        lab_tests[feature] = lab_tests[feature].map(lambda x: to_float(x))
    elif len(values) == 2:
        lab_tests[feature] = lab_tests[feature].map(lambda x: nan_and_not_nan(x))
    else:
        other_feature_dict[feature] = values

# In[ ]:


def format_time(t):
    if '/' in t:
        return str(datetime.datetime.strptime(t, '%d/%m/%Y %H:%M'))
    else:
        return str(datetime.datetime.strptime(t, '%d-%m-%Y %H:%M'))

lab_tests['RECORD_TIME'] = lab_tests['LAB_DATE'] + ' ' + lab_tests['TIME_LAB']
lab_tests['RECORD_TIME'] = lab_tests['RECORD_TIME'].map(lambda x: format_time(x))
lab_tests = lab_tests.drop(['LAB_NUMBER', 'LAB_DATE', 'TIME_LAB'], axis=1)
lab_tests.head()

# In[ ]:


lab_tests_patient = lab_tests.groupby(['PATIENT_ID'], dropna=True, as_index = False).mean(numeric_only=True)
print(len(lab_tests_patient))

# In[ ]:


patient_total = len(lab_tests_patient)
threshold = patient_total * 0.1
reserved_keys = []

for key in lab_tests_patient.keys():
    if lab_tests_patient[key].count() > threshold:
        reserved_keys.append(key)

print(len(reserved_keys))

# In[ ]:


reserved_keys.insert(1, 'RECORD_TIME')
lab_tests = lab_tests.groupby(['PATIENT_ID', 'RECORD_TIME'], dropna=True, as_index = False).mean()
lab_tests = lab_tests[reserved_keys]
lab_tests.head()

# In[ ]:


lab_tests.to_csv(os.path.join(data_dir, 'processed', 'lab_test.csv'), index=False)
lab_tests.head()

# # Concat data

# In[ ]:


demographic['PATIENT_ID'] = demographic['PATIENT_ID'].map(lambda x: str(int(x)))
vital_signs['PATIENT_ID'] = vital_signs['PATIENT_ID'].map(lambda x: str(int(x)))
lab_tests['PATIENT_ID'] = lab_tests['PATIENT_ID'].map(lambda x: str(int(x)))

# In[ ]:


len(demographic['PATIENT_ID'].unique()), len(vital_signs['PATIENT_ID'].unique()), len(lab_tests['PATIENT_ID'].unique())

# In[ ]:


df = pd.merge(vital_signs, lab_tests, on=['PATIENT_ID', 'RECORD_TIME'], how='outer')
df = df.groupby(['PATIENT_ID', 'RECORD_TIME'], dropna=True, as_index = False).mean()
df = pd.merge(demographic, df, on=['PATIENT_ID'], how='left')

df.head()

# In[ ]:


# del rows without patient_id, admission_date, record_time, or outcome
df = df.dropna(axis=0, how='any', subset=['PATIENT_ID', 'ADMISSION_DATE', 'RECORD_TIME', 'OUTCOME'])

# In[ ]:


df.to_csv(os.path.join(data_dir, 'processed', 'cdsl_dataset_all.csv'), index=False)
df.describe()


# ## Export to unified CSV dataset format 

# - features: demographic & lab test & vital signs
# - targets: outcome & length of stay

# In[ ]:


patient_ids = df['PATIENT_ID'].unique()

demo_cols = ['AGE', 'SEX'] # , 'DIFFICULTY_BREATHING', 'FEVER', 'SUSPECT_COVID', 'EMERGENCY'
test_cols = []

# get column names
for k in df.keys():
    if not k in demographic.keys():
        if not k == 'RECORD_TIME':
            test_cols.append(k)

test_median = df[test_cols].median()

# In[ ]:


df['RECORD_TIME_DAY'] = df['RECORD_TIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
df['RECORD_TIME_HOUR'] = df['RECORD_TIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H'))
df.head()

# In[ ]:


df_day = df.groupby(['PATIENT_ID', 'ADMISSION_DATE', 'DEPARTURE_DATE', 'RECORD_TIME_DAY'], dropna=True, as_index = False).mean(numeric_only=True)
df_hour = df.groupby(['PATIENT_ID', 'ADMISSION_DATE', 'DEPARTURE_DATE', 'RECORD_TIME_HOUR'], dropna=True, as_index = False).mean(numeric_only=True)

len(df), len(df_day), len(df_hour)

# Number of visits (total):
# 
# - Original data: 168777
# - Merge by hour: 130141
# - Merge by day:  42204

# In[ ]:


len(df['PATIENT_ID'].unique())

# In[ ]:


df_hour['LOS'] = df_hour['ADMISSION_DATE']
df_hour['LOS_HOUR'] = df_hour['ADMISSION_DATE']

# In[ ]:


df_hour = df_hour.reset_index()

# In[ ]:


for idx in tqdm(range(len(df_hour))):
    info = df_hour.loc[idx]
    admission = datetime.datetime.strptime(info['ADMISSION_DATE'], '%Y-%m-%d %H:%M:%S')
    departure = datetime.datetime.strptime(info['DEPARTURE_DATE'], '%Y-%m-%d %H:%M:%S')
    visit_hour = datetime.datetime.strptime(info['RECORD_TIME_HOUR'], '%Y-%m-%d %H')
    hour = (departure - visit_hour).seconds / (24 * 60 * 60) + (departure - visit_hour).days
    los = (departure - admission).seconds / (24 * 60 * 60) + (departure - admission).days
    df_hour.at[idx, 'LOS'] = float(los)
    df_hour.at[idx, 'LOS_HOUR'] = float(hour)

# In[ ]:


df_hour_idx = df_hour.reset_index()

# In[ ]:


df_hour_idx['LOS'] = df_hour_idx['ADMISSION_DATE']

for idx in tqdm(range(len(df_hour_idx))):
    info = df_hour_idx.loc[idx]
    # admission = datetime.datetime.strptime(info['ADMISSION_DATE'], '%Y-%m-%d %H:%M:%S')
    departure = datetime.datetime.strptime(info['DEPARTURE_DATE'], '%Y-%m-%d %H:%M:%S')
    visit_hour = datetime.datetime.strptime(info['RECORD_TIME_HOUR'], '%Y-%m-%d %H')
    hour = (departure - visit_hour).seconds / (24 * 60 * 60) + (departure - visit_hour).days
    df_hour_idx.at[idx, 'LOS'] = float(hour)

# In[ ]:


df_hour['LOS'] = df_hour['LOS_HOUR']
df_hour.drop(columns=['LOS_HOUR'])

# In[ ]:


df = df_hour
df.head()

# In[ ]:


df['LOS'] = df['LOS'].clip(lower=0)

# In[ ]:


index = df.loc[0].index

csv = dict()
for key in ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime', 'Outcome', 'LOS', 'Sex', 'Age']:
    csv[key] = []
for key in index[8:-2]:
    csv[key] = []
    
for pat in tqdm(patient_ids): # for all patients
    # get visits for pat.id == PATIENT_ID
    info = df[df['PATIENT_ID'] == pat]
    info = info[max(0, len(info) - 76):]
    idxs = info.index
    for i in idxs:
        visit = info.loc[i]
        for key in index[8:-2]:
            csv[key].append(visit[key])
        # ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime', 'Outcome', 'LOS', 'Sex', 'Age']
        csv['PatientID'].append(visit['PATIENT_ID'])
        t, h = visit['RECORD_TIME_HOUR'].split()
        t = t.split('-')
        csv['RecordTime'].append(t[1]+'/'+t[2]+'/'+t[0]+' '+h) # 2020-04-06 10 -> 04/06/2020 10
        t = visit['ADMISSION_DATE'][:10].split('-')
        csv['AdmissionTime'].append(t[1]+'/'+t[2]+'/'+t[0])
        t = visit['DEPARTURE_DATE'][:10].split('-')
        csv['DischargeTime'].append(t[1]+'/'+t[2]+'/'+t[0])
        csv['Outcome'].append(visit['OUTCOME'])
        csv['LOS'].append(visit['LOS_HOUR'])
        csv['Sex'].append(visit['SEX'])
        csv['Age'].append(visit['AGE'])


# ### Export data to files

# In[ ]:


pd.DataFrame(csv).to_csv(os.path.join(data_dir, 'processed', 'cdsl_dataset_formatted.csv'), index=False)

# In[3]:


df = pd.read_csv(os.path.join(data_dir, 'processed', 'cdsl_dataset_formatted.csv'))

# ### Record feature names

# In[2]:


basic_records = ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime']
target_features = ['Outcome', 'LOS']
demographic_features = ['Sex', 'Age']
labtest_features = ['MAX_BLOOD_PRESSURE', 'MIN_BLOOD_PRESSURE', 'TEMPERATURE', 'HEART_RATE', 'OXYGEN_SATURATION', 'ADW -- Coeficiente de anisocitosis', 'ADW -- SISTEMATICO DE SANGRE', 'ALB -- ALBUMINA', 'AMI -- AMILASA', 'AP -- ACTIVIDAD DE PROTROMBINA', 'APTT -- TIEMPO DE CEFALINA (APTT)', 'AU -- ACIDO URICO', 'BAS -- Bas¢filos', 'BAS -- SISTEMATICO DE SANGRE', 'BAS% -- Bas¢filos %', 'BAS% -- SISTEMATICO DE SANGRE', 'BD -- BILIRRUBINA DIRECTA', 'BE(b) -- BE(b)', 'BE(b)V -- BE (b)', 'BEecf -- BEecf', 'BEecfV -- BEecf', 'BT -- BILIRRUBINA TOTAL', 'BT -- BILIRRUBINA TOTAL                                                               ', 'CA -- CALCIO                                                                          ', 'CA++ -- Ca++ Gasometria', 'CHCM -- Conc. Hemoglobina Corpuscular Media', 'CHCM -- SISTEMATICO DE SANGRE', 'CK -- CK (CREATINQUINASA)', 'CL -- CLORO', 'CREA -- CREATININA', 'DD -- DIMERO D', 'EOS -- Eosin¢filos', 'EOS -- SISTEMATICO DE SANGRE', 'EOS% -- Eosin¢filos %', 'EOS% -- SISTEMATICO DE SANGRE', 'FA -- FOSFATASA ALCALINA', 'FER -- FERRITINA', 'FIB -- FIBRINàGENO', 'FOS -- FOSFORO', 'G-CORONAV (RT-PCR) -- Tipo de muestra: Exudado Far¡ngeo/Nasofar¡ngeo', 'GGT -- GGT (GAMMA GLUTAMIL TRANSPEPTIDASA)', 'GLU -- GLUCOSA', 'GOT -- GOT (AST)', 'GPT -- GPT (ALT)', 'HCM -- Hemoglobina Corpuscular Media', 'HCM -- SISTEMATICO DE SANGRE', 'HCO3 -- HCO3-', 'HCO3V -- HCO3-', 'HCTO -- Hematocrito', 'HCTO -- SISTEMATICO DE SANGRE', 'HEM -- Hemat¡es', 'HEM -- SISTEMATICO DE SANGRE', 'HGB -- Hemoglobina', 'HGB -- SISTEMATICO DE SANGRE', 'INR -- INR', 'K -- POTASIO', 'LAC -- LACTATO', 'LDH -- LDH', 'LEUC -- Leucocitos', 'LEUC -- SISTEMATICO DE SANGRE', 'LIN -- Linfocitos', 'LIN -- SISTEMATICO DE SANGRE', 'LIN% -- Linfocitos %', 'LIN% -- SISTEMATICO DE SANGRE', 'MG -- MAGNESIO', 'MONO -- Monocitos', 'MONO -- SISTEMATICO DE SANGRE', 'MONO% -- Monocitos %', 'MONO% -- SISTEMATICO DE SANGRE', 'NA -- SODIO', 'NEU -- Neutr¢filos', 'NEU -- SISTEMATICO DE SANGRE', 'NEU% -- Neutr¢filos %', 'NEU% -- SISTEMATICO DE SANGRE', 'PCO2 -- pCO2', 'PCO2V -- pCO2', 'PCR -- PROTEINA C REACTIVA', 'PH -- pH', 'PHV -- pH', 'PLAQ -- Recuento de plaquetas', 'PLAQ -- SISTEMATICO DE SANGRE', 'PO2 -- pO2', 'PO2V -- pO2', 'PROCAL -- PROCALCITONINA', 'PT -- PROTEINAS TOTALES', 'SO2C -- sO2c (Saturaci¢n de ox¡geno)', 'SO2CV -- sO2c (Saturaci¢n de ox¡geno)', 'TCO2 -- tCO2(B)c', 'TCO2V -- tCO2 (B)', 'TP -- TIEMPO DE PROTROMBINA', 'TROPO -- TROPONINA', 'U -- UREA', 'VCM -- SISTEMATICO DE SANGRE', 'VCM -- Volumen Corpuscular Medio', 'VPM -- SISTEMATICO DE SANGRE', 'VPM -- Volumen plaquetar medio', 'VSG -- VSG']

require_impute_features = labtest_features
normalize_features = ['Age'] + labtest_features + ['LOS']

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
# - For CDSL dataset, use 7:1:2 splitting strategy, 70% training, 10% validation, 20% testing
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
# - For CDSL dataset, use 7:1:2 splitting strategy, 70% training, 10% validation, 20% testing
# 
# **time order**
# 

# In[4]:


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
