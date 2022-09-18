# Introduction

This repository contains software and data for "[Adapting Neural Networks for the Estimation of Treatment Effects](https://arxiv.org/pdf/1906.02120.pdf)".

The paper describes approaches to estimating causal effects from observational data using neural networks. The high-level idea is to modify standard neural net design and training in order to induce a bias towards accurate estimates.

# Requirements and setup
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/" 

export PYTHONPATH="${PYTHONPATH}:/Users/jiaweizhang/med/dragonnet"

conda create -n dragonnet python=3.7 

conda activate dragonnet 

go into the src directory 

python experiment/mbqip_main.py

You will need to install tensorflow 1.13, sklearn, numpy 1.15, keras 2.2.4 and, pandas 0.24.1

# Data
From my google drive

# Reproducing neural net training for IHDP experiments
The default setting would let you run Dragonnet, TARNET, and NEDnet under targeted regularization and default mode

You'll run the from `src` code as 
`./experiment/run_ihdp.sh`
Before doing this, you'll need to edit `run_ihdp.sh` and change the following:
`data_base_dir= where you stored the data`
`output_base_dir=wherer you want the result to be`

If you only want to run one of the frameworks, delete the rest of the options in `run_ihdp.sh`

# Reproducing neural net training for the ACIC experiment
Same as above except you run the from `src` code as `./experiment/run_acic.sh`

# Computing the ATE
All of the estimators functions are in `semi_parametric_estimation.ate`

To reproduce the table in the paper: i) get the neural net predictions; ii) update the output file location in `ihdp_ate.py` iii) run `ihdp_ate.py`. The `make_table` function should generate the mean absolute error for each framework. 

Note: the default code use all the data for prediction and estimation. If you want to get the in-sample or out-sample error: i) change the `train_test_split` criteria in `ihdp_main.py`; ii) rerun the neural net training; iii) run `ihdp_ate.py` with apporiate in-sample data and out-sample data. 

# MBQIP data
CASEID,AGE,SEX_b'Male',race_PUF_b'American Indian or Alaska Native',race_PUF_b'Asian',race_PUF_b'Black or African American',race_PUF_b'Native Hawaiian or Other Pacific Islander',race_PUF_b'Unknown/Not Reported',race_PUF_b'White',CPTUNLISTED_REVCONV_1.0,CPTUNLISTED_GASPLICATION_1.0,GERD_b'Yes',MOBILITY_DEVICE_b'Yes',MI_ALL_HISTORY_b'Yes',PTC_b'Yes',PCARD_b'Yes',HIP_b'Yes',HTN_MEDS_b'0',HTN_MEDS_b'1',HTN_MEDS_b'2',HTN_MEDS_b'3+',HYPERLIPIDEMIA_b'Yes',HISTORY_DVT_b'Yes',VENOUS_STASIS_b'Yes',DIALYSIS_b'Yes',RENAL_INSUFFICIENCY_b'Yes',THERAPEUTIC_ANTICOAGULATION_b'Yes',PREVIOUS_SURGERY_b'Yes',DIABETES_b'Insulin',DIABETES_b'No',DIABETES_b'Non-Insulin',SMOKER_b'Yes',FUNSTATPRESURG_b'Independent',FUNSTATPRESURG_b'Partially Dependent',FUNSTATPRESURG_b'Totally Dependent',COPD_b'Yes',OXYGEN_DEPENDENT_b'Yes',HISTORY_PE_b'Yes',SLEEP_APNEA_b'Yes',CHRONIC_STEROIDS_b'Yes',IVC_FILTER_b'Yes',SWALLOW_STUDY_b'No',"SWALLOW_STUDY_b'Yes, routine'","SWALLOW_STUDY_b'Yes, selective'",ASACLASS_b'1-No Disturb',ASACLASS_b'2-Mild Disturb',ASACLASS_b'3-Severe Disturb',ASACLASS_b'4-Life Threat',ASACLASS_b'5-Moribund',REOP30_b'Yes',treament
