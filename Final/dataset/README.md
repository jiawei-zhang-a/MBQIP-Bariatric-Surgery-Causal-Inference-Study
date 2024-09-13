### Dataset Overview

Our dataset originates from the Medicare Beneficiary Quality Improvement Project (MBQIP), covering the years 2015 to 2020. This dataset was subject to basic data cleaning, where extreme outliers were removed and relevant data extracted. The data cleaning process is presented in the `data_cleaning.ipynb` notebook. After cleaning, the resulting dataset, `all_bmi&main_data.csv`, contains all the necessary data for further analysis.

### Data Processing Workflow

After obtaining `all_bmi&main_data.csv`, the data will be used in two separate preprocessing scripts:

1. **Effect Processing:**
   - **File:** `mbqip_pre_data_processing_effect.py`
   - **Purpose:** This script focuses on analyzing the effect of surgery, particularly by calculating the difference in BMI, which represents the effect of the surgical intervention.

2. **Risk Processing:**
   - **File:** `mbqip_pre_data_processing_risk.py`
   - **Purpose:** This script assesses intervention, reoperation, readmission, and death associated with the surgeries to measure the risk from different aspects.

### Output Datasets

The processing scripts will generate five folders, each corresponding to different categories of surgical procedures. Within each folder, there will be four datasets, named as follows:

- **`mbqip_1.csv`** – Represents the effect of RYGB surgery.
- **`mbqip_2.csv`** – Represents the effect of Band surgery.
- **`mbqip_3.csv`** – Represents the effect of BPD surgery.
- **`mbqip_4.csv`** – Represents the effect of SADI-S surgery.

These files will contain the processed data necessary for further analysis of both the effects and risks associated with each type of surgery.

