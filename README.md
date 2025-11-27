# Cooling Effects of Large Urban Mountains  
### Code and Workflow for the Chengdu Longquan Mountains Urban Forest Park Study

This repository contains all scripts, workflow descriptions, and supplementary materials used in the study:

**“Cooling Effects in Large Urban Mountains: A Case Study of Chengdu Longquan Mountains Urban Forest Park, China.”**

<img width="1119" height="1474" alt="graphical abstract forests-3996816" src="https://github.com/user-attachments/assets/53b0b153-44d9-4586-87c4-bba0eab7332e" />


1. Overview

This repository contains the code, datasets, and analysis for the paper titled "Cooling Effects in Large Urban Mountains: A Case Study of Chengdu Longquan Mountains Urban Forest Park." The study investigates the cooling effects of the Longquan Mountain Forest Park (LMFP) and its role in mitigating the Urban Heat Island (UHI) effect. The main focus is on the spatiotemporal variation of cooling effects and their driving factors.

2. Files in the Repository

/data: Contains all the raw and processed data, including LST datasets, vegetation data, and environmental factors.

/scripts: Python and R scripts for data processing, analysis, and model building.

data_preprocessing.py: Scripts for preprocessing various datasets (e.g., vegetation indices, road density).

xgboost_model.py: Code for training the XGBoost model to predict cooling intensity.

shap_analysis.py: Code for generating SHAP plots to explain the model.

3. Installation and Setup

To use the scripts, ensure you have the following Python libraries installed:

numpy

pandas

matplotlib

seaborn

xgboost

shap

scikit-learn

geopandas

rasterio

You can install the necessary libraries using pip:

pip install numpy pandas matplotlib seaborn xgboost shap scikit-learn geopandas rasterio

4. Data Access

The datasets used in the study are publicly available:

Landsat Data: Available from the USGS Earth Explorer website.

Vegetation and Environmental Data: Available from Google Earth Engine and the USGS DEM data repository.

5. Methodology

A brief summary of the methodology used in the study:

LST Retrieval: Land Surface Temperature (LST) was retrieved from Landsat imagery for three time periods (2001, 2011, 2023). The radiative transfer equation was used to convert thermal infrared data into LST.

XGBoost Model: An XGBoost model was trained using various environmental factors (e.g., vegetation cover, elevation, population density) to predict cooling intensity.

SHAP Analysis: The SHAP method was applied to the model to understand the contributions of individual features.

6. Reproducibility

To reproduce the results, follow these steps:

Download the required raw data (Landsat imagery, vegetation indices, etc.) from the specified data sources.

Run the preprocessing scripts to clean and prepare the data.

Train the XGBoost model using the provided code.

7. Acknowledgments

The authors would like to thank the following data providers for their contributions:

USGS for Landsat imagery.

Oak Ridge National Laboratory for population density data.

Google Earth Engine for vegetation indices and evapotranspiration data.

8. License

This repository is open source and available under the MIT License.

Folder Structure
Cooling_Effects_Large_Urban_Mountains/
│
├── data/
│   ├── 20010614_lst.tif
│   ├── 20110630_lst.tif
│   ├── 20230705_lst.tif
│   ├── vegetation_data.csv
│   └── human_activity_data.csv
│
├── scripts/
│   ├── LST_retrieval.py
│   ├── data_preprocessing.py
│   ├── xgboost_model.py
│   └── shap_analysis.py

│
└── README.md


This structure is clear, supports reproducibility, and ensures that all aspects of the research workflow (data, preprocessing, modeling, and analysis) are available to others in a transparent way.

