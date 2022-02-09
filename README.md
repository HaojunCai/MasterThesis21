# Master Thesis: EV predictions & Charging Strategies Simulation and Evaluation
Master Thesis, Spring 2021, Institute of Cartography and Geoinformatics at ETH Zurich 

Author: Haojun Cai (haojuncai1996@gmail.com)

## Introduction
The research aimed to predict the next-day energy consumption and parking duration of 113 electric vehicles (EV) using three probabilistic models: linear quantile regression, quantile random forest, and gradient boosting quantile regression. Furthermore, two time-shifting smart charging strategies were simulated based on the prediction results to evaluate the technical benefits brought to the grid system and monetary gains brought to the EV users. The prediction results have shown that the mobility features of EV users could help the prediction in some cases and better prepare for the two time-shift charging strategies.

Poster: https://ethz.ch/content/dam/ethz/special-interest/baug/ikg/geoinformation-engineering-dam/Lehre/Studienarbeiten/MSc/21_FS_Cai.pdf

## Getting Started

In order to run the whole pipelie, you need to run the file main.py. It requires the python 3.

### Prerequisites

The following python packages are required: 
```
* pandas
* numpy
* geopandas
* sqlalchemy
* datetime
* trackintel
* skmob
* haversine
* csv
* matplotlib
* statsmodels
* skgarden (! not under good maintainence currently, consider to install directly by git install command)
* sklearn
* scipy
* sys
* math
* os
```

### File Structure
   - main.py: main function to run the whole pipeline
   - extract_mobility.py: extract daily mobility features
   - extract_evfeatures.py: extract ev-related features 
   - extract_soc.py: preprocess and extract energy consumption input and target features
   - extract_arrival.py: preprocess and extract arrival input and target featrues
   - extract_depart.py: preprocess and extract departure input and target featrues
   - predict_probablistic_results.py: run quantile regression predictions for three targets (energy consumption, arrival time, departure time)
   - calculate_under_overestimation.py: calculate overestimation and underestimation ratio
   - calculate_feature_importance.py: calculate feature importances for quantile random forest model
   - compare_probablistic_results.py: calculate evaluation metrics over all users by model 
   - evaluate_unidirectional_smartcharging.py: simulate unidirectional smart charging process
   - evaluate_bidirectional_smartcharging.py: simulate bidirectional smart charging process
   - evaluate_uncontrolled_charging.py: simulate uncontrolled charging process as baseline
   - compare_baseline_unismart.py: compare baseline and unidirectional smart charging
   - compare_baseline_bismart.py: compare baseline and nidirectional smart charging
   - compare_three_charging_onpeakdef2.py: plot load profile of three charging strategies using on-peak definition 2
