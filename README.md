# README
This repository contains scripts and python3 environments used to produce the analysis and figures in  
"Short-term pollution exposure and environmental justice implications" by O’Dell et al. submitted ES&T Letters (Manuscript ID: ez-2023-00548t) as of 07.31.23. 
If you have any questions about these codes, want to use them, or find any errors, please contact Kate O’Dell at the contact email provided with this github page. 
All scripts in this folder were written by Kate O'Dell unless otherwise specified.

## Instructions
To complete the analysis and create figures in the paper, the python scripts in this folder are run in the following order. 
Note that the local and remote codes use different python environments. 
Both environments are provided in this folder and are entitled local_py_env.yml and remote_py_env.yml, for the local and remote environments, respectively. 

### Step 1) average gridded data to the census tract level
#### Avg_to_census_tract.py
Averages gridded ABI-daytime and ABI-1pm annual mean surface PM2.5 and alert day counts to the census tract level.
This code is run remotely and run separately for ABI-daytime and ABI-1pm.
This code relies heavily on census tract averaging code shared by Dr. Gagie Kerr and available https://github.com/gaigekerr/edf/blob/main/harmonize_afacs.py 
##### Inputs:
- Gridded ABI-daytime (or ABI-1pm) annual concentrations, alert day counts, and observations counts output from https://github.com/kaodell/GeoXO/blob/main/calc_annual_alert_HIA.py 
- 2019 state-level census tract shapefiles for all US states, available: https://www2.census.gov/geo/tiger/TIGER2019/TRACT/ 
#### Outputs:
- csv of annual mean PM2.5, alert day counts, and observation counts estimated at the census tract level

### Step 2) prepare demographics datasets
#### prep_ACS_data.py
Loads data from the American Community Survey (ACS) and the Climate and Economic Justice Screening Tool (CEJST), combines categories, and combines different datasets into a single array.
This script is run locally.

#### Inputs:
- CEJST version 1.0 data, available https://screeningtool.geoplatform.gov/en/downloads
- 2015-2019 5-year American community survey data at the census tract level downloaded from NHGIS here, https://data2.nhgis.org/main . Specifically, the following tables are downloaded:
  (1) Hispanic or Latino Origin by Race, ACS code: B03002, NHGIS code: ALUK
(2) Language Spoken at Home for the Population 5 Years and Over, ACS code: C16001, NHGIS code: AMBB
(3) Educational Attainment for the Population 25 Years and Over, ACS code: B15003, NHGIS code: ALWG
#### Outputs:
- Csv with the demographic variables we use for the present analysis with the 2019 census tract ids from the ACS dataset

### Step 3) run analysis and create figures in manuscript
#### match_EJ_alerts.py
Loads data output from above scripts, aligns pm and demographic data, and makes all figures in the manuscript. This script is run locally.
#### Inputs:
- ABI-daytime annual mean PM2.5 and alert day counts output from avg_to_census_tract.py
- ABI-1pm annual mean PM2.5 and alert day counts output from avg_to_census_tract.py
- ACS and CEJST data csv output from prep_ACS_data.py
- 2019 tigerline census tract shapefile downloaded from NHGIS, https://data2.nhgis.org/main
- CEJST v1.0 census tract shapefile available, https://screeningtool.geoplatform.gov/en/downloads#3/33.47/-97.5 
#### Outputs:
- All figures in the manuscript

Last updated 07.28.23
