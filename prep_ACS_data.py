#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_ACS_data.py
    python script to prepare ACS data and merge with CEJST data to match with tract-level PM2.5 estimates
Created on Wed Sep  7 14:12:47 2022
@author: kodell
v0 - inital code
"""
#%% user inputs
ACS_data_path = '/Users/kodell/Library/CloudStorage/GoogleDrive-kodell@email.gwu.edu/My Drive/Ongoing Projects/GeoXO/population_data/ACS_IPUMS/'
CEJST_data_path = '/Users/kodell/Library/CloudStorage/GoogleDrive-kodell@email.gwu.edu/My Drive/Ongoing Projects/GeoXO/population_data/CEJST_data/1.0-communities.csv'
out_fn = ACS_data_path + 'combined_trimmed_files_v1.csv'

#%% import modules
import pandas as pd
import numpy as np

#%% load CEJST data v 1.0
# available, https://screeningtool.geoplatform.gov/en/downloads#3/33.47/-97.5
cejst_df = pd.read_csv(CEJST_data_path,usecols = ['Census tract 2010 ID','State/Territory','Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?'])
# note this is on 2010 census tract geometries, but NHGIS data is 2019 Tigerline files

#%% load ACS data, downloaded from NHGIS https://www.nhgis.org/
# race and ethnicity
# Hispanic or Latino Origin by Race, ACS code: B03002, NHGIS code: ALUK
race_fn = 'nhgis0014_csv/nhgis0014_ds244_20195_tract_E.csv'
# educational attainment 
# Educational Attainment for the Population 25 Years and Over, ACS code: B15003, NHGIS code: ALWG
ed_fn = 'nhgis0016_csv/nhgis0016_ds244_20195_tract_E.csv'
# language spoken at home 
# Language Spoken at Home for the Population 5 Years and Over, ACS code: C16001, NHGIS code: AMBB
lng_fn = 'nhgis0015_csv/nhgis0015_ds245_20195_tract_E.csv'

# to be honest, not entirely sure why we need "latin1". python says the file encoding is utf-8
# but that doesn't work. stack overflow suggested this based on the error message and it worked.
# *shrug emoji* 
# https://stackoverflow.com/questions/59751700/unicodedecodeerror-utf-8-codec-cant-decode-byte-0xf1-in-position-2-invalid
race_df = pd.read_csv(ACS_data_path + race_fn, skiprows=1,engine='python',encoding='latin1')
ed_df = pd.read_csv(ACS_data_path + ed_fn,skiprows=1,engine='python',encoding='latin1')
lng_df = pd.read_csv(ACS_data_path + lng_fn, skiprows=1,engine='python',encoding='latin1')

#%% calculate values we need, race/ethnicity table
# variables we need
race_subset_vars = ['Census Geographic Identifier',
       'Total', 'Not Hispanic or Latino: White alone','Hispanic or Latino',
       'Not Hispanic or Latino: Black or African American alone',
       'Not Hispanic or Latino: American Indian and Alaska Native alone', 'Not Hispanic or Latino: Asian alone',
       'Not Hispanic or Latino: Native Hawaiian and Other Pacific Islander alone',
       'Not Hispanic or Latino: Some other race alone','Not Hispanic or Latino: Two or more races',
       'Not Hispanic or Latino: Two or more races: Two races including Some other race',
       'Not Hispanic or Latino: Two or more races: Two races excluding Some other race, and three or more races']
race_df_trim = race_df[race_subset_vars].copy()
# combine variables
race_df_trim['Not Hispanic or Latino: Asian or Pacific Islander'] = race_df_trim['Not Hispanic or Latino: Asian alone'] + race_df_trim['Not Hispanic or Latino: Native Hawaiian and Other Pacific Islander alone']
race_df_trim['Not Hispanic or Latino: Other'] = race_df_trim[['Not Hispanic or Latino: Some other race alone',
                                                              'Not Hispanic or Latino: Two or more races']].sum(axis=1)
# drop combined variables
drop_cols = ['Not Hispanic or Latino: Asian alone','Not Hispanic or Latino: Native Hawaiian and Other Pacific Islander alone',
             'Not Hispanic or Latino: Some other race alone','Not Hispanic or Latino: Two or more races',
             'Not Hispanic or Latino: Two or more races: Two races including Some other race',
             'Not Hispanic or Latino: Two or more races: Two races excluding Some other race, and three or more races']
race_df_trim.drop(labels=drop_cols,axis='columns',inplace=True)

# check that these sum to the expexted total
diff = race_df_trim['Total'].values - race_df_trim[['Not Hispanic or Latino: White alone','Hispanic or Latino',
'Not Hispanic or Latino: Black or African American alone','Not Hispanic or Latino: American Indian and Alaska Native alone',
'Not Hispanic or Latino: Asian or Pacific Islander','Not Hispanic or Latino: Other']].sum(axis=1)
print('should be zeros:',diff.max(),diff.min())

#%% calculate values we need, language
# list variables to combine for each category we want
# less than high school
lt_hs_vars = ['No schooling completed','Nursery school','Kindergarten','1st grade','2nd grade','3rd grade',
           '4th grade','5th grade','6th grade','7th grade','8th grade','9th grade','10th grade','11th grade',
           '12th grade, no diploma']
# high school
hs_vars = ['Regular high school diploma','GED or alternative credential','Some college, less than 1 year',
           'Some college, 1 or more years, no degree',"Associate's degree"]
# bachelors
b_vars = ["Bachelor's degree"]
#grad
g_vars = ["Master's degree","Professional school degree","Doctorate degree"]

# create trimmed dataframe and sum across above variables
ed_trim = ed_df[['Census Geographic Identifier']].copy()
ed_trim.loc[:,'pop_25andup'] = ed_df.loc[:,'Total'].values
ed_trim.loc[:,'lt_hs'] = ed_df[lt_hs_vars].sum(axis=1)
ed_trim.loc[:,'hs'] = ed_df[hs_vars].sum(axis=1)
ed_trim.loc[:,'bachelors'] = ed_df[b_vars].sum(axis=1)
ed_trim.loc[:,'grad'] = ed_df[g_vars].sum(axis=1)

# check that these sum to the expected total
diff = ed_trim['pop_25andup'].values - ed_trim[['lt_hs','hs','bachelors','grad']].sum(axis=1)
print('should be zeros:',diff.max(),diff.min())

#%% calculate values we need, English profeciency
# create trimmed array
lng_trim = lng_df[['Census Geographic Identifier']].copy()
# variables to sum for speak english very well
very_well_vars = ["Spanish: Speak English 'very well'",
                "French, Haitian, or Cajun: Speak English 'very well'",
                "German or other West Germanic languages: Speak English 'very well'",
                "Russian, Polish, or other Slavic languages: Speak English 'very well'",
                "Other Indo-European languages: Speak English 'very well'",
                "Korean: Speak English 'very well'",
                "Chinese (incl. Mandarin, Cantonese): Speak English 'very well'",
                "Vietnamese: Speak English 'very well'",
                "Tagalog (incl. Filipino): Speak English 'very well'",
                "Other Asian and Pacific Island languages: Speak English 'very well'",
                "Arabic: Speak English 'very well'",
                "Other and unspecified languages: Speak English 'very well'"]
# variables to sum for speak english less than very well
lt_very_well_vars = ["Spanish: Speak English less than 'very well'",
                "French, Haitian, or Cajun: Speak English less than 'very well'",
                "German or other West Germanic languages: Speak English less than 'very well'",
                "Russian, Polish, or other Slavic languages: Speak English less than 'very well'",
                "Other Indo-European languages: Speak English less than 'very well'",
                "Korean: Speak English less than 'very well'",
                "Chinese (incl. Mandarin, Cantonese): Speak English less than 'very well'",
                "Vietnamese: Speak English less than 'very well'",
                "Tagalog (incl. Filipino): Speak English less than 'very well'",
                "Other Asian and Pacific Island languages: Speak English less than 'very well'",
                "Arabic: Speak English less than 'very well'",
                "Other and unspecified languages: Speak English less than 'very well'"]
# sum across above variables and add to new array
lng_trim.loc[:,'very_well'] = lng_df[very_well_vars].sum(axis=1)
lng_trim.loc[:,'lt_very_well'] = lng_df[lt_very_well_vars].sum(axis=1)
lng_trim.loc[:,'eng_only'] = lng_df['Speak only English'].values
lng_trim.loc[:,'pop_5andup'] = lng_df['Total'].values

# again, check that these sum to our expected total
diff = lng_trim['pop_5andup'].values - lng_trim[['very_well','lt_very_well','eng_only']].sum(axis=1)
print('should be zeros:',diff.max(),diff.min())

#%% combine data and add geoid to match tract level pm data
# first ACS data
acs_comb1 = lng_trim.merge(ed_trim,'left','Census Geographic Identifier')
acs_comb = acs_comb1.merge(race_df_trim,'left','Census Geographic Identifier')

# fix changes in census ids since 2010, ACS data in NHGIS uses 2019 tigerline tracts
cejst_df.loc[60108,'Census tract 2010 ID'] = 46102940500
cejst_df.loc[60109,'Census tract 2010 ID'] = 46102940800
cejst_df.loc[60110,'Census tract 2010 ID'] = 46102940900
cejst_df.loc[68970,'Census tract 2010 ID'] = 51019050100

# now let's add cejest data
# adjust ids to allign, removing 14000US from begining of ACS tract ids
us_id = []
for geoid in acs_comb['Census Geographic Identifier'].values:
    us_id.append(int(geoid.split('S')[1]))
acs_comb['geoid'] = us_id

# finally, merge with cejst
comb_all = acs_comb.merge(cejst_df,'left',left_on='geoid',right_on='Census tract 2010 ID')

# check for geoids that didn't copy over
# will be assigned a nan, there are no nans in the og cejst dataset
nans = np.where(pd.isna(comb_all['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?']))
# a few additional census tracts (21) are also nans but it's not due to documented 
# census tract label changes (https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.2010.html#list-tab-957819518)
# not sure what is causing this, but the number is so small it doesn't impact results

#%% save to csv
comb_all.to_csv(out_fn)

# we will plot and check these in the next plotting code!
