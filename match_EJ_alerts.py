#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match_EJ_alerts.py
    python script to conduct anlaysis at the tract level with demographic variables
    for an envrionmental justic (EJ) perspective
Created on Fri Jul 15 11:03:22 2022
@author: kodell
"""
#%% user inputs
# project folder which contains data output from prep_ACS_data.py, shapefiles for plotting,
# and dataset from Gaige
prj_folder = '/Users/kodell/Library/CloudStorage/GoogleDrive-kodell@email.gwu.edu/My Drive/Ongoing Projects/GeoXO/'

# annual mean pm2.5 at the census tract level data path
# output from avg_to_census_tract.py
pm_data_path = '/Users/kodell/Library/CloudStorage/Box-Box/Shobha_data/'

# demographic data by census tract
# output from prep_ACS_data.py
dem_fn = prj_folder+ 'population_data/ACS_IPUMS/combined_trimmed_files_v1.csv'

# pop-weighted mean PM from Gaige and NIFC wildland area burned
gaige_data_fp = prj_folder+ 'population_data/Gaige_data_wNIFC.csv'

# 2019 tigerline census tracts shapefile from NHGIS  to make map plots
shp_fp = prj_folder + 'population_data/ACS_IPUMS/nhgis0010_shape/nhgis0010_shapefile_tl2019_us_tract_2019/US_tract_2019.shp'
# cejst shapefile to use for plotting
cejst_shp_fn = prj_folder + 'population_data/CEJST_data/1.0-shapefile-codebook/usa/usa.shp'

# version to use and where to save figures
fig_desc = 'final'
fig_out_path = prj_folder + 'figures/EJ/'

# keys and assoc. colors for figures
# strings and colors for demographic data
dem_strs = ['Total','Not Hispanic or Latino: Black or African American alone',
                'Hispanic or Latino','Not Hispanic or Latino: White alone',
                'Not Hispanic or Latino: American Indian and Alaska Native alone',
                'Not Hispanic or Latino: Asian or Pacific Islander']
dem_strs_other=['Not Hispanic or Latino: Other','Not Hispanic or Latino: Black or African American alone',
                'Hispanic or Latino','Not Hispanic or Latino: White alone',
                'Not Hispanic or Latino: American Indian and Alaska Native alone',
                'Not Hispanic or Latino: Asian or Pacific Islander']
# color orders follow the strings above except for the first color. 
dem_colors = ['gray','#1b9e77', '#d95f02', '#e6ab02', '#e7298a', '#7570b3'] # 'other'is first color
bar_colors = ['black','#1b9e77', '#d95f02', '#e6ab02', '#e7298a', '#7570b3'] # 'total' is first color
# colors for educational attainment: <high school, high school, bachelors, grad
colors_ed = ['#bebada','#fdb462','#fccde5','#7fc97f']
# colors for english language proficiency: english only, very well, less than very well
colors_lng = ['#8dd3c7','#fb9a99','#80b1d3']
# colors for CEJST : disadvantaged, not disadvantaged
colors_cejst = ['royalblue','#b3b3b3']

#%% import modules
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

#%% load data
# v10 is most version used in submission of geoxo paper1
# v5 indicates version of avg_to_census_tract.py code used
# base indicates no reductions in pm2.5 on alert days are made for assumed behavior modification
abi_base = pd.read_csv(pm_data_path+'abi_og_v10_base.v5.csv')
abi1pm_base = pd.read_csv(pm_data_path+'abi1pm_proxy_v10_base.v5.csv')

# dmographic data 
# to be honest, not entirely sure why we need "latin1". python says the file encoding is utf-8
# but that doesn't work. stack overflow suggested this based on the error message and it worked.
# *shrug emoji* https://stackoverflow.com/questions/59751700/unicodedecodeerror-utf-8-codec-cant-decode-byte-0xf1-in-position-2-invalid
dem_df = pd.read_csv(dem_fn, engine='python',encoding='latin1')

# annual population-weighted mean pm2.5 and NIFC area burned by wildland fires
# from 2010-2020 tables from Dr. Gaige Kerr
# will likely add this csv to the github
gaige_data = pd.read_csv(gaige_data_fp)

#%% combine dataframes and add population, demographics
# first rename variables before merging
abi_base['abi_pm25_base'] = abi_base['pm25'].copy() # 1% are nans
abi_base['abi_alerts'] = abi_base['alerts'].copy() # only nans in AK and HI
abi1pm_base['abi1pm_pm25_base'] = abi1pm_base['pm25'].copy()
abi1pm_base['abi1pm_alerts'] = abi1pm_base['alerts'].copy()

# drop variables we dont need
for df in [abi_base, abi1pm_base]:
    df.drop(columns = ['pm25','Unnamed: 0','count'],inplace=True)
for df in [abi_base,abi1pm_base]:
    df.drop(columns = ['alerts'],inplace=True)

# make concentration nans also nans in the alert day dataset
# because these are locations with no observations from ABI-Daytime
# this only really makes a difference for montana in the state-level plot
# all other results remain the same
naninds = np.where(pd.isna(abi_base['abi_pm25_base']))
abi_base['abi_alerts'] = np.where(pd.isna(abi_base['abi_pm25_base']),np.nan,abi_base['abi_alerts'].values)
abi1pm_base['abi1pm_alerts'] = np.where(pd.isna(abi_base['abi_pm25_base']),np.nan,abi1pm_base['abi1pm_alerts'].values)

# merge using pd.merge function
# merge on the abi dataset
pm_a = pd.merge(dem_df,abi_base,'right','geoid')
pm_pop = pd.merge(pm_a,abi1pm_base,'left','geoid')

# remove areas outside contig US
pm_pop['SF'] = pm_pop['State/Territory']
states_rmv = ['Virgin Islands','Alaska','Hawaii','Guam','Northern Mariana Islands','American Samoa','Puerto Rico']
for st_str in states_rmv:
    pm_pop = pm_pop[pm_pop.SF!=st_str]
    pm_pop.reset_index(inplace=True,drop=True)
    
#%% Figure 1 - breakdown of categories by alert days
fig, axarr = plt.subplots(2,2,figsize=(9,8))
ax = axarr.flatten()
bins = np.arange(0,41,1)
bar_width = 0.8
# pre allocate arrays and counting variables to print numbers we report in the manuscript
tract_count = [] # count of tracts
pop_sums_ed = [] # educational attainment pop totals (pop >=25 years)
pop_sums_lang = [] # language pop totals (pop >=5 years)
pop_sums_tot = [] # full population totals
disadvantaged_pct = []

# for counting totals across the first 10 and last 10 bins
hisp_total_10, total_10, white_total_10, hisp_total_30, total_30, white_total_30 = 0,0,0,0,0,0
lths_10,b_10,g_10,total25_10,lths_30,b_30,g_30,total25_30 = 0,0,0,0,0,0,0,0
ltvw_10, total5_10, ltvw_30,total5_30 = 0,0,0,0

# loop through bins, plot, and count these statistics
for i in range(len(bins)):
    if i < (len(bins)-1):
        lbin = bins[i]
        rbin = bins[i+1]
    elif i == (len(bins)-1):
        lbin = bins[i]
        rbin = pm_pop['abi_alerts'].max() + 2
    inds = np.where(np.logical_and(pm_pop['abi_alerts'].values>=lbin,pm_pop['abi_alerts'].values<rbin))
    tract_count.append(len(inds[0]))
    inds = np.array(inds[0])

    # PANEL A - RACE and ETHNICIY
    tot_pop = np.nansum(pm_pop['Total'].iloc[inds])
    pop_sums_tot.append(tot_pop)
    ci = 0
    cat_total = 0
    for dem_str in dem_strs_other:
        dem_total = np.nansum(pm_pop[dem_str].iloc[inds])
        ax[0].bar(bins[i],100.0*dem_total/tot_pop,color = dem_colors[ci],bottom = cat_total,width=bar_width)
        cat_total += 100.0*dem_total/tot_pop
        ci += 1            
    # for paper calculate % <10 days and greater than 30 days
    if i <10: # for bins up to 10
        white_total_10 += np.nansum(pm_pop['Not Hispanic or Latino: White alone'].iloc[inds])
        hisp_total_10 +=  np.nansum(pm_pop['Hispanic or Latino'].iloc[inds])
        total_10 +=  np.nansum(pm_pop['Total'].iloc[inds])
    if i > 30: # for bins >30
        white_total_30 +=  np.nansum(pm_pop['Not Hispanic or Latino: White alone'].iloc[inds])
        hisp_total_30 +=  np.nansum(pm_pop['Hispanic or Latino'].iloc[inds])
        total_30 +=  np.nansum(pm_pop['Total'].iloc[inds])
      
    # PANEL B - DISADVANTAGED STATUS
    inds_true = inds[np.where(pm_pop['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?'].iloc[inds]==True)[0]]
    inds_false = inds[np.where(pm_pop['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?'].iloc[inds]==False)[0]]
    inds_nan = inds[np.where(pd.isna(pm_pop['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?'].iloc[inds]))[0]]
    true_total = np.nansum(pm_pop['Total'].iloc[inds_true])
    false_total = np.nansum(pm_pop['Total'].iloc[inds_false])
    nan_total = np.nansum(pm_pop['Total'].iloc[inds_nan])
    ax[1].bar(bins[i],100.0*true_total/tot_pop,color = colors_cejst[0],width=bar_width)
    ax[1].bar(bins[i],100.0*false_total/tot_pop,color = colors_cejst[1],bottom=100.0*true_total/tot_pop,width=bar_width)
    ax[1].bar(bins[i],100.0*nan_total/tot_pop,color = 'black',
              bottom = 100.0*(true_total+false_total)/tot_pop,width=bar_width)   
    # calcualte values for figure discussion in manuscript    
    disadvantaged_pct.append(100.0*true_total/tot_pop)

    # PANEL C - EDUCATION
    tot_pop_ed = np.nansum(pm_pop['pop_25andup'].iloc[inds])
    pop_sums_ed.append(tot_pop_ed)
    cat_total = 0
    vi = 0
    for var in ['lt_hs','hs','bachelors','grad']:
        var_total = np.nansum(pm_pop[var].iloc[inds])
        var_total_plot = 100.0*var_total/tot_pop_ed
        ax[2].bar(bins[i],var_total_plot,bottom = cat_total, color = colors_ed[vi],width=bar_width)   
        cat_total += var_total_plot
        vi+= 1
    # calcualte values for figure discussion in manuscript
    if i <10:
        lths_10 +=  np.nansum(pm_pop['lt_hs'].iloc[inds])
        b_10 +=  np.nansum(pm_pop['bachelors'].iloc[inds])
        g_10 +=  np.nansum(pm_pop['grad'].iloc[inds])
        total25_10 +=  np.nansum(pm_pop['pop_25andup'].iloc[inds])
    if i > 30:
        lths_30 +=  np.nansum(pm_pop['lt_hs'].iloc[inds])
        b_30 +=  np.nansum(pm_pop['bachelors'].iloc[inds])
        g_30 +=  np.nansum(pm_pop['grad'].iloc[inds])
        total25_30 +=  np.nansum(pm_pop['pop_25andup'].iloc[inds])
        
    # PANEL D - LANGUAGE
    tot_pop_lng = np.nansum(pm_pop['pop_5andup'].iloc[inds])
    eng_only_total = np.nansum(pm_pop['eng_only'].iloc[inds])
    vwell_total = np.nansum(pm_pop['very_well'].iloc[inds])
    lt_vwell_total = np.nansum(pm_pop['lt_very_well'].iloc[inds])
    pop_sums_lang.append(tot_pop_lng)
    ax[3].bar(bins[i],100.0*eng_only_total/tot_pop_lng,color = colors_lng[0],label = 'English Only',width=bar_width)
    ax[3].bar(bins[i],100.0*vwell_total/tot_pop_lng, bottom = 100.0*eng_only_total/tot_pop_lng,width=bar_width,
           color=colors_lng[1],label='Very Well')
    ax[3].bar(bins[i],100.0*lt_vwell_total/tot_pop_lng, bottom = 100.0*eng_only_total/tot_pop_lng + 100.0*vwell_total/tot_pop_lng,
           color=colors_lng[2],label = 'Less Than Very Well',width=bar_width)
    if i <10:
        ltvw_10 +=  np.nansum(pm_pop['lt_very_well'].iloc[inds])
        total5_10 +=  np.nansum(pm_pop['pop_5andup'].iloc[inds])
    if i > 30:
        ltvw_30 +=  np.nansum(pm_pop['lt_very_well'].iloc[inds])
        total5_30 +=  np.nansum(pm_pop['pop_5andup'].iloc[inds])

# format figure
for ai in range(4):
    ax[ai].set_xticks([0,10,20,30,40],['0','10','20','30','>40'],fontsize=12)
    ax[ai].set_yticks([0,20,40,60,80,100],['0','20','40','60','80','100'],fontsize=12)
    ax[ai].set_ylim((0,100))
    ax[ai].set_xlim((-1,41))
    ax[ai].spines[['right', 'top']].set_visible(False)

# add axes labels
ax[2].set_xlabel('Annual Alert Days in 2020',fontsize=12,fontweight='semibold')
ax[2].set_ylabel('Percent of Population',fontsize=12,fontweight='semibold')
ax[3].set_xlabel('Annual Alert Days in 2020',fontsize=12,fontweight='semibold')
ax[0].set_ylabel('Percent of Population',fontsize=14,fontweight='semibold')

# add titles
ax[0].set_title('(a) Race and Ethnicity',fontsize=14,fontweight='semibold')
ax[1].set_title('(b) Disadvantaged Status',fontsize=14,fontweight='semibold')
ax[2].set_title('(c) Educational Attainment',fontsize=14,fontweight='semibold')
ax[3].set_title('(d) English Language Proficiency',fontsize=14,fontweight='semibold')

# add text labels for bars
ax[0].text(0,75,'Non-Hispanic White',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='left')
ax[0].text(0,25,'Hispanic or Latino',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='left')
ax[0].spines[['right', 'top']].set_visible(False)
    
ax[1].text(40,85,'Not Disadvantaged',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='right')
ax[1].text(40,15,'Disadvantaged',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='right')

ax[2].text(0,92,'Graduate',fontsize=10,fontweight='semibold')
ax[2].text(0,78,'Bachelors',fontsize=10,fontweight='semibold')
ax[2].text(0,38,'High School',fontsize=10,fontweight='semibold')
ax[2].text(0,3,'< High School',fontsize=10,fontweight='semibold')

ax[3].text(40,85,'Speaks English\nLess Than Very Well',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='right')
ax[3].text(40,60,'Speaks English\nVery Well',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='right')
ax[3].text(40,20,'English Only',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='right')
# save figure
plt.savefig(fig_out_path+'final/language_ed_by_alert_days'+fig_desc+'.png',dpi=300)
fig.show()

# print values for discussion in paper
print('%hisp <10 alerts:',100*round(hisp_total_10/total_10,2))
print('%hisp >30 alerts:',100*round(hisp_total_30/total_30,2))
print('%increase',100*round(((hisp_total_30/total_30)-(hisp_total_10/total_10))/(hisp_total_10/total_10),3))
print('%white <10 alerts:',100*round(white_total_10/total_10,2))
print('%white >30 alerts:',100*round(white_total_30/total_30,2))
#print(np.mean(np.diff(disadvantaged_pct)))

print('%<HS <10 alerts:',100*round(lths_10/total25_10,2))
print('%<HS >30 alerts:',100*round(lths_30/total25_30,2))
print('%increase',100*round(((lths_30/total25_30)-(lths_10/total25_10))/(lths_10/total25_10),3))
print('%B increase',100*round(((b_30/total25_30)-(b_10/total25_10))/(b_10/total25_10),3))
print('%G increase',100*round(((g_30/total25_30)-(g_10/total25_10))/(g_10/total25_10),3))


print('%<"very well" <10 alerts:',100*round(ltvw_10/total5_10,2))
print('%<"very well" >30 alerts:',100*round(ltvw_30/total5_30,2))
print('%increase',100*round(((ltvw_30/total5_30)-(ltvw_10/total5_10))/(ltvw_10/total5_10),3))

# population ranges for figure caption
print('ed_pop range:',round(np.min(pop_sums_ed)/10**6,2),'-',round(np.max(pop_sums_ed)/10**6,1),'median:',
      round(np.median(pop_sums_ed)/10**6,1))
print('tot_pop range:',round(np.min(pop_sums_tot)/10**6,2),'-',round(np.max(pop_sums_tot)/10**6,1),'median:',
      round(np.median(pop_sums_tot)/10**6,1))
print('lng_pop range:',round(np.min(pop_sums_lang)/10**6,2),'-',round(np.max(pop_sums_lang)/10**6,1),'median:',
      round(np.median(pop_sums_lang)/10**6,1))
all_pop = np.concatenate([pop_sums_lang,pop_sums_tot,pop_sums_ed])
print('all pop median',round(np.median(pop_sums_ed)/10**6,1))

# version for TOC graphic
fig, axarr = plt.subplots(1,2,figsize=(9,4))
ax = axarr.flatten()
for i in range(len(bins)):
    if i < (len(bins)-1):
        lbin = bins[i]
        rbin = bins[i+1]
    elif i == (len(bins)-1):
        lbin = bins[i]
        rbin = pm_pop['abi_alerts'].max() + 2
    inds = np.where(np.logical_and(pm_pop['abi_alerts'].values>=lbin,pm_pop['abi_alerts'].values<rbin))
    # PANEL A - RACE and ETHNICIY
    tot_pop = np.nansum(pm_pop['Total'].iloc[inds])
    cat_total = 0
    for dem_str in dem_strs_other:
        dem_total = np.nansum(pm_pop[dem_str].iloc[inds])
        if dem_str == 'Not Hispanic or Latino: White alone':
            continue # want this one last
        ax[0].bar(bins[i],100.0*dem_total/tot_pop,color = dem_colors[0],bottom = cat_total,width=bar_width)    
        cat_total += 100.0*dem_total/tot_pop
    dem_total = np.nansum(pm_pop['Not Hispanic or Latino: White alone'].iloc[inds])
    ax[0].bar(bins[i],100.0*dem_total/tot_pop,color = '#e6ab02',bottom = cat_total,width=bar_width)

    # PANEL B - DISADVANTAGED STATUS
    inds = np.array(inds[0])
    inds_true = inds[np.where(pm_pop['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?'].iloc[inds]==True)[0]]
    inds_false = inds[np.where(pm_pop['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?'].iloc[inds]==False)[0]]
    inds_nan = inds[np.where(pd.isna(pm_pop['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?'].iloc[inds]))[0]]
    true_total = np.nansum(pm_pop['Total'].iloc[inds_true])
    false_total = np.nansum(pm_pop['Total'].iloc[inds_false])
    nan_total = np.nansum(pm_pop['Total'].iloc[inds_nan])
    ax[1].bar(bins[i],100.0*true_total/tot_pop,color = colors_cejst[0],width=bar_width)
    ax[1].bar(bins[i],100.0*false_total/tot_pop,color = colors_cejst[1],bottom=100.0*true_total/tot_pop,width=bar_width)
    ax[1].bar(bins[i],100.0*nan_total/tot_pop,color = 'black',
              bottom = 100.0*(true_total+false_total)/tot_pop,width=bar_width)
# format figure - addition labels added in powerpoint
ax[0].set_ylabel('Percent of Population',fontsize=12,fontweight='semibold')
ax[0].set_title('Race and Ethnicity',fontsize=14,fontweight='semibold')
ax[1].set_title('Disadvantaged Status',fontsize=14,fontweight='semibold')
# format figure
for ai in range(2):
    ax[ai].set_xticks([0,10,20,30,40],['','','','',''],fontsize=12)
    ax[ai].set_yticks([0,20,40,60,80,100],['0','20','40','60','80','100'],fontsize=12)
    ax[ai].set_ylim((0,100))
    ax[ai].set_xlim((-1,41))
    ax[ai].spines[['right', 'top']].set_visible(False)
# save figure, will futher edit in power point
plt.savefig(fig_out_path+'final/TOC_art_'+fig_desc+'.png',dpi=300)
fig.show()

#%% Figure 2 - decile plots by state
# pre-allocate arrays
states = []
dem_strs_st = []
st_mean_ud = [] # ud = upper decile
st_mean_ld = [] # ld = lower decile
st_alerts_ud = []
st_alerts_ld = []
nat_ud = []
nat_ld = []
# first calculate national and state level rankings using pandas groups and qcut
for dem_str in ['Hispanic or Latino','Not Hispanic or Latino: White alone']:
    # calc percent subgroup population for each census tract
    pct_dem = pm_pop[dem_str]/pm_pop['Total']
    # create a new dataframe with the variables we need
    dem_df = pd.DataFrame(data={'pct_dem':pct_dem,'alerts_geo':pm_pop['abi_alerts'],
                                'total':pm_pop['Total'],'mean_geo':pm_pop['abi_pm25_base'],
                                'state':pm_pop['State/Territory']})
    dem_df.reset_index(inplace=True,drop=True)
    # drop where pct_dem is nans, where total population is 0
    nan_inds = np.where(np.isnan(dem_df['pct_dem']))[0]
    dem_df.drop(nan_inds,inplace=True)
    dem_df.reset_index(inplace=True,drop=True)
    # now calculate deciles using pandas qcut
    dem_df.loc[:,'rank'] = pd.qcut(dem_df['pct_dem'],10,labels=False)
    # create pandas groups by rank
    groups = dem_df.groupby('rank')
    # pull and save values for upper and lower deciles
    nat_ld.append(groups['alerts_geo'].mean()[0])
    nat_ud.append(groups['alerts_geo'].mean()[9])
    
    # let's calculate a state-level version
    # pull state
    for state in ['Montana','Utah','Oregon','Wyoming','Colorado','Washington','Idaho','Nevada','California']:
        state_groups = dem_df.groupby('state')
        st_df = state_groups.get_group(state).copy()
        st_df.loc[:,'st_rank'] = pd.qcut(st_df['pct_dem'],10,labels=False)
        st_groups = st_df.groupby('st_rank')
        states.append(state)
        dem_strs_st.append(dem_str)
        st_mean_ud.append(st_groups.mean()['mean_geo'][9])
        st_mean_ld.append(st_groups.mean()['mean_geo'][0])
        st_alerts_ud.append(st_groups.mean()['alerts_geo'][9])
        st_alerts_ld.append(st_groups.mean()['alerts_geo'][0])
        del st_df

# now plot
i = 0
fig, ax = plt.subplots(1,3,figsize=(10,4))
n_states = int(len(states)/2)
for i in range(n_states):
    # hispanic or latino first
    ax[0].scatter(st_alerts_ud[i],states[i],color='teal')
    ax[0].scatter(st_alerts_ld[i],states[i],color='orange')
    # nh white
    # print(dem_strs_st[i+n_states]) # make sure we did this right
    ax[1].scatter(st_alerts_ud[i+n_states],states[i+n_states],color='orange')
    ax[1].scatter(st_alerts_ld[i+n_states],states[i+n_states],color='teal')
# add national values
ax[0].scatter(nat_ud[0],'National',color='teal')
ax[0].scatter(nat_ld[0],'National',color='orange')
ax[1].scatter(nat_ud[1],'National',color='orange')
ax[1].scatter(nat_ld[1],'National',color='teal')

# same as above, but using cejest data
pm_pop['cejst_plot'] = np.where(pm_pop['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?'],
                            1,0)
t_alerts = pm_pop['abi_alerts'].loc[pm_pop['cejst_plot']==1].mean()
nt_alerts = pm_pop['abi_alerts'].loc[pm_pop['cejst_plot']==0].mean()
# now by state
mean_talerts = []
mean_ntalerts = []
state_groups = pm_pop.groupby('State/Territory')
for state in ['Montana','Utah','Oregon','Wyoming','Colorado','Washington','Idaho','Nevada','California']:
    st_df = state_groups.get_group(state)
    mean_talerts = st_df['abi_alerts'].loc[st_df['cejst_plot']==1].mean()
    mean_ntalerts = st_df['abi_alerts'].loc[st_df['cejst_plot']==0].mean()
    if np.isnan(mean_talerts):
        print(state,'no disadvantaged tracts')
        ax[2].scatter(mean_ntalerts,state,color='orange')
        continue
    ax[2].scatter(mean_talerts,state,color='teal')
    ax[2].scatter(mean_ntalerts,state,color='orange')
ax[2].scatter(t_alerts,'National',color='teal')
ax[2].scatter(nt_alerts,'National',color='orange')

# figure formatting
for ai in range(3):
    ax[ai].plot([0,25],[8.5,8.5],'--',color='gray')
    ax[ai].set_xlim([0,25])
    ax[ai].spines[['right', 'top']].set_visible(False)
    ax[ai].set_xlabel('Number of alerts')

ax[0].set_title('(a) Ethnicity')
ax[1].set_title('(b) Race')
ax[2].set_title('(c) Disadvantage')
ax[1].set_yticks(np.arange(0,n_states+1),['']*(n_states+1))
ax[2].set_yticks(np.arange(0,n_states+1),['']*(n_states+1))

# add text labels
ax[0].text(24.5,1,'Most Hispanic',fontsize=10,fontweight='semibold',horizontalalignment='right',
           color='teal')
ax[0].text(24.5,0.5,'Least Hispanic',fontsize=10,fontweight='semibold',horizontalalignment='right',
           color='orange')
ax[1].text(24.5,1,'Most NH White',fontsize=10,fontweight='semibold',horizontalalignment='right',
           color='orange')
ax[1].text(24.5,0.5,'Least NH White',fontsize=10,fontweight='semibold',horizontalalignment='right',
           color='teal')
ax[2].text(24.5,1,'Not Disadvantaged',fontsize=10,fontweight='semibold',horizontalalignment='right',
           color='orange')
ax[2].text(24.5,0.5,'Disadvantaged',fontsize=10,fontweight='semibold',horizontalalignment='right',
           color='teal')
# save figure
plt.tight_layout()
plt.savefig(fig_out_path + 'final/race_cejst_bystate_'+fig_desc+'.png',dpi=300)

# print numbers for figure discussion
print('difference alerts most hisp - least hisp:',round(nat_ud[1]-nat_ld[1],1))
print(states[8],dem_strs_st[8],' difference alerts most hisp - least hisp:',round(st_alerts_ud[8]-st_alerts_ld[8],1))

#%% Figure 3 - population weighted alerts and mean PM for geo and leo
fig,axarr = plt.subplots(1,3,figsize=(11,5))
ax = axarr.flatten()
# create a nan flag to remove chunk of montana missing
pm_pop['nan_flag']= np.where(pd.isna(pm_pop['abi_alerts']),0,1)
total_abi_alerts =  np.nansum(pm_pop['abi_alerts']*pm_pop['Total'])/np.nansum(pm_pop['Total']*pm_pop['nan_flag'])
total_abi1pm_alerts = np.nansum(pm_pop['abi1pm_alerts']*pm_pop['Total'])/np.nansum(pm_pop['Total']*pm_pop['nan_flag'])
total_abi_conc = np.nansum(pm_pop['abi_pm25_base']*pm_pop['Total'])/np.nansum(pm_pop['Total']*pm_pop['nan_flag'])
total_abi1pm_conc = np.nansum(pm_pop['abi1pm_pm25_base']*pm_pop['Total'])/np.nansum(pm_pop['Total']*pm_pop['nan_flag'])

# plotting data
i = 0
ci=0
for dem_str in dem_strs:
    # panel a: total population weighted alerts
    abi_alerts =  np.nansum(pm_pop['abi_alerts']*pm_pop[dem_str])/np.nansum(pm_pop[dem_str]*pm_pop['nan_flag'])
    abi1pm_alerts =  np.nansum(pm_pop['abi1pm_alerts']*pm_pop[dem_str])/np.nansum(pm_pop[dem_str]*pm_pop['nan_flag'])
    ax[0].barh(i,abi1pm_alerts,color=bar_colors[ci],hatch='//',edgecolor='white',height=0.5)
    ax[0].barh(i+0.5,abi_alerts,color=bar_colors[ci],edgecolor='white', height=0.5)
    # print numbers for figure discussion in manuscript
    print(dem_str,'\ntotal popw alerts:',round(abi_alerts,1))

    if i >= 2: # to allin the bars across each panel
        # panel b: disparity ratio for annual alerts
        ax[1].barh(i+0.5,(abi_alerts/total_abi_alerts)-1,color=bar_colors[ci],edgecolor='white',height=0.5)
        ax[1].barh(i,(abi1pm_alerts/total_abi1pm_alerts)-1,color=bar_colors[ci],hatch='//',edgecolor='white',height=0.5)
        # panel c: disparity ratio for annual concentration
        abi_pm =  np.nansum(pm_pop['abi_pm25_base']*pm_pop[dem_str])/np.nansum(pm_pop[dem_str]*pm_pop['nan_flag'])
        abi1pm_pm =  np.nansum(pm_pop['abi1pm_pm25_base']*pm_pop[dem_str])/np.nansum(pm_pop[dem_str]*pm_pop['nan_flag'])
        ax[2].barh(i+0.5,abi_pm/total_abi_conc-1,color=bar_colors[ci],edgecolor='white',height=0.5,)
        ax[2].barh(i,abi1pm_pm/total_abi1pm_conc-1,color=bar_colors[ci],hatch='//',edgecolor='white',height=0.5)
        # print numbers for figure discussion in manuscript
        print('alert ratio:',round(abi_alerts/total_abi_alerts,2),'conc ratio:',round(abi_pm/total_abi_conc,2))
        print('%difference:',100*round(((abi_alerts/total_abi_alerts)-(abi_pm/total_abi_conc))/np.mean([(abi_alerts/total_abi_alerts),(abi_pm/total_abi_conc)]),2))
    i += 2
    ci += 1

# for legend
ax[2].barh(1,0,color=dem_colors[0],edgecolor='white',height=0.5,label='ABI-daytime')
ax[2].barh(1,0,color=dem_colors[0],hatch='//',edgecolor='white',height=0.5,label='ABI-1pm')

# formatting panels
for ai in range(1,3):
    ax[ai].set_ylim(-0.5,11)
    ax[ai].axvline(0,color='black')
    ax[ai].spines[["top","right"]].set_visible(False)
    ax[ai].set_yticks([2,4,6,8,10],['']*5,fontsize=12)

ax[0].set_xlabel('Mean Annual Population\nWeighted Alerts in 2020',fontsize=12)
ax[0].set_yticks([0,2,4,6,8,10],['Total','NH Black','Hispanic or Latino',
                                 'NH White','NH Native','NH Asian and \nPacific Islander'],fontsize=12)
ax[0].set_ylim(-0.5,11)
ax[0].spines[["top","right"]].set_visible(False)
ax[0].set_title('(a)')

ax[1].set_xlabel('Subgroup pop-mean annual alerts\nTotal pop-mean annual alerts',fontsize=12)
ax[1].set_xticks([-0.5,0,0.5,1],[0.5,1,1.5,2]) # adjust legend to base off of 1 for the ratio
ax[1].set_title('(b)')

ax[2].set_xlabel('Subgroup pop-mean annual PM$_{2.5}$\nTotal pop-mean annual PM$_{2.5}$',fontsize=12)
ax[2].set_xticks([-0.1,0,0.1,0.2],[-1.1,1,1.1,1.2]) # adjust legend to base off of 1 for the ratio
ax[2].legend(loc='best',frameon=False,fontsize=12)
ax[2].set_title('(c)')

# save figure
plt.tight_layout()
plt.savefig(fig_out_path + 'final/national_nalerts_bypop_'+fig_desc+'_comb.png',dpi=300)

# print numbers for figure disucssion
# done earlier in the for-loop

#%% Figure S1 - percentiles version of Fig 1
fig, axarr = plt.subplots(2,2,figsize=(9,8))
ax = axarr.flatten()
# define bins
pbins = np.arange(0,105,5)
# pull tracts with at least one alert day
inds_use = np.where(pm_pop['abi_alerts']>=1.0)
pm_pop_pct_plt = pm_pop.iloc[inds_use].copy()
pm_pop_pct_plt.reset_index(inplace=True,drop=True)
bar_width = 4
# first few bins are empty, count these for plotting purposes
empty_count = 0
# loop through and plot!
for i in range(len(pbins)-1):
    lbin = np.nanpercentile(pm_pop_pct_plt['abi_alerts'].values,pbins[i])
    rbin = np.nanpercentile(pm_pop_pct_plt['abi_alerts'].values,pbins[i+1])
    inds = np.where(np.logical_and(pm_pop_pct_plt['abi_alerts'].values>=lbin,
                                   pm_pop_pct_plt['abi_alerts'].values<rbin))
    inds = np.array(inds[0])
    tot_pop = pm_pop_pct_plt['Total'].iloc[inds]
    pop_sum = np.nansum(tot_pop)
    # check for data in bins - first few percentiles don't have any, 
    # many grid cells with exactly 1 alert day
    if pop_sum==0:
        empty_count +=1
        continue
    
    # PANEL A - race and ethnicity
    ci = 0
    cat_total = 0
    for dem_str in dem_strs_other:
        dem_total = np.nansum(pm_pop_pct_plt[dem_str].iloc[inds])
        if i == empty_count:
            ax[0].bar(pbins[empty_count+1]/2,100.0*dem_total/pop_sum,color = dem_colors[ci],bottom = cat_total,
                      width=4+pbins[empty_count])
        else:
            ax[0].bar(pbins[i]+2.5,100.0*dem_total/pop_sum,color = dem_colors[ci],bottom = cat_total,width=bar_width)
        cat_total += 100.0*dem_total/pop_sum
        ci += 1
    
    # PANEL B - Disadvantaged status
    inds_true = inds[np.where(pm_pop_pct_plt['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?'].iloc[inds]==True)[0]]
    inds_false = inds[np.where(pm_pop_pct_plt['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?'].iloc[inds]==False)[0]]
    inds_nan = inds[np.where(pd.isna(pm_pop_pct_plt['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?'].iloc[inds]))[0]]
    true_total = np.nansum(pm_pop_pct_plt['Total'].iloc[inds_true])
    false_total = np.nansum(pm_pop_pct_plt['Total'].iloc[inds_false])
    nan_total = np.nansum(pm_pop_pct_plt['Total'].iloc[inds_nan])
    if i ==empty_count:
        ax[1].bar(pbins[empty_count+1]/2,100.0*true_total/pop_sum,color = colors_cejst[0],width=4+pbins[empty_count])
        ax[1].bar(pbins[empty_count+1]/2,100.0*false_total/pop_sum,color = colors_cejst[1],bottom=100.0*true_total/pop_sum,width=4+pbins[empty_count])
        ax[1].bar(pbins[empty_count+1]/2,100.0*nan_total/pop_sum,color = 'black',bottom = 100.0*(true_total+false_total)/pop_sum,width=4+pbins[empty_count])
    else:
        ax[1].bar(pbins[i]+2.5,100.0*true_total/pop_sum,color = colors_cejst[0],width=bar_width)
        ax[1].bar(pbins[i]+2.5,100.0*false_total/pop_sum,color = colors_cejst[1],bottom=100.0*true_total/pop_sum,width=bar_width)
        ax[1].bar(pbins[i]+2.5,100.0*nan_total/pop_sum,color = 'black',bottom = 100.0*(true_total+false_total)/pop_sum,width=bar_width)
   
    # PANEL C - Educational Attainment
    tot_pop = pm_pop_pct_plt['pop_25andup'].iloc[inds]
    pop_sum = np.nansum(tot_pop)
    cat_total = 0
    vi = 0
    for var in ['lt_hs','hs','bachelors','grad']:
        var_total = np.nansum(pm_pop_pct_plt[var].iloc[inds])
        var_total_plot = 100.0*var_total/pop_sum
        if i == empty_count:
            ax[2].bar(pbins[empty_count+1]/2,var_total_plot,bottom = cat_total, color = colors_ed[vi],width=4+pbins[empty_count])   
        else:
            ax[2].bar(pbins[i]+2.5,var_total_plot,bottom = cat_total, color = colors_ed[vi],width=bar_width)   
        cat_total += var_total_plot
        vi+= 1
        
    # PANEL D - English Langauge Proficiency
    tot_pop = pm_pop_pct_plt['pop_5andup'].iloc[inds]
    eng_only_total = np.nansum(pm_pop_pct_plt['eng_only'].iloc[inds])
    vwell_total = np.nansum(pm_pop_pct_plt['very_well'].iloc[inds])
    lt_vwell_total = np.nansum(pm_pop_pct_plt['lt_very_well'].iloc[inds])
    pop_sum = np.nansum(tot_pop)
    if i == empty_count:
        ax[3].bar(pbins[empty_count+1]/2,100.0*eng_only_total/pop_sum,color = colors_lng[0],label = 'English Only',width=4+pbins[empty_count])
        ax[3].bar(pbins[empty_count+1]/2,100.0*vwell_total/pop_sum, bottom = 100.0*eng_only_total/pop_sum,
               color=colors_lng[1],label='Very Well',width=4+pbins[empty_count])
        ax[3].bar(pbins[empty_count+1]/2,100.0*lt_vwell_total/pop_sum, bottom = 100.0*eng_only_total/pop_sum + 100.0*vwell_total/pop_sum,
               color=colors_lng[2],label = 'Less Than Very Well',width=4+pbins[empty_count])
    else:
        ax[3].bar(pbins[i]+2.5,100.0*eng_only_total/pop_sum,color = colors_lng[0],label = 'English Only',width=bar_width)
        ax[3].bar(pbins[i]+2.5,100.0*vwell_total/pop_sum, bottom = 100.0*eng_only_total/pop_sum,
               color=colors_lng[1],label='Very Well',width=bar_width)
        ax[3].bar(pbins[i]+2.5,100.0*lt_vwell_total/pop_sum, bottom = 100.0*eng_only_total/pop_sum + 100.0*vwell_total/pop_sum,
               color=colors_lng[2],label = 'Less Than Very Well',width=bar_width)

# format panels
titles = ['(a) Race and Ethnicity','(b) Disadvantaged Status',
          '(c) Educational Attainment','(d) English Language Proficiency']
for i in range(4):
    ax[i].set_yticks([0,20,40,60,80,100],['0','20','40','60','80','100'],fontsize=12)
    ax[i].set_xticks([0,20,40,60,80,100],['0','20','40','60','80','100'],fontsize=12)
    ax[i].set_ylim((0,100))
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].set_title(titles[i],fontsize=14,fontweight='semibold')

ax[2].set_xlabel('Annual Alert Day Percentile',fontsize=12,fontweight='semibold')
ax[2].set_ylabel('Percent of Population',fontsize=12,fontweight='semibold')
ax[3].set_xlabel('Annual Alert Day Percentile',fontsize=12,fontweight='semibold')
ax[0].set_ylabel('Percent of Population',fontsize=14,fontweight='semibold')

# add text labels
ax[0].text(0,75,'Non-Hispanic White',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='left')
ax[0].text(0,25,'Hispanic or Latino',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='left')

ax[1].text(100,75,'Not Disadvantaged',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='right')
ax[1].text(100,3,'Disadvantaged',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='right')

ax[2].text(0,92,'Graduate',fontsize=10,fontweight='semibold')
ax[2].text(0,78,'Bachelors',fontsize=10,fontweight='semibold')
ax[2].text(0,38,'High School',fontsize=10,fontweight='semibold')
ax[2].text(0,3,'< High School',fontsize=10,fontweight='semibold')

ax[3].text(100,89,'Speaks English\nLess Than Very Well',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='right')
ax[3].text(100,67,'Speaks English\nVery Well',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='right')
ax[3].text(100,20,'English Only',fontsize=10,fontweight='semibold',color = 'black',
        horizontalalignment='right')

plt.savefig(fig_out_path+'final/dem_by_alert_days_percentile_'+fig_desc+'.png',dpi=300)
fig.show()


#%% Figure S2 pie-charts of person alerts and percent difference in alerts
fig, axarr = plt.subplots(2,2,figsize=(10,8))
ax = axarr.flatten()
diffs = []
pct_diffs = []
total_dem_pop = []
total_pop_alerts_dem = []
for dem_str in np.concatenate([['Total'],dem_strs_other]):
    geo_alerts = np.nansum(pm_pop[dem_str]*pm_pop['abi_alerts'])
    leo_alerts = np.nansum(pm_pop[dem_str]*pm_pop['abi1pm_alerts'])
    diff = geo_alerts - leo_alerts
    pct_diff = diff/np.mean([geo_alerts,leo_alerts])
    diffs.append(diff)
    pct_diffs.append(pct_diff)
    # for supplemental figure in next section
    total_dem_pop.append(np.nansum(pm_pop[dem_str]))
    total_pop_alerts_dem.append(geo_alerts)
diffs = np.array(diffs)
pct_diffs = np.array(pct_diffs)
colors = np.concatenate([['black'],dem_colors])
# plot data
ax[0].pie(diffs[1:],colors = dem_colors,autopct='%1.1f%%', pctdistance=1.3,startangle=180)
ax[1].bar(np.arange(0,len(pct_diffs)),height = 100.0*pct_diffs,color = colors)
# format plot
ax[0].set_title('(a) Additional person-alerts\nby race and ethnicity',fontsize=14)
ax[1].set_xticks([0,1,2,3,4,5,6],
                 ['All','Other','NH Black','Hispanic or\nLatino','NH White','NH Native','NH Asian or\nPacific Islander'],
                 rotation=45,fontsize=12)
ax[1].set_ylabel('Percent\nDifference',fontsize=12)
ax[1].set_title('(b) Difference in person alerts\ngeostationary vs. polar-orbiting',fontsize=14)
ax[1].spines[['right', 'top']].set_visible(False)

# now by CEJST groups
diffs = []
total_pop_alerts_cejst = []
pct_diffs = []
for flag in [True,False]:
    inds = np.where(pm_pop['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?']==flag)[0]
    geo_alerts = np.nansum(pm_pop['Total'].iloc[inds].values*pm_pop['abi_alerts'].iloc[inds].values)
    leo_alerts = np.nansum(pm_pop['Total'].iloc[inds].values*pm_pop['abi1pm_alerts'].iloc[inds].values)
    diff = geo_alerts - leo_alerts
    pct_diff = diff/np.mean([geo_alerts,leo_alerts])
    diffs.append(diff)
    pct_diffs.append(pct_diff)
    # save values for figure is next section
    total_pop_alerts_cejst.append(geo_alerts)
diffs = np.array(diffs)
pct_diffs = np.array(pct_diffs)
# plot data
ax[2].pie(diffs,colors = colors_cejst,autopct='%1.1f%%', pctdistance=1.3,startangle=180)
ax[3].bar(np.arange(0,len(pct_diffs)),height = 100.0*pct_diffs,color = colors_cejst)
# format plot
ax[2].set_title('(c) Additional person-alerts\nby disadvantaged status',fontsize=14)
ax[3].set_xticks([0,1],['Disadvantaged ',' Not Disadvantaged'],fontsize=12)
ax[3].set_ylabel('Percent\nDifference',fontsize=12)
ax[3].set_title('(d) Difference in person-alerts\nby disadvantaged status',fontsize=14)
ax[3].spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.savefig(fig_out_path+'final/pie_charts_alert_diffs_'+fig_desc+'.png',dpi=300)

#%% Figure S3 - pie charts for the total population and total person-alerts
tinds = np.where(pm_pop['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?']==True)[0]
ntinds = np.where(pm_pop['Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?']==False)[0]
tpop = pm_pop['Total'].iloc[tinds].sum()
ntpop = pm_pop['Total'].iloc[ntinds].sum()

fig,ax = plt.subplots(2,2,figsize=(6,6))
ax = ax.flatten()
ax[0].pie([tpop,ntpop],colors=colors_cejst,autopct='%1.1f%%', pctdistance=1.3,startangle=180)
ax[0].set_title('(a) Total population\nby disadvantaged status',fontsize=14)
ax[1].pie(total_dem_pop[1:],colors=dem_colors,autopct='%1.1f%%', pctdistance=1.3,startangle=180)
ax[1].set_title('(b) Total population\nby race/ethnicity',fontsize=14)
ax[2].pie(total_pop_alerts_cejst, colors = colors_cejst,autopct='%1.1f%%', pctdistance=1.3,startangle=180)
ax[2].set_title('(c) Total person-alerts\nby disadvantaged status',fontsize=14)
ax[3].pie(total_pop_alerts_dem[1:], colors = dem_colors,autopct='%1.1f%%', pctdistance=1.3,startangle=180)
ax[3].set_title('(d) Total person-alerts\nby race/ethnicity',fontsize=14)
plt.tight_layout()
plt.savefig(fig_out_path+'final/pie_charts_total_pop_'+fig_desc+'.png',dpi=300)


#%% Create geopandas array for map plotting
# merge part of pm_pop array with geopandas array for plotting
# get CRS from cejst shapefile because I like the way it looks for plotting
# probably a better way to do this, but this is quick and easy
crs_use_df = gpd.read_file(cejst_shp_fn)
df1 = gpd.read_file(shp_fp)
df1=df1.to_crs(crs_use_df.crs)
df1['geoid'] = df1['GEOID'].astype(int)
pm_plot = df1.merge(pm_pop[['geoid','Not Hispanic or Latino: Other','Not Hispanic or Latino: Black or African American alone',
                'Hispanic or Latino','Not Hispanic or Latino: White alone',
                'Not Hispanic or Latino: American Indian and Alaska Native alone',
                'Not Hispanic or Latino: Asian or Pacific Islander','Total','abi_pm25_base',
                'abi_alerts','abi1pm_alerts','pop_25andup','pop_5andup',
                'lt_hs','hs','bachelors','grad','SF',
                'eng_only','very_well','lt_very_well']],
                    how='left',left_on='geoid',right_on='geoid')

# remove areas outside contig US from new geopandas array
states_rmv = ['78','02','15','66','69','60','72']
for st_str in states_rmv:
    pm_plot = pm_plot[pm_plot.STATEFP!=st_str]
    pm_plot.reset_index(inplace=True,drop=True)

#%% Figure S4 - pm alerts and concentration by census tract
fig,ax=plt.subplots(2,2,figsize=(6,4))
ax=ax.flatten()
pm_plot.plot('abi_pm25_base',ax=ax[0],cmap='OrRd',vmin=0,vmax=15,legend=True,
             legend_kwds={"orientation": "horizontal",'label':'PM2.5 [ug/m3]','shrink':0.7,
                          'extend':'max'})
ax[0].set_title('2020 Annual Mean PM$_{2.5}$')
pm_plot.plot('abi_alerts',ax=ax[1],cmap='viridis',vmin=0,vmax=20,legend=True,
             legend_kwds={"orientation": "horizontal",'label':'N Alert Days','shrink':0.7,
                          'extend':'max'})
ax[1].set_title('2020 PM$_{2.5}$ Alert Count')
# use geopandas quantiles scheme to plot quartiles
# https://geopandas.org/en/stable/gallery/choropleths.html#Classification-by-quantiles
pm_plot.plot('abi_pm25_base',ax=ax[2],cmap='OrRd',legend=True,edgecolor=None,
             scheme='quantiles',k=4,legend_kwds={"bbox_to_anchor": (0.4, 0.3),
                            "interval": True, # show intervals for quartiles in legend
                          'markerscale':0.3, 'fontsize':'x-small',
                          'frameon':False,'title':'Quartiles'})
pm_plot.plot('abi_alerts',ax=ax[3],cmap='viridis',legend=True,
             scheme='quantiles',k=4,legend_kwds={"bbox_to_anchor": (0.4, 0.3),"interval": True,
                          'markerscale':0.3, 'fontsize':'x-small','frameon':False,
                          'title':'Quartiles'})
# format plot
for ai in range(4):
    ax[ai].set_axis_off()
# adjust to make the maps closer in size
ax[0].set_xlim(-125, -66)
ax[0].set_ylim(25, 50)
ax[1].set_xlim(-125, -66)
ax[1].set_ylim(25, 50)
 
plt.tight_layout()
# save figure
plt.savefig(fig_out_path+'final/pm_maps_'+fig_desc+'.png',dpi=400)

#%% Figure S5 map of race/ethnicity by census tract
fig,ax = plt.subplots(3,2,figsize=(8,9))
ax = ax.flatten()
titles = ['(a) NH Black','(b) Hispanic or Latino','(c) NH White',
          '(d) NH Native','(e) NH Asian or Pacific Islander']
i = 0
for dem_str in dem_strs[1:]:
    # create variable name to add to pct population to dataframe
    str_plot = dem_str + ' pct'
    pm_plot[str_plot] = 100.0*pm_plot[dem_str]/pm_plot['Total']
    pm_plot.plot(str_plot,ax = ax[i],cmap='magma',legend=True,k=4,
                 scheme='quantiles',legend_kwds={"bbox_to_anchor": (0.4, 0.3),
                              'markerscale':0.3, 'fontsize':'x-small',
                              'frameon':False,'title':'Quartiles'})
    ax[i].set_title(titles[i])
    ax[i].set_axis_off()
    i +=1
# this figure has an empty panel, 
# remove the axis from the panel so it is only white space
ax[i].set_axis_off()
plt.tight_layout()
# save figure
plt.savefig(fig_out_path + 'final/dem_maps_'+fig_desc+'.png',dpi=300)
 
#%% Figure S6 - educational attainment by census tract
fig,ax = plt.subplots(2,2,figsize=(8,6))
ax = ax.flatten()
titles = ['(a) < High School','(b) High School','(c) Bachelors','(d) Graduate']
i = 0
for ed_str in ['lt_hs','hs','bachelors','grad']:
    str_plot = ed_str + '_pct'
    pm_plot[str_plot] = 100.0*pm_plot[ed_str]/pm_plot['pop_25andup']
    pm_plot.plot(str_plot,ax = ax[i],cmap='magma',legend=True,
                 scheme='quantiles',k=4,legend_kwds={"bbox_to_anchor": (0.4, 0.3),
                              'markerscale':0.3, 'fontsize':'small','frameon':False,'title':'Quartiles'})
    ax[i].set_title(titles[i])
    ax[i].set_axis_off()
    i +=1
plt.tight_layout()
plt.savefig(fig_out_path + 'final/education_maps_'+fig_desc+'.png',dpi=300)

#%% Figure S7 - english language proficiency by census tract
fig,ax = plt.subplots(1,3,figsize=(11,7))
ax = ax.flatten()
titles = ['(a) English only','(b) "very well"','(c) less than "very well"']
i = 0
for lng_str in ['eng_only','very_well','lt_very_well']:
    str_plot = lng_str + '_pct'
    pm_plot[str_plot] = 100.0*pm_plot[lng_str]/pm_plot['pop_5andup']
    pm_plot.plot(str_plot,ax = ax[i],cmap='magma',legend=True,
                 scheme='quantiles',k=4,legend_kwds={"bbox_to_anchor": (0.4, 0.3),
                              'markerscale':0.3, 'fontsize':'x-small','frameon':False,'title':'Quartiles'})
    ax[i].set_title(titles[i])
    ax[i].set_axis_off()
    i +=1
plt.tight_layout()
plt.savefig(fig_out_path + 'final/lng_maps_'+fig_desc+'.png',dpi=300)

#%% figure S8 - Gaiges data
fig,ax=plt.subplots(1,1,figsize=(6,6))
ax.scatter(gaige_data['NIFC wildland fire acres burned'],
           gaige_data['ratio-hispanic'],color='#d95f02',label = 'Hispanic or Latino')
ax.scatter(gaige_data['NIFC wildland fire acres burned'],
           gaige_data['ratio-black'],color='#1b9e77',label = 'Black')
# plot 2020 as a star
ax.scatter(gaige_data['NIFC wildland fire acres burned'].iloc[[-1]],
           gaige_data['ratio-hispanic'].iloc[[-1]],color='#d95f02',marker='*',s=250)
ax.scatter(gaige_data['NIFC wildland fire acres burned'].iloc[[-1]],
           gaige_data['ratio-black'].iloc[[-1]],color='#1b9e77',marker='*',s=250)
# format figure
ax.set_xticks(np.array([4,5,6,7,8,9,10])*10**6,['4','5','6','7','8','9','10'],fontsize=12)
ax.set_yticks([0.95,1.0,1.05,1.10],['0.95','1.00','1.05','1.10'],fontsize=12)
ax.set_xlabel('Wildland Fire Area Burned [Millions of Acres]',fontsize=14)
ax.set_ylabel('Annual Mean PM$_{2.5}$ Disparity Ratio',fontsize=14)
ax.spines[['right', 'top']].set_visible(False)
ax.text(3.5*10**6,0.97,'Hispanic or Latino',color='#d95f02',fontsize=12,fontweight='bold')
ax.text(3.5*10**6,0.96,'Black',color='#1b9e77',fontsize=12,fontweight='bold')
plt.savefig(fig_out_path+'/final/Gaige_analysis_'+fig_desc+'.png',dpi=300)
fig.show()

#%% #### BELOW FIGURES NOT IN MANUSCRIPT #####
# But are used for checking analysis 
'''
#%% map of % difference in alerts for each census tract - not in paper but for checking why all are about a 40% diff in person alerts
mean = np.nanmean([pm_plot['abi_alerts'].values*pm_plot['Total'].values,pm_plot['Total'].values*pm_plot['abi1pm_alerts'].values],axis=0)
pm_plot['pct_diff'] = pm_plot['Total'].values*(pm_plot['abi_alerts'].values - pm_plot['abi1pm_alerts'].values)/mean
pm_plot.plot('pct_diff',legend=True)

#%% combine shapefile with datasets for plotting by states - not in paper but for checking state-level trends
# check census tract averaging against gridded version
fip_plot = 32
name = 'Nevada'
shp_fp = prj_folder + '/population_data/census_tract_shapefiles_19/tl_2019_'+str(fip_plot).zfill(2)+'_tract/tl_2019_'+str(fip_plot).zfill(2)+'_tract.shp'
df1 = gpd.read_file(shp_fp)
df1.to_crs("EPSG:4326")
df1['geoid'] = df1['GEOID'].astype(int)

df = df1.merge(pm_pop,left_on='geoid',right_on='geoid')
var_descs = ['pop_white','pop_hisp','pop_black','pop_native','pop_asian']
vi = 0
for var in [ 'Not Hispanic or Latino: White alone', 'Hispanic or Latino',
            'Not Hispanic or Latino: Black or African American alone',
            'Not Hispanic or Latino: American Indian and Alaska Native alone', 
            'Not Hispanic or Latino: Asian or Pacific Islander']:
    pop_var = df[var].values
    df[var_descs[vi]] = pop_var
    pop_fraction = pop_var/df['Total'].values
    new_var_f = var_descs[vi] + '_f'
    df[new_var_f[vi]] = pop_fraction
    
    frank = df[new_var_f[vi]].rank(pct=True)
    new_var_r = var_descs[vi] + '_r'
    df[new_var_r] = frank
    vi += 1
    
f,ax = plt.subplots(2,2, figsize=(8,4), sharex=True, sharey=True, dpi=300)
ax = ax.flatten()
f.suptitle(name+' Alerts Check')
ax[0].title.set_text('(a) Alerts')
ax[1].title.set_text('(b) Black')
ax[2].title.set_text('(c) Hispanic')
ax[3].title.set_text('(d) White')
vmaxs = [5,1,1,1]
ai = 0
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
for str_plot in ['abi_alerts','pop_black_r','pop_hisp_r','pop_white_r']:
    divider = make_axes_locatable(ax[ai])
    cax = divider.append_axes("right", size="3%",pad=0,alpha=0.5)
    df.plot(str_plot, ax=ax[ai], alpha=1, cmap='OrRd',edgecolor='white', legend=True, 
        cax=cax, linewidth=0.1,vmax=vmaxs[ai])
    ai += 1
plt.show()


# make state-specific alerts/person
# alert days by demographics
# let's do delta's from the total
i = 0
ci=0
fig,ax = plt.subplots(1,1,figsize=(6,4))
for dem_str in ['Total','pop_black','pop_hisp','pop_white','pop_native','pop_asian']:
    abi_alerts =  np.nansum(df['abi_alerts']*df[dem_str])/np.nansum(df[dem_str])
    ax.barh(i,abi_alerts,color=colors[ci],edgecolor='white',height=0.5)

    i += 2
    ci += 1
ax.set_xlabel('mean annual alerts',fontsize=14)
ax.set_yticks([0.5,2.5,4.5,6.5,8.5,10.5],['All','Black','Hispanic or Latino',
                                 'White\n(not Hispanic or Latino)','Native','Asian'],fontsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(fig_out_path + 'preliminary/state_level_figures/'+name +'_alerts_bypop_'+fig_desc+'.png',dpi=300)
fig.show()

# ratio version
i = 0
ci=1
fig,ax = plt.subplots(1,1,figsize=(6,4))
tot_abi = np.nansum(df['abi_alerts']*df['Total'])/np.nansum(df['Total'])

for dem_str in ['pop_black','pop_hisp','pop_white','pop_native','pop_asian']:
    abi_alerts =  np.nansum(df['abi_alerts']*df[dem_str])/np.nansum(df[dem_str])
    ax.barh(i+0.5,abi_alerts/tot_abi-1,color=colors[ci],edgecolor='white',height=0.5)
    i += 2
    ci += 1
ax.set_xlabel('mean annual alerts, ratio to total',fontsize=14)
ax.set_yticks([0.5,2.5,4.5,6.5,8.5],['Black','Hispanic or Latino',
                                 'White\n(not Hispanic or Latino)','Native','Asian'],fontsize=12)
ax.axvline(0,color='black')
ax.set_xticks([-0.25,0,0.25,0.5],[0.25,1,1.25,1.5])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(fig_out_path + 'preliminary/state_level_figures/'+name +'_alerts_ratio_bypop_'+fig_desc+'.png',dpi=300)
fig.show()

# education, state version
fig, ax = plt.subplots(1,1,figsize=(5,5))
bins = np.arange(0,16)
pbins = np.arange(0,100)
colors_ed = ['#bebada','#fdb462','#fccde5','#7fc97f']
for i in range(len(bins)):
    if i < (len(bins)-1):
        lbin = bins[i]
        rbin = bins[i+1]
    elif i == (len(bins)-1):
        lbin = bins[i]
        rbin = df['abi_alerts'].max()
    #lbin = np.nanpercentile(pm_pop['abi_alerts'].values,pbins[i])
    #rbin = np.nanpercentile(pm_pop['abi_alerts'].values,pbins[i+1])

    inds = np.where(np.logical_and(df['abi_alerts'].values>lbin,
                                   df['abi_alerts'].values<=rbin))
    tot_pop = df['pop_25andup'].iloc[inds]
    pop_sum = np.nansum(tot_pop)
    print(pop_sum)
    cat_total = 0
    vi = 0
    for var in ['lt_hs','hs','bachelors','grad']:
        var_total = np.nansum(df[var].iloc[inds])
        var_total_plot = 100.0*var_total/pop_sum
        ax.bar(bins[i],var_total_plot,bottom = cat_total, color = colors_ed[vi])
        cat_total += var_total_plot
        vi+= 1
ax.set_xlabel('Annual Alert Days')
ax.set_xticks([0,3,6,9,12,15],['0','3','6','9','12','>15'])
ax.set_ylabel('Percent of Population')
plt.savefig(fig_out_path+'preliminary/state_level_figures/'+'pop_education_by_alert_days_'+name+'.png',dpi=300)
fig.show()

# english speaking breakdown, state version
fig, ax = plt.subplots(1,1,figsize=(5,5))
bins = np.arange(0,16)
pbins = np.arange(0,100)
for i in range(len(bins)):
    if i < (len(bins)-1):
        lbin = bins[i]
        rbin = bins[i+1]
    elif i == (len(bins)-1):
        lbin = bins[i]
        rbin = df['abi_alerts'].max()
    #lbin = np.nanpercentile(pm_pop['abi_alerts'].values,pbins[i])
    #rbin = np.nanpercentile(pm_pop['abi_alerts'].values,pbins[i+1])

    inds = np.where(np.logical_and(df['abi_alerts'].values>lbin,
                                   df['abi_alerts'].values<=rbin))
    tot_pop = df['pop_5andup'].iloc[inds]
    eng_only_total = np.nansum(df['eng_only'].iloc[inds])
    vwell_total = np.nansum(df['very_well'].iloc[inds])
    lt_vwell_total = np.nansum(df['lt_very_well'].iloc[inds])
    pop_sum = np.nansum(tot_pop)
    print(pop_sum)
    ax.bar(bins[i],100.0*eng_only_total/pop_sum,color = '#8dd3c7',label = 'English Only')
    ax.bar(bins[i],100.0*vwell_total/pop_sum, bottom = 100.0*eng_only_total/pop_sum,
           color='#fb9a99',label='Very Well')
    ax.bar(bins[i],100.0*lt_vwell_total/pop_sum, bottom = 100.0*eng_only_total/pop_sum + 100.0*vwell_total/pop_sum,
           color='#80b1d3',label = 'Less Than Very Well')
#ax.legend()
ax.set_xticks([0,3,6,9,12,15],['0','3','6','9','12','>15'])
ax.set_xlabel('Annual Alert Days')
ax.set_ylabel('Percent of Population')
plt.savefig(fig_out_path+'preliminary/state_level_figures/'+'pop_language_by_alert_days'+name+'.png',dpi=300)
fig.show()
'''
