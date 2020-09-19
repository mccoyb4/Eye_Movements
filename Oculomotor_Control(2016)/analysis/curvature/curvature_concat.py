from __future__ import division
import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from IPython import embed as shell


#Subjects
subjects = ['LP01','LP02','LP03', 'LP04','LP05', 'LP06','LP07','LP08','LP09','LP10','LP11','LP12','LP13','LP14','LP15','LP16','LP17','LP18','LP19','LP20','LP21', 'LP22','LP23','LP24'] 
subject_files = ['LP01.csv','LP02.csv','LP03.csv','LP04.csv','LP05.csv','LP06.csv','LP07.csv','LP08.csv','LP09.csv','LP10.csv','LP11.csv','LP12.csv','LP13.csv','LP14.csv','LP15.csv','LP16.csv','LP17.csv','LP18.csv','LP19.csv','LP20.csv','LP21.csv','LP22.csv','LP23.csv','LP24.csv']

curvature_files = ['LP01_curvature_peak.csv','LP02_curvature_peak.csv','LP03_curvature_peak.csv','LP04_curvature_peak.csv','LP05_curvature_peak.csv','LP06_curvature_peak.csv','LP07_curvature_peak.csv','LP08_curvature_peak.csv','LP09_curvature_peak.csv','LP10_curvature_peak.csv', 'LP11_curvature_peak.csv', 'LP12_curvature_peak.csv', 'LP13_curvature_peak.csv','LP14_curvature_peak.csv','LP15_curvature_peak.csv','LP16_curvature_peak.csv','LP17_curvature_peak.csv','LP18_curvature_peak.csv','LP19_curvature_peak.csv','LP20_curvature_peak.csv','LP21_curvature_peak.csv','LP22_curvature_peak.csv','LP23_curvature_peak.csv','LP24_curvature_peak.csv']

all_curve_target_120, all_curve_target_30, all_curve_target_180, all_curve_distractor = [[] for i in range(4)]
all_curve_target_samehemi, all_curve_target_opphemi = [[] for i in range(2)]

curvature_30, percentage_30 , curvature_120, percentage_120 = [[] for i in range(4)]

px_per_deg = 27.5
roi_check = 1.6*px_per_deg #this is 1.5 * size of circle stimuli

for sb in range(len(subjects)):
	
	# Read in CSV file as a data frame	
	os.chdir('/Users/bronagh/Documents/LePelley/LePelley_2/data/edf_reports')
	this_subject = pd.read_csv(subject_files[sb])
	
	list(this_subject.columns.values)
	
	exp_trials = this_subject[(this_subject['CURRENT_SAC_INDEX']== 1)]
	
	#Converting strings ('objects' in pandas terminology) that should be numeric to floats
	exp_trials = exp_trials.convert_objects(convert_numeric=True)
	
	os.chdir('/Users/bronagh/Documents/LePelley/LePelley_2/results/curvature')
	curve_df = pd.read_csv(curvature_files[sb], header = None, names = ['trial_no','curvature'])
	
	#Converting strings ('objects' in pandas terminology) that should be numeric to floats
	curve_df = curve_df.convert_objects(convert_numeric=True)
	
	# Merge the two dataframes
	sacc_curve = pd.merge(exp_trials,curve_df,how = 'outer', on= 'trial_no')
	
	#Drop the second saccade curvature that came from curve_df (from parsing script)
	curve_trials = sacc_curve.drop_duplicates(subset = 'trial_no')
	curve_trials = curve_trials[(np.isfinite(curve_trials['curvature'])) & (curve_trials['curvature']!= 0)]

	# Exclude practice and particular blocks per subject
	if subjects[sb] not in ('LP06','LP09','LP11','LP13','LP14','LP16','LP18','LP20','LP21'):
		trials = curve_trials[curve_trials['trial_no']>75] #skipping practice
	elif subjects[sb] == 'LP06':
		trials = curve_trials[((curve_trials['trial_no']>75) & (curve_trials['trial_no']<=615)) | (curve_trials['trial_no']>705)] #skip block 7
	elif subjects[sb] in ('LP09','LP11'):
		trials = curve_trials[(curve_trials['trial_no']>165)] #skipping block 1
	elif subjects[sb] == 'LP13':
		trials = curve_trials[((curve_trials['trial_no']>75) & (curve_trials['trial_no']<=900)) | (curve_trials['trial_no']>975)]# exclude block 10
	elif subjects[sb] == 'LP14':
		trials = curve_trials[((curve_trials['trial_no']>75) & (curve_trials['trial_no']<=900)) | ((curve_trials['trial_no']>975) & (curve_trials['trial_no']<=1335)) | (this_subject['trial_no']>1425)]
	elif subjects[sb] == 'LP16':
		trials = curve_trials[((curve_trials['trial_no']>75) & (curve_trials['trial_no']<=1245)) | (curve_trials['trial_no']>1335)]	
	elif subjects[sb] == 'LP18':
		trials = curve_trials[((curve_trials['trial_no']>75) & (curve_trials['trial_no']<=1335)) | (curve_trials['trial_no']>1425)]  # skipping block 15
	elif subjects[sb] == 'LP20':
		trials = curve_trials[((curve_trials['trial_no']>75) & (curve_trials['trial_no']<=1425)) | (curve_trials['trial_no']<=1515)]# skipping block 16
	elif subjects[sb] == 'LP21':
		trials = curve_trials[((curve_trials['trial_no']>75) & (curve_trials['trial_no']<=900)) | (curve_trials['trial_no']>=1065)] # skipping block 10 and 11
	
	# Remove blink trials,too slow/fast saccades
	trials = trials[(trials['CURRENT_SAC_CONTAINS_BLINK']== False)]
	trials = trials[((trials['saccade_onset_target'] > 80) & (trials['saccade_onset_target'] < trials['rt_cutoff'])) | ((trials['saccade_onset_distractor'] > 80) & (trials['saccade_onset_distractor'] < trials['rt_cutoff']))]
	
	# Starting from origin
	trials = trials[(trials['CURRENT_SAC_START_X'] > (trials['cur_x_start']-(roi_check))) & (trials['CURRENT_SAC_START_X'] < (trials['cur_x_start'] + (roi_check))) & (trials['CURRENT_SAC_START_Y']> (trials['cur_y_start']-(roi_check))) & (trials['CURRENT_SAC_START_Y']< (trials['cur_y_start']+(roi_check)))]
	
	target_trials_all = trials[trials['which_circle']=='t']
	distractor_trials_all = trials[trials['which_circle']=='d']

	# Ending in target circle
	#target_trials_all = target_trials_all[(target_trials_all['CURRENT_SAC_END_X'] > (target_trials_all['t_x']-(roi_check))) & (target_trials_all['CURRENT_SAC_END_X'] < (target_trials_all['t_x'] + (roi_check))) & (target_trials_all['CURRENT_SAC_END_Y']> (target_trials_all['t_y']-(roi_check))) & (target_trials_all['CURRENT_SAC_END_Y']< (target_trials_all['t_y']+(roi_check)))]
	
	# Ending in distractor circle
	#distractor_trials_all = distractor_trials_all[(distractor_trials_all['CURRENT_SAC_END_X'] > (distractor_trials_all['d_x']-(roi_check))) & (distractor_trials_all['CURRENT_SAC_END_X'] < (distractor_trials_all['d_x'] + (roi_check))) & (distractor_trials_all['CURRENT_SAC_END_Y']> (distractor_trials_all['d_y']-(roi_check))) & (distractor_trials_all['CURRENT_SAC_END_Y']< (distractor_trials_all['d_y']+(roi_check)))]
	
	# Target 120 or 30 deg
	target_trials_120 = target_trials_all[(target_trials_all['distractor_angle_deg'] == 119)]
	target_trials_30 = target_trials_all[(target_trials_all['distractor_angle_deg'] == 29)]
	target_trials_180 = target_trials_all[(target_trials_all['distractor_angle_deg'] == 180)]

	# Distractor 120 or 30 deg	
	distractor_trials_120 = distractor_trials_all[(distractor_trials_all['distractor_angle_deg'] == 119)]
	distractor_trials_30 = distractor_trials_all[(distractor_trials_all['distractor_angle_deg'] == 29)]
	
	# Target reward
	target_high_120 = target_trials_120[target_trials_120['reward_available'] == 0]
	target_low_120 = target_trials_120[target_trials_120['reward_available'] == 1]
	target_no_120 = target_trials_120[target_trials_120['reward_available'] == 2]
	
	target_high_30 = target_trials_30[target_trials_30['reward_available'] == 0]
	target_low_30 = target_trials_30[target_trials_30['reward_available'] == 1]
	target_no_30 = target_trials_30[target_trials_30['reward_available'] == 2]
	
	target_high_180 = target_trials_180[target_trials_180['reward_available'] == 0]
	target_low_180 = target_trials_180[target_trials_180['reward_available'] == 1]
	target_no_180 = target_trials_180[target_trials_180['reward_available'] == 2]
	
	# Hemifields for the 120 deg distractor
	target_samehemi =  target_trials_120[abs(target_trials_120['d_x']-target_trials_120['t_x'])<60]
	target_samehemi_high = target_samehemi[target_samehemi['reward_available'] == 0]
	target_samehemi_low = target_samehemi[target_samehemi['reward_available'] == 1]	
	target_samehemi_no = target_samehemi[target_samehemi['reward_available'] == 2]
	
	target_opphemi =  target_trials_120[abs(target_trials_120['d_x']-target_trials_120['t_x'])>60]
	target_opphemi_high = target_opphemi[target_opphemi['reward_available'] == 0]
	target_opphemi_low = target_opphemi[target_opphemi['reward_available'] == 1]	
	target_opphemi_no = target_opphemi[target_opphemi['reward_available'] == 2]

	# Target means : all + curvature values are TOWARDS the distractor (i.e. in a clockwise direction)
	mean_curve_target_high_120 = np.mean(target_high_120['curvature'])
	mean_curve_target_low_120 = np.mean(target_low_120['curvature'])
	mean_curve_target_no_120= np.mean(target_no_120['curvature'])
	
	mean_curve_target_high_30 = np.mean(target_high_30['curvature'])
	mean_curve_target_low_30 = np.mean(target_low_30['curvature'])
	mean_curve_target_no_30= np.mean(target_no_30['curvature'])
	
	mean_curve_target_high_180 = np.mean(target_high_180['curvature'])
	mean_curve_target_low_180 = np.mean(target_low_180['curvature'])
	mean_curve_target_no_180= np.mean(target_no_180['curvature'])
	
	mean_curve_target_samehemi_high = np.mean(target_samehemi_high['curvature'])
	mean_curve_target_samehemi_low = np.mean(target_samehemi_low['curvature'])
	mean_curve_target_samehemi_no = np.mean(target_samehemi_no['curvature'])	
	
	mean_curve_target_opphemi_high = np.mean(target_opphemi_high['curvature'])
	mean_curve_target_opphemi_low = np.mean(target_opphemi_low['curvature'])
	mean_curve_target_opphemi_no = np.mean(target_opphemi_no['curvature'])
	
	all_curve_target_120.append([mean_curve_target_high_120,mean_curve_target_low_120,mean_curve_target_no_120])
	all_curve_target_30.append([mean_curve_target_high_30,mean_curve_target_low_30,mean_curve_target_no_30])
	all_curve_target_180.append([mean_curve_target_high_180,mean_curve_target_low_180,mean_curve_target_no_180])	
	
	all_curve_target_samehemi.append([mean_curve_target_samehemi_high, mean_curve_target_samehemi_low, mean_curve_target_samehemi_no])
	all_curve_target_opphemi.append([mean_curve_target_opphemi_high, mean_curve_target_opphemi_low, mean_curve_target_opphemi_no])	
	
	# Distractor reward :  group over 30 and 120 deg since not very many trials	
	distractor_trials_30and120 = distractor_trials_all[((distractor_trials_all['distractor_angle_deg'] == 119) | (distractor_trials_all['distractor_angle_deg'] == 29))]
	
	distractor_high_30and120 = distractor_trials_30and120[distractor_trials_30and120['reward_available'] == 0]
	distractor_low_30and120 = distractor_trials_30and120[distractor_trials_30and120['reward_available'] == 1]
	distractor_no_30and120 = distractor_trials_30and120[distractor_trials_30and120['reward_available'] == 2]
	
	# Distractor means
	mean_curve_distractor_high = np.mean(distractor_high_30and120['curvature'])
	mean_curve_distractor_low = np.mean(distractor_low_30and120['curvature'])
	mean_curve_distractor_no= np.mean(distractor_no_30and120['curvature'])
	
	all_curve_distractor.append([mean_curve_distractor_high,mean_curve_distractor_low,mean_curve_distractor_no])

	print subjects[sb]
	
	# 30 degrees
	
	#High
	high_30_pos = target_high_30[target_high_30['curvature']>0]
	high_30_neg = target_high_30[target_high_30['curvature']<0]
	
	high_30_pos_mean = np.mean(high_30_pos['curvature'])
	high_30_neg_mean = np.mean(high_30_neg['curvature'])
	
	high_30_total = high_30_pos.shape[0] + high_30_neg.shape[0]
	high_30_pos_percentage = (high_30_pos.shape[0]/high_30_total)*100
	high_30_neg_percentage = (high_30_neg.shape[0]/high_30_total)*100
	
	#Low
	low_30_pos = target_low_30[target_low_30['curvature']>0]
	low_30_neg = target_low_30[target_low_30['curvature']<0]
	
	low_30_pos_mean = np.mean(low_30_pos['curvature'])
	low_30_neg_mean = np.mean(low_30_neg['curvature'])
	
	low_30_total = low_30_pos.shape[0] + low_30_neg.shape[0]
	low_30_pos_percentage = (low_30_pos.shape[0]/low_30_total)*100
	low_30_neg_percentage = (low_30_neg.shape[0]/low_30_total)*100
	
	#No
	no_30_pos = target_no_30[target_no_30['curvature']>0]
	no_30_neg = target_no_30[target_no_30['curvature']<0]
	
	no_30_pos_mean = np.mean(no_30_pos['curvature'])
	no_30_neg_mean = np.mean(no_30_neg['curvature'])
	
	no_30_total = no_30_pos.shape[0] + no_30_neg.shape[0]
	no_30_pos_percentage = (no_30_pos.shape[0]/no_30_total)*100
	no_30_neg_percentage = (no_30_neg.shape[0]/no_30_total)*100
	
	curvature_30.append([high_30_pos_mean,low_30_pos_mean,no_30_pos_mean, high_30_neg_mean,low_30_neg_mean,no_30_neg_mean])
	percentage_30.append([high_30_pos_percentage, low_30_pos_percentage, no_30_pos_percentage, high_30_neg_percentage, low_30_neg_percentage, no_30_neg_percentage])


	# 120 degrees
	
	#High
	high_120_pos = target_high_120[target_high_120['curvature']>0]
	high_120_neg = target_high_120[target_high_120['curvature']<0]
	
	high_120_pos_mean = np.mean(high_120_pos['curvature'])
	high_120_neg_mean = np.mean(high_120_neg['curvature'])
	
	high_120_total = high_120_pos.shape[0] + high_120_neg.shape[0]
	high_120_pos_percentage = (high_120_pos.shape[0]/high_120_total)*100
	high_120_neg_percentage = (high_120_neg.shape[0]/high_120_total)*100
	
	#Low
	low_120_pos = target_low_120[target_low_120['curvature']>0]
	low_120_neg = target_low_120[target_low_120['curvature']<0]
	
	low_120_pos_mean = np.mean(low_120_pos['curvature'])
	low_120_neg_mean = np.mean(low_120_neg['curvature'])
	
	low_120_total = low_120_pos.shape[0] + low_120_neg.shape[0]
	low_120_pos_percentage = (low_120_pos.shape[0]/low_120_total)*100
	low_120_neg_percentage = (low_120_neg.shape[0]/low_120_total)*100
	
	#No
	no_120_pos = target_no_30[target_no_30['curvature']>0]
	no_120_neg = target_no_30[target_no_30['curvature']<0]
	
	no_120_pos_mean = np.mean(no_30_pos['curvature'])
	no_120_neg_mean = np.mean(no_30_neg['curvature'])
	
	no_120_total = no_120_pos.shape[0] + no_120_neg.shape[0]
	no_120_pos_percentage = (no_120_pos.shape[0]/no_120_total)*100
	no_120_neg_percentage = (no_120_neg.shape[0]/no_120_total)*100
	
	curvature_120.append([high_120_pos_mean,low_120_pos_mean,no_120_pos_mean, high_120_neg_mean,low_120_neg_mean,no_120_neg_mean])
	percentage_120.append([high_120_pos_percentage, low_120_pos_percentage, no_120_pos_percentage, high_120_neg_percentage, low_120_neg_percentage, no_120_neg_percentage])

shell()