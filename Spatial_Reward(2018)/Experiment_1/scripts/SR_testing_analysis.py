from __future__ import division

import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from scipy.optimize import curve_fit
#import seaborn as sns
import sympy as sym
from IPython import embed as shell

# Read in Eye-tracker Saccade Report file as a data frame -> ALL participants
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/analysis/fixation_reports')
saccades_main_df = pd.read_csv('SR01-24_TE.csv') 
# Can use variable RECORDING_SESSION_LABEL to get the subject name, e.g. 'S01tr1' 

# In this experiment, reward is always in different hemifields	
subjects_test1 = ['SR01TE1','SR02TE1','SR03TE1','SR04TE1','SR05TE1','SR06TE1','SR07TE1','SR08TE1','SR09TE1','SR10TE1','SR11TE1','SR12TE1','SR13TE1','SR14TE1','SR15TE1','SR16TE1','SR17TE1','SR18TE1','SR19TE1','SR20TE1','SR21TE1','SR22TE1','SR23TE1','SR24TE1'] 
subjects_test2 = ['SR01TE2','SR02TE2','SR03TE2','SR04TE2','SR05TE2','SR06TE2','SR07TE2','SR08TE2','SR09TE2','SR10TE2','SR11TE2','SR12TE2','SR13TE2','SR14TE2','SR15TE2','SR16TE2','SR17TE2','SR18TE2','SR19TE2','SR20TE2','SR21TE2','SR22TE2','SR23TE2','SR24TE2']
	
subject_nr= ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
# High v Low = 180 degrees apart (different hemifield in both vertical and horizontal axis)
# subjects_test1 = ['SR01TE1','SR02TE1','SR03TE1','SR04TE1','SR09TE1','SR10TE1','SR11TE1','SR12TE1','SR17TE1','SR18TE1','SR19TE1','SR20TE1']
# subjects_test2 = ['SR01TE2','SR02TE2','SR03TE2','SR04TE2','SR09TE2','SR10TE2','SR11TE2','SR12TE2','SR17TE2','SR18TE2','SR19TE2','SR20TE2']

# High v Low = 90 degrees apart (different hemifield in vertical only)
# subjects_test1 = ['SR05TE1','SR06TE1','SR07TE1','SR08TE1','SR13TE1','SR14TE1','SR15TE1','SR16TE1','SR21TE1','SR22TE1','SR23TE1','SR24TE1']
# subjects_test2 = ['SR05TE2','SR06TE2','SR07TE2','SR08TE2','SR13TE2','SR14TE2','SR15TE2','SR16TE2','SR21TE2','SR22TE2','SR23TE2','SR24TE2']

timeout = 1000
nr_blocks = 1
nr_trials_per_condition = 30.0
nr_timebins = 10
trials_per_condition = 30

#Carried out in K1F46
#Screen resolution: 1680 x 1050 pixels
#Screen dimensions: 47.5 X 29.5 cm
#Distance to screen: 70 cm
#Refresh rate: 120 Hz

xc = 840
yc = 525

x_pixels = 134 # pixels per 3 degrees of visual angle on each side; this is usually the visual angle taken as a ROI 
y_pixels = 132 


### PRE-TEST

all_subjects, all_subjects_competition, all_subjects_bin1, all_subjects_bin2, accuracy_all_subjects = [[] for i in range(5)]
all_subjects_saccades = []
mean_template_1 = []
regression_file_1 = []
regression_high_1, regression_low_1 = [[] for i in range(2)]

for sb in range(len(subjects_test1)):
	print(subjects_test1[sb])

	subject_df = saccades_main_df[saccades_main_df['RECORDING_SESSION_LABEL'] == subjects_test1[sb]]

	subject_df = subject_df.convert_objects(convert_numeric=True)
	
	all_trials_df = subject_df[(subject_df['response_key'] == subject_df['answer']) & (subject_df['target_correct'] == 1) & (subject_df['response_time']<timeout) & (subject_df['response_time'] > 200) & (subject_df['CURRENT_FIX_BLINK_AROUND'] == 'NONE') & (subject_df['CURRENT_FIX_X'] > (xc-x_pixels)) & (subject_df['CURRENT_FIX_X'] < (xc+x_pixels)) & (subject_df['CURRENT_FIX_Y'] > (yc-y_pixels)) & (subject_df['CURRENT_FIX_Y'] < (yc+y_pixels)) & ((np.isfinite(subject_df['NEXT_SAC_END_X'])==False) | ((subject_df['NEXT_SAC_END_X'] > (xc-x_pixels)) & (subject_df['NEXT_SAC_END_X'] < (xc+x_pixels)) & (subject_df['NEXT_SAC_END_Y'] > (yc-y_pixels)) & (subject_df['NEXT_SAC_END_Y'] < (yc+y_pixels))))]

	#drop duplicates
	pre_correct_df = all_trials_df.drop_duplicates(subset = 'trial_number')
	
	#Exclude outliers : only if looking at means
	rt_mean = np.mean(pre_correct_df['response_time'])
	rt_std = np.std(pre_correct_df['response_time'])
	outlier_adjust = rt_std*3 #3 SDs
	outlier_plus = rt_mean + outlier_adjust
	outlier_minus = rt_mean - outlier_adjust

	correct_df = pre_correct_df[(pre_correct_df['response_time']<outlier_plus) & (pre_correct_df['response_time']>outlier_minus)]

	#Condition information
	high_df = correct_df[correct_df['high_location']==correct_df['target_position']]
	low_df = correct_df[correct_df['low_location']==correct_df['target_position']]

	#Getting neutral positions: neut1 = same hemisphere as high, neut2 = same hemisphere as low
	if correct_df['high_location'].iloc[0] == 'north_west':
		neut1 = 'south_west'
	elif correct_df['high_location'].iloc[0] == 'north_east':
		neut1 = 'south_east'
	elif correct_df['high_location'].iloc[0] == 'south_west':
		neut1 = 'north_west'
	elif correct_df['high_location'].iloc[0] == 'south_east':
		neut1 = 'north_east'
	
	if correct_df['low_location'].iloc[0] == 'north_west':
		neut2 = 'south_west'
	elif correct_df['low_location'].iloc[0] == 'north_east':
		neut2 = 'south_east'
	elif correct_df['low_location'].iloc[0] == 'south_west':
		neut2 = 'north_west'
	elif correct_df['low_location'].iloc[0] == 'south_east':
		neut2 = 'north_east'

	neut1_df = correct_df[(correct_df['target_position']== neut1)]
	neut2_df = correct_df[(correct_df['target_position']== neut2)]
		
	#print 'high, low, neut1, neut2: ', [high_df.shape[0],low_df.shape[0], neut1_df.shape[0], neut2_df.shape[0]]
	
	# Quick check on eye movements (includes anything picked up as a saccade by the eyetracker)
	saccade_trials = subject_df[(subject_df['NEXT_SAC_END_X']>0) | (subject_df['NEXT_SAC_END_Y']>0)]
	high_saccades = len(saccade_trials[saccade_trials['high_location']==saccade_trials['target_position']])/len(saccade_trials)
	low_saccades = len(saccade_trials[saccade_trials['low_location']==saccade_trials['target_position']])/len(saccade_trials)
	neut1_saccades = len(saccade_trials[saccade_trials['target_position']==neut1])/len(saccade_trials)
	neut2_saccades = len(saccade_trials[saccade_trials['target_position']==neut2])/len(saccade_trials)
	all_subjects_saccades.append([high_saccades, low_saccades, neut1_saccades, neut2_saccades])
	
	# Back to RTs
	high_mean_rt = np.mean(high_df['response_time'])
	low_mean_rt = np.mean(low_df['response_time'])
	neut1_mean_rt = np.mean(neut1_df['response_time'])
	neut2_mean_rt = np.mean(neut2_df['response_time'])
	
	# Split into 2 TIMEBINS - suggested by Jan, Feb 2017
	# Bin 1
	high_1 = high_df[:(int(high_df.shape[0]/2.0))]
	high_1_mean_rt = np.mean(high_1['response_time'])
	
	low_1 = low_df[:(int(low_df.shape[0]/2.0))]
	low_1_mean_rt = np.mean(low_1['response_time'])	
	
	neut1_1 = neut1_df[:(int(neut1_df.shape[0]/2.0))]
	neut1_1_mean_rt = np.mean(neut1_1['response_time'])	
	
	neut2_1 = neut2_df[:(int(neut2_df.shape[0]/2.0))]
	neut2_1_mean_rt = np.mean(neut2_1['response_time'])	
	
	# Bin 2
	high_2 = high_df[(int(high_df.shape[0]/2.0)):]
	high_2_mean_rt = np.mean(high_2['response_time'])
	
	low_2 = low_df[(int(low_df.shape[0]/2.0)):]
	low_2_mean_rt = np.mean(low_2['response_time'])	
	
	neut1_2 = neut1_df[(int(neut1_df.shape[0]/2.0)):]
	neut1_2_mean_rt = np.mean(neut1_2['response_time'])	
	
	neut2_2 = neut2_df[(int(neut2_df.shape[0]/2.0)):]
	neut2_2_mean_rt = np.mean(neut2_2['response_time'])


	# Competition between all locations
	high_low_diff_mean = np.mean(high_df['response_time']) - np.mean(low_df['response_time'])
	high_neut1_diff_mean = np.mean(high_df['response_time']) - np.mean(neut1_df['response_time'])
	high_neut2_diff_mean = np.mean(high_df['response_time']) - np.mean(neut2_df['response_time'])
	low_neut1_diff_mean = np.mean(low_df['response_time']) - np.mean(neut1_df['response_time'])
	low_neut2_diff_mean = np.mean(low_df['response_time']) - np.mean(neut2_df['response_time'])
	neut1_neut2_diff_mean = np.mean(neut1_df['response_time']) - np.mean(neut2_df['response_time'])
	
	# RT
	all_subjects.append([high_mean_rt,low_mean_rt,neut1_mean_rt,neut2_mean_rt])
	all_subjects_competition.append([high_low_diff_mean,high_neut1_diff_mean, high_neut2_diff_mean,low_neut1_diff_mean,low_neut2_diff_mean,neut1_neut2_diff_mean])
	
	all_subjects_bin1.append([high_1_mean_rt,low_1_mean_rt,neut1_1_mean_rt,neut2_1_mean_rt])
	all_subjects_bin2.append([high_2_mean_rt,low_2_mean_rt,neut1_2_mean_rt,neut2_2_mean_rt])
	
	# Accuracy
	accuracy_all_subjects.append([high_df.shape[0]*100/trials_per_condition,low_df.shape[0]*100/trials_per_condition, neut1_df.shape[0]*100/trials_per_condition, neut2_df.shape[0]*100/trials_per_condition])
	
	# Reviews 
	# Template duration
	for i in range(len(high_df)):
		regression_high_1.append([int(subject_nr[sb]),high_df.iloc[i]['response_time']/1000,1,high_df.iloc[i]['temp_to_stim']/1000])
		regression_file_1.append([int(subject_nr[sb]),high_df.iloc[i]['response_time']/1000,1,high_df.iloc[i]['temp_to_stim']/1000])
	
	for i in range(len(low_df)):
		regression_low_1.append([int(subject_nr[sb]),low_df.iloc[i]['response_time']/1000,-1,low_df.iloc[i]['temp_to_stim']/1000])
		regression_file_1.append([int(subject_nr[sb]),low_df.iloc[i]['response_time']/1000,-1,low_df.iloc[i]['temp_to_stim']/1000])

	template_rt_correlation=stats.pearsonr(correct_df['temp_to_stim'],correct_df['response_time'])
	print(template_rt_correlation)

	mean_template_1.append([high_df['temp_to_stim'].mean(),low_df['temp_to_stim'].mean(),neut1_df['temp_to_stim'].mean(),neut2_df['temp_to_stim'].mean()])
	### INTER-TRIAL SLOWING, as per Doug Munoz ###

	# DO HERE

pre_test = pd.DataFrame(all_subjects)	
pre_test_competition = pd.DataFrame(all_subjects_competition)	
pre_test_accuracy = pd.DataFrame(accuracy_all_subjects)
pre_test_accuracy['all']=pre_test_accuracy.mean(axis=1)

pre_test_bin1 = pd.DataFrame(all_subjects_bin1)
pre_test_bin2 = pd.DataFrame(all_subjects_bin2)

pre_test_saccades = pd.DataFrame(all_subjects_saccades)

## POST-TEST ###

all_subjects, all_subjects_competition, all_subjects_bin1, all_subjects_bin2, accuracy_all_subjects = [[] for i in range(5)]
all_subjects_saccades = []
mean_template_2 = []
regression_file_2 = []
regression_high_2, regression_low_2 = [[] for i in range(2)]

for sb in range(len(subjects_test2)):
	print(subjects_test2[sb])
	
	subject_df = saccades_main_df[saccades_main_df['RECORDING_SESSION_LABEL'] == subjects_test2[sb]]

	subject_df = subject_df.convert_objects(convert_numeric=True)
	
	all_trials_df = subject_df[(subject_df['response_key'] == subject_df['answer']) & (subject_df['target_correct'] == 1) & (subject_df['response_time']<timeout) & (subject_df['response_time'] > 200) & (subject_df['CURRENT_FIX_BLINK_AROUND'] == 'NONE') & (subject_df['CURRENT_FIX_X'] > (xc-x_pixels)) & (subject_df['CURRENT_FIX_X'] < (xc+x_pixels)) & (subject_df['CURRENT_FIX_Y'] > (yc-y_pixels)) & (subject_df['CURRENT_FIX_Y'] < (yc+y_pixels)) & ((np.isfinite(subject_df['NEXT_SAC_END_X'])==False) | ((subject_df['NEXT_SAC_END_X'] > (xc-x_pixels)) & (subject_df['NEXT_SAC_END_X'] < (xc+x_pixels)) & (subject_df['NEXT_SAC_END_Y'] > (yc-y_pixels)) & (subject_df['NEXT_SAC_END_Y'] < (yc+y_pixels))))]

	#drop duplicates
	pre_correct_df = all_trials_df.drop_duplicates(subset = 'trial_number')	

	# #Exclude outliers : only if looking at means
	rt_mean = np.mean(pre_correct_df['response_time'])
	rt_std = np.std(pre_correct_df['response_time'])
	outlier_adjust = rt_std*3 #3 SDs
	outlier_plus = rt_mean + outlier_adjust
	outlier_minus = rt_mean - outlier_adjust

	correct_df = pre_correct_df[(pre_correct_df['response_time']<outlier_plus) & (pre_correct_df['response_time']>outlier_minus)]
	
	# Only for median analysis
	# correct_df = pre_correct_df
	
	#Condition information
	high_df = correct_df[correct_df['high_location']==correct_df['target_position']]
	low_df = correct_df[correct_df['low_location']==correct_df['target_position']]

	#Getting neutral positions: neut1 = same hemisphere as high, neut2 = same hemisphere as low
	if correct_df['high_location'].iloc[0] == 'north_west':
		neut1 = 'south_west'
	elif correct_df['high_location'].iloc[0] == 'north_east':
		neut1 = 'south_east'
	elif correct_df['high_location'].iloc[0] == 'south_west':
		neut1 = 'north_west'
	elif correct_df['high_location'].iloc[0] == 'south_east':
		neut1 = 'north_east'
	
	if correct_df['low_location'].iloc[0] == 'north_west':
		neut2 = 'south_west'
	elif correct_df['low_location'].iloc[0] == 'north_east':
		neut2 = 'south_east'
	elif correct_df['low_location'].iloc[0] == 'south_west':
		neut2 = 'north_west'
	elif correct_df['low_location'].iloc[0] == 'south_east':
		neut2 = 'north_east'

	neut1_df = correct_df[(correct_df['target_position']== neut1)]
	neut2_df = correct_df[(correct_df['target_position']== neut2)]

	neut1_df = correct_df[(correct_df['target_position']== neut1)]
	neut2_df = correct_df[(correct_df['target_position']== neut2)]

	#print 'high, low, neut1, neut2: ', [high_df.shape[0],low_df.shape[0], neut1_df.shape[0], neut2_df.shape[0]]
			
	# Quick check on eye movements (includes anything picked up as a saccade by the eyetracker)
	saccade_trials = subject_df[(subject_df['NEXT_SAC_END_X']>0) | (subject_df['NEXT_SAC_END_Y']>0)]
	high_saccades = len(saccade_trials[saccade_trials['high_location']==saccade_trials['target_position']])/len(saccade_trials)
	low_saccades = len(saccade_trials[saccade_trials['low_location']==saccade_trials['target_position']])/len(saccade_trials)
	neut1_saccades = len(saccade_trials[saccade_trials['target_position']==neut1])/len(saccade_trials)
	neut2_saccades = len(saccade_trials[saccade_trials['target_position']==neut2])/len(saccade_trials)
	all_subjects_saccades.append([high_saccades, low_saccades, neut1_saccades, neut2_saccades])
	
	# Back to RTs
	high_mean_rt = np.mean(high_df['response_time'])
	low_mean_rt = np.mean(low_df['response_time'])
	neut1_mean_rt = np.mean(neut1_df['response_time'])
	neut2_mean_rt = np.mean(neut2_df['response_time'])
	
	# Split into 2 TIMEBINS - suggested by Jan, Feb 2017
	# Bin 1
	high_1 = high_df[:(int(high_df.shape[0]/2.0))]
	high_1_mean_rt = np.mean(high_1['response_time'])
	
	low_1 = low_df[:(int(low_df.shape[0]/2.0))]
	low_1_mean_rt = np.mean(low_1['response_time'])	
	
	neut1_1 = neut1_df[:(int(neut1_df.shape[0]/2.0))]
	neut1_1_mean_rt = np.mean(neut1_1['response_time'])	
	
	neut2_1 = neut2_df[:(int(neut2_df.shape[0]/2.0))]
	neut2_1_mean_rt = np.mean(neut2_1['response_time'])	
	
	# Bin 2
	high_2 = high_df[(int(high_df.shape[0]/2.0)):]
	high_2_mean_rt = np.mean(high_2['response_time'])
	
	low_2 = low_df[(int(low_df.shape[0]/2.0)):]
	low_2_mean_rt = np.mean(low_2['response_time'])	
	
	neut1_2 = neut1_df[(int(neut1_df.shape[0]/2.0)):]
	neut1_2_mean_rt = np.mean(neut1_2['response_time'])	
	
	neut2_2 = neut2_df[(int(neut2_df.shape[0]/2.0)):]
	neut2_2_mean_rt = np.mean(neut2_2['response_time'])


	# Competition between all locations
	high_low_diff_mean = np.mean(high_df['response_time']) - np.mean(low_df['response_time'])
	high_neut1_diff_mean = np.mean(high_df['response_time']) - np.mean(neut1_df['response_time'])
	high_neut2_diff_mean = np.mean(high_df['response_time']) - np.mean(neut2_df['response_time'])
	low_neut1_diff_mean = np.mean(low_df['response_time']) - np.mean(neut1_df['response_time'])
	low_neut2_diff_mean = np.mean(low_df['response_time']) - np.mean(neut2_df['response_time'])
	neut1_neut2_diff_mean = np.mean(neut1_df['response_time']) - np.mean(neut2_df['response_time'])
	
	# RT
	all_subjects.append([high_mean_rt,low_mean_rt,neut1_mean_rt,neut2_mean_rt])
	all_subjects_competition.append([high_low_diff_mean,high_neut1_diff_mean, high_neut2_diff_mean,low_neut1_diff_mean,low_neut2_diff_mean,neut1_neut2_diff_mean])
	
	all_subjects_bin1.append([high_1_mean_rt,low_1_mean_rt,neut1_1_mean_rt,neut2_1_mean_rt])
	all_subjects_bin2.append([high_2_mean_rt,low_2_mean_rt,neut1_2_mean_rt,neut2_2_mean_rt])	
	
	# Accuracy
	accuracy_all_subjects.append([high_df.shape[0]*100/trials_per_condition,low_df.shape[0]*100/trials_per_condition, neut1_df.shape[0]*100/trials_per_condition, neut2_df.shape[0]*100/trials_per_condition])

	# Reviews 
	# Template duration

	for i in range(len(high_df)):
		regression_high_2.append([int(subject_nr[sb]),high_df.iloc[i]['response_time']/1000,1,high_df.iloc[i]['temp_to_stim']/1000])
		regression_file_2.append([int(subject_nr[sb]),high_df.iloc[i]['response_time']/1000,1,high_df.iloc[i]['temp_to_stim']/1000])
	
	for i in range(len(low_df)):
		regression_low_2.append([int(subject_nr[sb]),low_df.iloc[i]['response_time']/1000,-1,low_df.iloc[i]['temp_to_stim']/1000])
		regression_file_2.append([int(subject_nr[sb]),low_df.iloc[i]['response_time']/1000,-1,low_df.iloc[i]['temp_to_stim']/1000])


	template_rt_correlation=stats.pearsonr(correct_df['temp_to_stim'],correct_df['response_time'])
	print(template_rt_correlation)

	mean_template_2.append([high_df['temp_to_stim'].mean(),low_df['temp_to_stim'].mean(),neut1_df['temp_to_stim'].mean(),neut2_df['temp_to_stim'].mean()])

	### INTER-TRIAL SLOWING, as per Doug Munoz ###


	# DO HERE

post_test = pd.DataFrame(all_subjects)	
post_test_competition = pd.DataFrame(all_subjects_competition)	
post_test_accuracy = pd.DataFrame(accuracy_all_subjects)
post_test_accuracy['all']=post_test_accuracy.mean(axis=1)

post_test_bin1 = pd.DataFrame(all_subjects_bin1)
post_test_bin2 = pd.DataFrame(all_subjects_bin2)

post_test_saccades = pd.DataFrame(all_subjects_saccades)


# Combining neutral locations
neutcomb_pre = np.mean([pre_test[2],pre_test[3]],axis=0)
neutcomb_post = np.mean([post_test[2],post_test[3]],axis=0)

os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results')

# RT
pre_test.to_csv('test_pre.csv', header = ['High_1','Low_1','No1_1','No2_1'], index = False)
post_test.to_csv('test_post.csv', header = ['High_2','Low_2','No1_2','No2_2'], index = False)

pre_test_competition.to_csv('test_competition_pre.csv', header = None, index = False)
post_test_competition.to_csv('test_competition_post.csv', header = None, index = False)


pre_test_bin1.to_csv('test_pre_bin1.csv', header = ['High_1','Low_1','No1_1','No2_1'], index = False)
pre_test_bin2.to_csv('test_pre_bin2.csv', header = ['High_1','Low_1','No1_1','No2_1'], index = False)

post_test_bin1.to_csv('test_post_bin1.csv', header = ['High_1','Low_1','No1_1','No2_1'], index = False)
post_test_bin2.to_csv('test_post_bin2.csv', header = ['High_1','Low_1','No1_1','No2_1'], index = False)

# Saccades
pre_test_saccades.to_csv('test_saccades_pre.csv', header = None, index = False)
post_test_saccades.to_csv('test_saccades_post.csv', header = None, index = False)

# Accuracy
pre_test_accuracy.to_csv('test_pre_accuracy.csv', header = None, index = False)
post_test_accuracy.to_csv('test_post_accuracy.csv', header = None, index = False)
accuracy_gain = post_test_accuracy - pre_test_accuracy
stats.ttest_rel(accuracy_gain[0],accuracy_gain[1]) # checked - no differences across conditions

# Gains
gain_high = pre_test[0] - post_test[0]
gain_low = pre_test[1] - post_test[1]
gain_neut1 = pre_test[2] - post_test[2]
gain_neut2 = pre_test[3] - post_test[3]
gain_neutcomb = pd.DataFrame(np.mean((gain_neut1, gain_neut2), axis=0))

test_gains = pd.concat([gain_high,gain_low,gain_neut1,gain_neut2], axis=1)

gain_high.to_csv('test_gain_high.csv',header = None, index = False)
gain_low.to_csv('test_gain_low.csv',header = None, index = False)
gain_neut1.to_csv('test_gain_neut1.csv',header = None, index = False)
gain_neut2.to_csv('test_gain_neut2.csv',header = None, index = False)
test_gains.to_csv('test_gains.csv', header = ['high','low','neut1','neut2'], index=False)

# Reviews
# Template durations
template_df_1 = pd.DataFrame(mean_template_1)
template_df_2 = pd.DataFrame(mean_template_2)

regression_file_df_1 = pd.DataFrame(regression_file_1, columns=['Subject','RT','condition','template_duration'])
regression_file_df_1.to_csv('highlow_regression_file_baseline.csv',index = False)

regression_file_df_2 = pd.DataFrame(regression_file_2, columns=['Subject','RT','condition','template_duration'])
regression_file_df_2.to_csv('highlow_regression_file_test.csv',index = False)

regression_high_df_1 = pd.DataFrame(regression_high_1, columns=['Subject','RT','condition','template_duration'])
regression_high_df_1.to_csv('high_regression_file_baseline.csv',index = False)

regression_high_df_2 = pd.DataFrame(regression_high_2, columns=['Subject','RT','condition','template_duration'])
regression_high_df_2.to_csv('high_regression_file_test.csv',index = False)

regression_low_df_1 = pd.DataFrame(regression_low_1, columns=['Subject','RT','condition','template_duration'])
regression_low_df_1.to_csv('low_regression_file_baseline.csv',index = False)

regression_low_df_2 = pd.DataFrame(regression_low_2, columns=['Subject','RT','condition','template_duration'])
regression_low_df_2.to_csv('low_regression_file_test.csv',index = False)

# Bar plot
width = 0.08

fig, ax = plt.subplots()
rects1 = ax.bar(0.15, np.mean(gain_high), width, hatch='/', color='white', edgecolor='k', yerr=np.std(gain_high)/np.sqrt(len(subjects_test2)), ecolor = 'k')
rects2 = ax.bar(0.35, np.mean(gain_low), width, hatch='/',color='white', edgecolor='k',yerr=np.std(gain_low)/np.sqrt(len(subjects_test2)), ecolor = 'k')
rects3 = ax.bar(0.55, np.mean(gain_neutcomb), width, hatch='/',color='white', edgecolor='k', yerr=np.std(gain_neutcomb)/np.sqrt(len(subjects_test2)), ecolor = 'k')
ax.set_ylabel('RT improvement (ms)', fontsize = '12')
ax.set_xlabel('Reward Location', fontsize = '12')
ax.yaxis.set_label_coords(-0.07, 0.5) # -.1 on x-axis, .5 = halfway along y-axis
ax.xaxis.set_label_coords(0.5, -0.07)
ax.axhline(y=0, color='k')
ax.set_ylim(0, 60)
ax.set_xlim(0, 0.7)
ax.set_xticks([0.15, 0.35, 0.55])
ax.set_xticklabels( ('High','Low', 'No') )
# Use following 2 lines to put significance marker between high and low reward location
ax.axhline(y=50, color='k', xmin=0.2, xmax=0.5)
#ax.axhline(y=55, color='k', xmin=0.2, xmax=0.78)
ax.text(0.235, 51, '**', fontweight='bold')
#ax.text(0.34, 56, '**', fontweight='bold')
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results/figs')
plt.savefig('Test_gain.pdf')
plt.savefig('Test_gain.png')
plt.show()
plt.close()

### WITHIN-SUBJECT ERRORBARS - Cousineau 2005 ###

pre_test.columns=['High_1','Low_1','No1_1','No2_1']
post_test.columns=['High_2','Low_2','No1_2','No2_2']
pre_test['Nocomb_1'] = pre_test[['No1_1','No2_1']].mean(axis=1)
post_test['Nocomb_2'] = post_test[['No1_2','No2_2']].mean(axis=1)

test_all = pd.concat([pre_test,post_test], axis=1)

test_all['mean_high']=test_all[['High_1','High_2']].mean(axis=1)
test_all['mean_low']=test_all[['Low_1','Low_2']].mean(axis=1)
test_all['mean_nocomb']=test_all[['Nocomb_1','Nocomb_2']].mean(axis=1)

allsubs_mean_high = np.mean(test_all['mean_high'])
allsubs_mean_low = np.mean(test_all['mean_low'])
allsubs_mean_nocomb = np.mean(test_all['mean_nocomb'])

# New columns for mean of each condition across reward condition (per participant)
std_calc = test_all.copy()

std_calc['High_1']=test_all['High_1']-test_all['mean_high'] + allsubs_mean_high
std_calc['High_2']=test_all['High_2']-test_all['mean_high'] + allsubs_mean_high

std_calc['Low_1']=test_all['Low_1']-test_all['mean_low'] + allsubs_mean_low
std_calc['Low_2']=test_all['Low_2']-test_all['mean_low'] + allsubs_mean_low

std_calc['Nocomb_1']=test_all['Nocomb_1']-test_all['mean_nocomb'] + allsubs_mean_nocomb
std_calc['Nocomb_2']=test_all['Nocomb_2']-test_all['mean_nocomb'] + allsubs_mean_nocomb

# Reward data across blocks
high_list = [test_all['High_1'].mean(),test_all['High_2'].mean()]
low_list = [test_all['Low_1'].mean(),test_all['Low_2'].mean()]
nocomb_list = [test_all['Nocomb_1'].mean(),test_all['Nocomb_2'].mean()]

# STD
stdev_list = [std_calc['High_1'].std()/np.sqrt(len(subjects_test1)), std_calc['High_2'].std()/np.sqrt(len(subjects_test1)),std_calc['Low_1'].std()/np.sqrt(len(subjects_test1)),std_calc['Low_2'].std()/np.sqrt(len(subjects_test1)),std_calc['Nocomb_1'].std()/np.sqrt(len(subjects_test1)),std_calc['Nocomb_2'].std()/np.sqrt(len(subjects_test1))]

fig = plt.figure()
ax = fig.add_subplot(111)
x=np.arange(1,3,1) # 2 blocks
plt.xlabel('Block', fontsize = '12')
plt.ylabel('RT (ms)', fontsize = '12')
ax.yaxis.set_label_coords(-0.1, 0.5) # -.1 on x-axis, .5 = halfway along y-axis
ax.xaxis.set_label_coords(0.5, -0.09) 
plt.errorbar(x,high_list, yerr = stdev_list[:2], linestyle = '-', color = 'k', label = 'High reward') 
plt.errorbar(x,low_list, yerr = stdev_list[2:4], linestyle = '--', color = 'k', label = 'Low reward')  
plt.errorbar(x,nocomb_list, yerr = stdev_list[4:6], linestyle = '-.', color = 'k', label = 'No reward')
plt.legend(loc='best')
plt.xlim(xmin=0.5, xmax = 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels( ('Baseline','Test') )
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results/figs')
plt.savefig('Testing_RT.pdf')
plt.savefig('Testing_RT.png')
plt.show()
plt.close()


## Bar plot for differences in 90 v 180 degree group
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results')

test_pre_90 = pd.read_csv('test_pre_90.csv', header=None, skiprows=1)
test_post_90 = pd.read_csv('test_post_90.csv', header=None, skiprows=1)
test_pre_180 = pd.read_csv('test_pre_180.csv', header=None, skiprows=1)
test_post_180 = pd.read_csv('test_post_180.csv', header=None, skiprows=1)

gains_90= test_pre_90 - test_post_90
gains_180= test_pre_180 - test_post_180

gains_90.columns=['high','low','no1','no2']
gains_180.columns=['high','low','no1','no2']

gains_90['neutcomb']=(gains_90['no1']+gains_90['no2'])/2
gains_180['neutcomb']=(gains_180['no1']+gains_180['no2'])/2

# Bar plot
angle_group_size = 12

width = 0.1
fig, ax = plt.subplots()
# High
rects1 = ax.bar(0.15, np.mean(gains_90['high']), width, hatch='/', color='b', edgecolor='b', yerr=np.std(gains_90['high'])/np.sqrt(angle_group_size), ecolor = 'k', label='90 degree group')
rects2 = ax.bar(0.25, np.mean(gains_180['high']), width, hatch='/',color='g', edgecolor='g',yerr=np.std(gains_180['high'])/np.sqrt(angle_group_size), ecolor = 'k', label='180 degree group')
# Low
rects3 = ax.bar(0.60, np.mean(gains_90['low']), width, hatch='/',color='b', edgecolor='b', yerr=np.std(gains_90['low'])/np.sqrt(angle_group_size), ecolor = 'k')
rects4 = ax.bar(0.70, np.mean(gains_180['low']), width, hatch='/', color='g', edgecolor='g', yerr=np.std(gains_180['low'])/np.sqrt(angle_group_size), ecolor = 'k')
# No
rects5 = ax.bar(1.05, np.mean(gains_90['neutcomb']), width, hatch='/',color='b', edgecolor='b',yerr=np.std(gains_90['neutcomb'])/np.sqrt(angle_group_size), ecolor = 'k')
rects6 = ax.bar(1.15, np.mean(gains_180['neutcomb']), width, hatch='/',color='g', edgecolor='g', yerr=np.std(gains_180['neutcomb'])/np.sqrt(angle_group_size), ecolor = 'k')

ax.set_ylabel('RT improvement (ms)', fontsize = '12')
ax.set_xlabel('Reward Location', fontsize = '12')
ax.yaxis.set_label_coords(-0.07, 0.5) # -.1 on x-axis, .5 = halfway along y-axis
ax.xaxis.set_label_coords(0.45, -0.07)
ax.axhline(y=0, color='k')
ax.set_ylim(0, 65)
ax.set_xlim(0, 1.5)
ax.set_xticks([0.2, 0.65, 1.1])
ax.set_xticklabels( ('High','Low', 'No') )
ax.legend((rects1[0], rects2[0]), ('90 degrees', '180 degrees'))
# Use following 2 lines to put significance marker between high and low reward location
ax.axhline(y=60, color='k', xmin=0.13, xmax=0.42)
ax.text(0.39, 60.5, '**', fontweight='bold')
ax.axhline(y=56, color='k', xmin=0.17, xmax=0.46)
ax.text(0.44, 56.5, '***', fontweight='bold')
ax.axhline(y=52, color='k', xmin=0.17, xmax=0.76)
ax.text(0.68, 52.5, '**', fontweight='bold')

os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results/figs')
plt.savefig('Test_gain_groups.pdf')
plt.savefig('Test_gain_groups.png')
plt.show()
plt.close()



shell()







