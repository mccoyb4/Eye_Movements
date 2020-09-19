
from __future__ import division

import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
from scipy import stats
import scipy.optimize as optimization
import statsmodels.api as sm
from scipy.optimize import curve_fit
import sympy as sym
from IPython import embed as shell

#All subjects
subjects = ['SR01TR','SR02TR','SR03TR','SR04TR','SR05TR','SR06TR','SR07TR','SR08TR','SR09TR','SR10TR','SR11TR','SR12TR','SR13TR','SR14TR','SR15TR','SR16TR','SR17TR','SR18TR','SR19TR','SR20TR','SR21TR','SR22TR','SR23TR','SR24TR']

# High v Low = 180 degrees apart (different hemifield in both vertical and horizontal axis)
# subjects = ['SR01TR','SR02TR','SR03TR','SR04TR','SR09TR','SR10TR','SR11TR','SR12TR','SR17TR','SR18TR','SR19TR','SR20TR']

# High v Low = 90 degrees apart (different hemifield in vertical only)
# subjects = ['SR05TR','SR06TR','SR07TR','SR08TR','SR13TR','SR14TR','SR15TR','SR16TR','SR21TR','SR22TR','SR23TR','SR24TR']

# Read in Eye-tracker Saccade Report file as a data frame -> ALL participants
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/analysis/saccade_reports')
saccades_main_df = pd.read_csv('SR01-24_TR.csv') 
# Can use variable RECORDING_SESSION_LABEL to get the subject name, e.g. 'S01tr1' 

os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results')

##################

#Experiment details

nr_trials = 480
nr_blocks = 4 #1 practice, 3 reward blocks
timeout = 500

#Carried out in K1F46
#Screen resolution: 1680 x 1050 pixels
#Screen dimensions: 47.5 X 29.5 cm
#Distance to screen: 70 cm
#Refresh rate: 120 Hz

xc = 840
yc = 525

x_pixels = 134 # pixels per 3 degrees of visual angle on each side; this is usually the visual angle taken as a ROI
y_pixels = 132

x_ecc = 302  # from experimental script. Equivalent to 6.6 degrees of visual angle for the eccentricity of the stimulus locations
y_ecc = 290


# Fit a linear function to the data	
def linear_func(x, a, c):
	return (a*x) + c	


# all subject variables
all_subjects_srt = [[] for i in range(len(subjects))]
all_subjects_overall_srt =  [[] for i in range(len(subjects))]
all_subjects_amp = [[] for i in range(len(subjects))]
training_all_trials, training_all_trials_combined = [[] for i in range(2)]
training_gains, training_gains_overall, training_gains_block1_mean = [[] for i in range(3)]
outlier_percentage = []
subjects_vigor = []

###### SUBJECT LOOP ##########
for sb in range(len(subjects)):
	
	print subjects[sb]
	
	high_block_srt, low_block_srt, neut1_block_srt, neut2_block_srt, all_block_srt = [np.zeros(nr_blocks) for i in range(5)]
	high_block_amp, low_block_amp, neut1_block_amp, neut2_block_amp = [np.zeros(nr_blocks) for i in range(4)]
	
	#Taking this subject from csv file containing all subjects
	pre_subject_df = saccades_main_df[saccades_main_df['RECORDING_SESSION_LABEL'] == subjects[sb]]

	#Taking only rows that are actual eye movements ('.' stands for slight movements of eye, but not enough to be a proper saccade)
	subject_df = pre_subject_df[np.isfinite(pre_subject_df['CURRENT_SAC_ANGLE'] != '.')] #taking all rows that are actual eye movements

	#Converting strings ('objects' in pandas terminology) that should be numeric to floats
	subject_df = subject_df.convert_objects(convert_numeric=True)
	
	# Taking only first saccade, no blink trials, target found, saccade start times to exclude anticipation errors (<80ms), saccades ending before 500ms timeout, saccades beginning within 3 degrees of fixation only
	pre_exclusion_df = subject_df[(subject_df['CURRENT_SAC_INDEX']== 1)]

	exclusion_df = pre_exclusion_df[(pre_exclusion_df['CURRENT_SAC_CONTAINS_BLINK']==False) & (pre_exclusion_df['sac_latency'] > 80) & (pre_exclusion_df['sac_latency'] < timeout)  & (pre_exclusion_df['PREVIOUS_FIX_BLINK_AROUND'] == 'NONE') & (pre_exclusion_df['CURRENT_SAC_START_X']> (xc-x_pixels)) & (pre_exclusion_df['CURRENT_SAC_START_X'] < (xc+x_pixels)) & (pre_exclusion_df['CURRENT_SAC_START_Y']> (yc-y_pixels)) & (pre_exclusion_df['CURRENT_SAC_START_Y']< (yc+y_pixels))]

	#Create a dict to make new variables target_x, target_y, and target_angle based on the current target position
	exclusion_dict = {}
	
	coords_nw = [xc - (x_ecc/np.sqrt(2)),yc - (y_ecc/np.sqrt(2))] #[626,320]
	coords_ne = [xc + (x_ecc/np.sqrt(2)),yc - (y_ecc/np.sqrt(2))] #[1054,320]
	coords_sw = [xc - (x_ecc/np.sqrt(2)), yc + (y_ecc/np.sqrt(2))] #[626,730]
	coords_se = [xc + (x_ecc/np.sqrt(2)), yc + (y_ecc/np.sqrt(2))] #[1054,320]
	
	nw_x_diff = coords_nw[0] - xc #this is the x distance NE location SHOULD be from fixation. Want to location at landing location variance later, so we need to adjust the exact end point per trial based on where the eye actually started from, i.e. the current saccade start location will not be exactly at the centre.
	nw_y_diff = coords_nw[1] - yc
	ne_x_diff = coords_ne[0] - xc
	ne_y_diff = coords_ne[1] - yc
	sw_x_diff = coords_sw[0] - xc
	sw_y_diff = coords_sw[1] - yc
	se_x_diff = coords_se[0] - xc
	se_y_diff = coords_se[1] - yc


	for i in range(len(exclusion_df)):
		if exclusion_df['target_position'].iloc[i]=='north_west': 
			exclusion_dict[i] = {'target_x':coords_nw[0],'target_y':coords_nw[1],'target_actual_x': exclusion_df['CURRENT_SAC_START_X'].iloc[i] + nw_x_diff, 'target_actual_y': exclusion_df['CURRENT_SAC_START_Y'].iloc[i] + nw_y_diff}
		elif exclusion_df['target_position'].iloc[i]=='north_east':
			exclusion_dict[i] = {'target_x': coords_ne[0], 'target_y' : coords_ne[1],'target_actual_x': exclusion_df['CURRENT_SAC_START_X'].iloc[i] + ne_x_diff,'target_actual_y': exclusion_df['CURRENT_SAC_START_Y'].iloc[i] + ne_y_diff}
		elif exclusion_df['target_position'].iloc[i]=='south_west': 
			exclusion_dict[i] = {'target_x': coords_sw[0], 'target_y' : coords_sw[1], 'target_actual_x': exclusion_df['CURRENT_SAC_START_X'].iloc[i] + sw_x_diff, 'target_actual_y': exclusion_df['CURRENT_SAC_START_Y'].iloc[i] + sw_y_diff}
		elif exclusion_df['target_position'].iloc[i]=='south_east': 
			exclusion_dict[i] = {'target_x': coords_se[0], 'target_y' : coords_se[1], 'target_actual_x': exclusion_df['CURRENT_SAC_START_X'].iloc[i] + se_x_diff,'target_actual_y': exclusion_df['CURRENT_SAC_START_Y'].iloc[i] + se_y_diff}
				
	#Turn this dict into a dataframe
	exclusion_extra = pd.DataFrame.from_dict(exclusion_dict, orient = 'index', dtype = float)
	
	#Need to change the range of original exclusion_df so that it matches the new exclusion_extra dataframe. This is needed to perform the concatenation in correct way.
	exclusion_df.index = range(len(exclusion_df))
	
	#Concatenate
	exclusion_df = pd.concat([exclusion_df, exclusion_extra], axis=1)
	
	target_df = exclusion_df[(exclusion_df['CURRENT_SAC_END_X'] > (exclusion_df['target_x'] - x_pixels)) & (exclusion_df['CURRENT_SAC_END_X'] < (exclusion_df['target_x'] + x_pixels)) & (exclusion_df['CURRENT_SAC_END_Y'] > (exclusion_df['target_y'] - y_pixels)) & (exclusion_df['CURRENT_SAC_END_Y'] < (exclusion_df['target_y']+ y_pixels))]	
	nontarget_df = exclusion_df[(exclusion_df['CURRENT_SAC_END_X'] < (exclusion_df['target_x'] - x_pixels)) | (exclusion_df['CURRENT_SAC_END_X'] > (exclusion_df['target_x'] + x_pixels)) | (exclusion_df['CURRENT_SAC_END_Y'] < (exclusion_df['target_y'] - y_pixels)) | (exclusion_df['CURRENT_SAC_END_Y'] > (exclusion_df['target_y']+ y_pixels))]

	#Getting neutral positions: neut1 = same hemisphere as high, neut2 = same hemisphere as low
	if target_df['high_location'].iloc[0] == 'north_west':
		neut1 = 'south_west'
	elif target_df['high_location'].iloc[0] == 'north_east':
		neut1 = 'south_east'
	elif target_df['high_location'].iloc[0] == 'south_west':
		neut1 = 'north_west'
	elif target_df['high_location'].iloc[0] == 'south_east':
		neut1 = 'north_east'
	
	if target_df['low_location'].iloc[0] == 'north_west':
		neut2 = 'south_west'
	elif target_df['low_location'].iloc[0] == 'north_east':
		neut2 = 'south_east'
	elif target_df['low_location'].iloc[0] == 'south_west':
		neut2 = 'north_west'
	elif target_df['low_location'].iloc[0] == 'south_east':
		neut2 = 'north_east'


	#Exclude SRT outliers
	rt_mean = np.mean(target_df['sac_latency'])
	rt_std = np.std(target_df['sac_latency'])
	outlier_adjust = rt_std*3 #3 SDs
	outlier_plus = rt_mean + outlier_adjust
	outlier_minus = rt_mean - outlier_adjust

	correct_df = target_df[(outlier_minus < (target_df['sac_latency'])) &  ((target_df['sac_latency'])< outlier_plus)]
	
	outlier_percentage.append((target_df.shape[0] - correct_df.shape[0])/float(nr_trials))

	# Condition information for SRTs
	high_df = correct_df[correct_df['high_location']==correct_df['target_position']]
	low_df = correct_df[correct_df['low_location']==correct_df['target_position']]
	rewarded_df = correct_df[(correct_df['high_location']==correct_df['target_position']) | (correct_df['low_location']==correct_df['target_position'])]
	
	neut1_df = correct_df[(correct_df['target_position']== neut1)]
	neut2_df = correct_df[(correct_df['target_position']== neut2)]
	
	print 'high, low, neut1, neut2: ', [high_df.shape[0],low_df.shape[0], neut1_df.shape[0], neut2_df.shape[0]]

	### LEARNING CURVES - fitting across all trials ###
	
	# High
	high_y = high_df['sac_latency']
	x = np.arange(len(high_df))	
	popt_high, pcov_high = curve_fit(linear_func, x, high_y)
	plt.plot(x, linear_func(x, *popt_high), label = 'High Curve', color = 'b')
	
	# Low
	low_y = low_df['sac_latency']
	x = np.arange(len(low_df))	
	popt_low, pcov_low = curve_fit(linear_func, x, low_y)
	plt.plot(x, linear_func(x, *popt_low), label = 'Low Curve',color = 'g')
	
	# Neut1
	neut1_y = neut1_df['sac_latency']
	x = np.arange(len(neut1_df))	
	popt_neut1, pcov_neut1 = curve_fit(linear_func, x, neut1_y)
	plt.plot(x, linear_func(x, *popt_neut1), label = 'Neut1 Curve',color = 'y')	
	
	# Neut1
	neut2_y = neut2_df['sac_latency']
	x = np.arange(len(neut2_df))	
	popt_neut2, pcov_neut2 = curve_fit(linear_func, x, neut2_y)
	plt.plot(x, linear_func(x, *popt_neut2), label = 'Neut2 Curve',color = 'r')
	
	#Overall
	combined_y = correct_df['sac_latency']
	x = np.arange(len(correct_df))	
	popt_combined, pcov_combined = curve_fit(linear_func, x, combined_y)
	# plt.plot(x, linear_func(x, *popt_combined), label = 'Combined Curve')
	
	plt.xlim([0,120])
	plt.ylabel('Saccadic Latency (ms)')
	plt.xlabel('# of trials')
	plt.legend(loc='best')
	os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results/figs/learning_curves')
	plt.savefig(subjects[sb] + '_learning_curves')
	#plt.show()
	plt.close()
	
	# Slope for each condition (and combined) across all trials
	train_high_all_slope = popt_high[0]
	train_low_all_slope = popt_low[0]
	train_neut1_all_slope = popt_neut1[0]		
	train_neut2_all_slope = popt_neut2[0]
	train_combined_all_slope = popt_combined[0]
	
	training_all_trials.append([train_high_all_slope, train_low_all_slope, train_neut1_all_slope, train_neut2_all_slope])
	training_all_trials_combined.append(train_combined_all_slope)
	
	for i in range(0,nr_blocks):
		
		all_block_df = correct_df[correct_df['block_number']==i]		
		high_block_df = high_df[high_df['block_number']==i]
		low_block_df = low_df[low_df['block_number']==i]
		neut1_block_df = neut1_df[neut1_df['block_number']==i]
		neut2_block_df = neut2_df[neut2_df['block_number']==i]

		all_block_srt[i] = np.mean(all_block_df['sac_latency'])
		high_block_srt[i] = np.mean(high_block_df['sac_latency'])
		high_block_amp[i] = np.mean(high_block_df['CURRENT_SAC_AMPLITUDE'])		
		low_block_srt[i] = np.mean(low_block_df['sac_latency'])
		low_block_amp[i] = np.mean(low_block_df['CURRENT_SAC_AMPLITUDE'])		
		neut1_block_srt[i] = np.mean(neut1_block_df['sac_latency'])
		neut1_block_amp[i] = np.mean(neut1_block_df['CURRENT_SAC_AMPLITUDE'])		
		neut2_block_srt[i] = np.mean(neut2_block_df['sac_latency'])
		neut2_block_amp[i] = np.mean(neut2_block_df['CURRENT_SAC_AMPLITUDE'])		
	
	# SRT
	for art in all_block_srt:
		all_subjects_overall_srt[sb].append(art)
	for hrt in high_block_srt:
		all_subjects_srt[sb].append(hrt)
	for lrt in low_block_srt:
		all_subjects_srt[sb].append(lrt)
	for n1rt in neut1_block_srt:
		all_subjects_srt[sb].append(n1rt)
	for n2rt in neut2_block_srt:
		all_subjects_srt[sb].append(n2rt)
	
	train_gain_overall = all_block_srt[0] - all_block_srt[-1]
	train_gain_high = high_block_srt[0] - high_block_srt[-1]
	train_gain_low = low_block_srt[0] - low_block_srt[-1]
	train_gain_neut1 = neut1_block_srt[0] - neut1_block_srt[-1]
	train_gain_neut2 = neut2_block_srt[0] - neut2_block_srt[-1]
	
	training_gains.append([train_gain_high,train_gain_low,train_gain_neut1,train_gain_neut2])
	training_gains_overall.append(train_gain_overall)
	
	# Comparing everything to mean of first block
	train_block1_mean = all_block_srt[0]
	training_gains_compare_block1 = [train_block1_mean-high_block_srt[1],train_block1_mean-high_block_srt[2],train_block1_mean - high_block_srt[3],train_block1_mean - low_block_srt[1],train_block1_mean - low_block_srt[2],train_block1_mean - low_block_srt[3],train_block1_mean - neut1_block_srt[1],train_block1_mean - neut1_block_srt[2],train_block1_mean - neut1_block_srt[3],train_block1_mean - neut2_block_srt[1],train_block1_mean - neut2_block_srt[2],train_block1_mean - neut2_block_srt[3]]
	training_gains_block1_mean.append(training_gains_compare_block1)

	#print 'reward blocks: ', [train_gain_high,train_gain_low,train_gain_neut1,train_gain_neut2]
		
	# Amplitude
	for ha in high_block_amp:
		all_subjects_amp[sb].append(ha)
	for la in low_block_amp:
		all_subjects_amp[sb].append(la)
	for n1a in neut1_block_amp:
		all_subjects_amp[sb].append(n1a)
	for n2a in neut2_block_amp:
		all_subjects_amp[sb].append(n2a)	


	### SACCADE VIGOR : LOCATION PER BLOCK ###
	# Using all trials for a single location for curve fit, then compare expected vigor (from overall fit) with actual vigor at that location per block #

	# Fit a function to the data (Choi et al, 2014; Reppert et al., 2015)	
	# They use hyperbolic in papers, but for smaller saccades this appromixates to a linear curve)
	
	def linear_vel(x, a, b):
		return (a*x) + b

	high_loc = correct_df[(correct_df['high_location']==correct_df['target_position']) & (correct_df['CURRENT_SAC_AMPLITUDE'] > 0) & (correct_df['CURRENT_SAC_PEAK_VELOCITY'] > 0)]
	low_loc = correct_df[(correct_df['low_location']==correct_df['target_position']) & (correct_df['CURRENT_SAC_AMPLITUDE'] > 0) & (correct_df['CURRENT_SAC_PEAK_VELOCITY'] > 0)]
	neut1_loc = correct_df[(neut1==correct_df['target_position']) & (correct_df['CURRENT_SAC_AMPLITUDE'] > 0) & (correct_df['CURRENT_SAC_PEAK_VELOCITY'] > 0)]
	neut2_loc = correct_df[(neut2==correct_df['target_position']) & (correct_df['CURRENT_SAC_AMPLITUDE'] > 0) & (correct_df['CURRENT_SAC_PEAK_VELOCITY'] > 0)]

	high_blocks = [high_loc[high_loc['block_number']==0], high_loc[high_loc['block_number']==1], high_loc[high_loc['block_number']==2], high_loc[high_loc['block_number']==3]]
	low_blocks = [low_loc[low_loc['block_number']==0], low_loc[low_loc['block_number']==1], low_loc[low_loc['block_number']==2], low_loc[low_loc['block_number']==3]]
	neut1_blocks = [neut1_loc[neut1_loc['block_number']==0], neut1_loc[neut1_loc['block_number']==1], neut1_loc[neut1_loc['block_number']==2], neut1_loc[neut1_loc['block_number']==3]]
	neut2_blocks = [neut2_loc[neut2_loc['block_number']==0], neut2_loc[neut2_loc['block_number']==1], neut2_loc[neut2_loc['block_number']==2], neut2_loc[neut2_loc['block_number']==3]]

	high_vigor_mean = [[] for i in range(4)]
	low_vigor_mean = [[] for i in range(4)]
	neut1_vigor_mean = [[] for i in range(4)]
	neut2_vigor_mean = [[] for i in range(4)]

	# HIGH REWARD LOCATION #

	# Get rid of outliers in the velocity vs amplitude profile, so that fit is not skewed
	y_div_x = high_loc['CURRENT_SAC_PEAK_VELOCITY']/high_loc['CURRENT_SAC_AMPLITUDE']
	mean_y_div_x = np.mean(y_div_x)
	std_y_div_x = np.std(y_div_x)
	
	trials_processed = high_loc[((high_loc['CURRENT_SAC_PEAK_VELOCITY']/high_loc['CURRENT_SAC_AMPLITUDE'])<(mean_y_div_x+(2.5*std_y_div_x))) & ((high_loc['CURRENT_SAC_PEAK_VELOCITY']/high_loc['CURRENT_SAC_AMPLITUDE'])>(mean_y_div_x-(2.5*std_y_div_x)))]

	xdata = trials_processed['CURRENT_SAC_AMPLITUDE']
	ydata = trials_processed['CURRENT_SAC_PEAK_VELOCITY']

	#Change x and y data into arrays so curve_fit will work
	x = np.array(xdata, dtype=float)
	y = np.array(ydata, dtype = float)

	fig = plt.figure()
	plt.plot(x,y, 'ro')
	plt.xlabel('Amplitude (deg)')
	plt.ylabel('Peak Velocity (deg/s)')
		
	# Using scipy.optimize optimization (least-squares)
	[params,covariance] = optimization.curve_fit(linear_vel, x, y)
	# This gives parameters for a and b, and a 2x2 covariance matrix

	curve_y = linear_vel(x, params[0],params[1])
	plt.plot(x, curve_y, label = 'High location')
	#plt.show()
	plt.close()

	for bl in range(len(high_blocks)):

		expected_v = []
		for i in range(high_blocks[bl].shape[0]):
			expected_v.append([linear_vel(high_blocks[bl]['CURRENT_SAC_AMPLITUDE'].iloc[i], params[0],params[1]), high_blocks[bl]['trial_number'].iloc[i]]) 

		expected_vel = pd.DataFrame(expected_v, columns = ['expected_vel','trial_number'])
		
		trials = pd.merge(high_blocks[bl], expected_vel, how = 'outer', on= 'trial_number')
		
		# Calculate saccade vigor : ratio peak_vel/expected_peak_vel
		sac_vigor = []

		for i in range(trials.shape[0]):
			sac_vigor.append([trials.iloc[i]['CURRENT_SAC_PEAK_VELOCITY']/trials.iloc[i]['expected_vel'], trials['trial_number'].iloc[i]])

		saccade_vigor = pd.DataFrame(sac_vigor, columns = ['saccade_vigor','trial_number'])
		trials = pd.merge(trials, saccade_vigor, how = 'outer', on= 'trial_number')		
		
		high_vigor_mean[bl]=np.mean(trials['saccade_vigor'])


	# LOW REWARD LOCATION #

	# Get rid of outliers in the velocity vs amplitude profile, so that fit is not skewed
	y_div_x = low_loc['CURRENT_SAC_PEAK_VELOCITY']/low_loc['CURRENT_SAC_AMPLITUDE']
	mean_y_div_x = np.mean(y_div_x)
	std_y_div_x = np.std(y_div_x)
	
	trials_processed = low_loc[((low_loc['CURRENT_SAC_PEAK_VELOCITY']/low_loc['CURRENT_SAC_AMPLITUDE'])<(mean_y_div_x+(2.5*std_y_div_x))) & ((low_loc['CURRENT_SAC_PEAK_VELOCITY']/low_loc['CURRENT_SAC_AMPLITUDE'])>(mean_y_div_x-(2.5*std_y_div_x)))]

	xdata = trials_processed['CURRENT_SAC_AMPLITUDE']
	ydata = trials_processed['CURRENT_SAC_PEAK_VELOCITY']

	#Change x and y data into arrays so curve_fit will work
	x = np.array(xdata, dtype=float)
	y = np.array(ydata, dtype = float)

	fig = plt.figure()
	plt.plot(x,y, 'ro')
	plt.xlabel('Amplitude (deg)')
	plt.ylabel('Peak Velocity (deg/s)')
		
	# Using scipy.optimize optimization (least-squares)
	[params,covariance] = optimization.curve_fit(linear_vel, x, y)
	# This gives parameters for a and b, and a 2x2 covariance matrix

	curve_y = linear_vel(x, params[0],params[1])
	plt.plot(x, curve_y, label = 'Low location')
	#plt.show()
	plt.close()

	for bl in range(len(low_blocks)):

		expected_v = []
		for i in range(low_blocks[bl].shape[0]):
			expected_v.append([linear_vel(low_blocks[bl]['CURRENT_SAC_AMPLITUDE'].iloc[i], params[0],params[1]), low_blocks[bl]['trial_number'].iloc[i]]) 

		expected_vel = pd.DataFrame(expected_v, columns = ['expected_vel','trial_number'])
		
		trials = pd.merge(low_blocks[bl], expected_vel, how = 'outer', on= 'trial_number')
		
		# Calculate saccade vigor : ratio peak_vel/expected_peak_vel
		sac_vigor = []

		for i in range(trials.shape[0]):
			sac_vigor.append([trials.iloc[i]['CURRENT_SAC_PEAK_VELOCITY']/trials.iloc[i]['expected_vel'], trials['trial_number'].iloc[i]])

		saccade_vigor = pd.DataFrame(sac_vigor, columns = ['saccade_vigor','trial_number'])
		trials = pd.merge(trials, saccade_vigor, how = 'outer', on= 'trial_number')		
		
		low_vigor_mean[bl]=np.mean(trials['saccade_vigor'])


	# NEUT1 LOCATION #

	# Get rid of outliers in the velocity vs amplitude profile, so that fit is not skewed
	y_div_x = neut1_loc['CURRENT_SAC_PEAK_VELOCITY']/neut1_loc['CURRENT_SAC_AMPLITUDE']
	mean_y_div_x = np.mean(y_div_x)
	std_y_div_x = np.std(y_div_x)
	
	trials_processed = neut1_loc[((neut1_loc['CURRENT_SAC_PEAK_VELOCITY']/neut1_loc['CURRENT_SAC_AMPLITUDE'])<(mean_y_div_x+(2.5*std_y_div_x))) & ((neut1_loc['CURRENT_SAC_PEAK_VELOCITY']/neut1_loc['CURRENT_SAC_AMPLITUDE'])>(mean_y_div_x-(2.5*std_y_div_x)))]

	xdata = trials_processed['CURRENT_SAC_AMPLITUDE']
	ydata = trials_processed['CURRENT_SAC_PEAK_VELOCITY']

	#Change x and y data into arrays so curve_fit will work
	x = np.array(xdata, dtype=float)
	y = np.array(ydata, dtype = float)

	fig = plt.figure()
	plt.plot(x,y, 'ro')
	plt.xlabel('Amplitude (deg)')
	plt.ylabel('Peak Velocity (deg/s)')
		
	# Using scipy.optimize optimization (least-squares)
	[params,covariance] = optimization.curve_fit(linear_vel, x, y)
	# This gives parameters for a and b, and a 2x2 covariance matrix

	curve_y = linear_vel(x, params[0],params[1])
	plt.plot(x, curve_y, label = 'Neut1 location')
	#plt.show()
	plt.close()

	for bl in range(len(neut1_blocks)):

		expected_v = []
		for i in range(neut1_blocks[bl].shape[0]):
			expected_v.append([linear_vel(neut1_blocks[bl]['CURRENT_SAC_AMPLITUDE'].iloc[i], params[0],params[1]), neut1_blocks[bl]['trial_number'].iloc[i]]) 

		expected_vel = pd.DataFrame(expected_v, columns = ['expected_vel','trial_number'])
		
		trials = pd.merge(neut1_blocks[bl], expected_vel, how = 'outer', on= 'trial_number')
		
		# Calculate saccade vigor : ratio peak_vel/expected_peak_vel
		sac_vigor = []

		for i in range(trials.shape[0]):
			sac_vigor.append([trials.iloc[i]['CURRENT_SAC_PEAK_VELOCITY']/trials.iloc[i]['expected_vel'], trials['trial_number'].iloc[i]])

		saccade_vigor = pd.DataFrame(sac_vigor, columns = ['saccade_vigor','trial_number'])
		trials = pd.merge(trials, saccade_vigor, how = 'outer', on= 'trial_number')		
		
		neut1_vigor_mean[bl]=np.mean(trials['saccade_vigor'])


	# NEUT2 LOCATION #

	# Get rid of outliers in the velocity vs amplitude profile, so that fit is not skewed
	y_div_x = neut2_loc['CURRENT_SAC_PEAK_VELOCITY']/neut2_loc['CURRENT_SAC_AMPLITUDE']
	mean_y_div_x = np.mean(y_div_x)
	std_y_div_x = np.std(y_div_x)
	
	trials_processed = neut2_loc[((neut2_loc['CURRENT_SAC_PEAK_VELOCITY']/neut2_loc['CURRENT_SAC_AMPLITUDE'])<(mean_y_div_x+(2.5*std_y_div_x))) & ((neut2_loc['CURRENT_SAC_PEAK_VELOCITY']/neut2_loc['CURRENT_SAC_AMPLITUDE'])>(mean_y_div_x-(2.5*std_y_div_x)))]

	xdata = trials_processed['CURRENT_SAC_AMPLITUDE']
	ydata = trials_processed['CURRENT_SAC_PEAK_VELOCITY']

	#Change x and y data into arrays so curve_fit will work
	x = np.array(xdata, dtype=float)
	y = np.array(ydata, dtype = float)

	fig = plt.figure()
	plt.plot(x,y, 'ro')
	plt.xlabel('Amplitude (deg)')
	plt.ylabel('Peak Velocity (deg/s)')
		
	# Using scipy.optimize optimization (least-squares)
	[params,covariance] = optimization.curve_fit(linear_vel, x, y)
	# This gives parameters for a and b, and a 2x2 covariance matrix

	curve_y = linear_vel(x, params[0],params[1])
	plt.plot(x, curve_y, label = 'Neut2 location')
	#plt.show()
	plt.close()

	for bl in range(len(neut2_blocks)):

		expected_v = []
		for i in range(neut2_blocks[bl].shape[0]):
			expected_v.append([linear_vel(neut2_blocks[bl]['CURRENT_SAC_AMPLITUDE'].iloc[i], params[0],params[1]), neut2_blocks[bl]['trial_number'].iloc[i]]) 

		expected_vel = pd.DataFrame(expected_v, columns = ['expected_vel','trial_number'])
		
		trials = pd.merge(neut2_blocks[bl], expected_vel, how = 'outer', on= 'trial_number')
		
		# Calculate saccade vigor : ratio peak_vel/expected_peak_vel
		sac_vigor = []

		for i in range(trials.shape[0]):
			sac_vigor.append([trials.iloc[i]['CURRENT_SAC_PEAK_VELOCITY']/trials.iloc[i]['expected_vel'], trials['trial_number'].iloc[i]])

		saccade_vigor = pd.DataFrame(sac_vigor, columns = ['saccade_vigor','trial_number'])
		trials = pd.merge(trials, saccade_vigor, how = 'outer', on= 'trial_number')		
		
		neut2_vigor_mean[bl]=np.mean(trials['saccade_vigor'])
	

	subjects_vigor.append([high_vigor_mean, low_vigor_mean, neut1_vigor_mean, neut2_vigor_mean])

srt_overall_df = pd.DataFrame(all_subjects_overall_srt)
srt_df = pd.DataFrame(all_subjects_srt)
gain_srt_df = pd.DataFrame(training_gains)
amp_df = pd.DataFrame(all_subjects_amp)

os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results')
srt_df.to_csv('training.csv',header = ['High_1','High_2','High_3','High_4','Low_1','Low_2','Low_3','Low_4','No1_1','No1_2','No1_3','No1_4','No2_1','No2_2','No2_3','No2_4'], index = False)
gain_srt_df.to_csv('training_gains.csv',header = ['High','Low','No1','No2'])


# Combining neutral locations into one variable
neutcomb_srt = [srt_df[[8, 12]].mean(axis=1),srt_df[[9, 13]].mean(axis=1),srt_df[[10, 14]].mean(axis=1),srt_df[[11, 15]].mean(axis=1)]
neutcomb_srt = pd.DataFrame(neutcomb_srt)
neutcomb_srt = neutcomb_srt.transpose()

# Mean SRTs per block
overall_srt_mean = srt_overall_df.mean(0)
high_srt_mean = srt_df.mean(0)[:nr_blocks]
low_srt_mean = srt_df.mean(0)[nr_blocks:2*nr_blocks]
low_srt_mean.reset_index(drop=True, inplace = True) #resets the index so that it isn't from 4-7 anymore
neut1_srt_mean = srt_df.mean(0)[2*nr_blocks:3*nr_blocks]
neut1_srt_mean.reset_index(drop=True, inplace = True) 
neut2_srt_mean = srt_df.mean(0)[3*nr_blocks:4*nr_blocks]
neut2_srt_mean.reset_index(drop=True, inplace = True) 
neutcomb_srt_mean = neutcomb_srt.mean()

# SEM of SRTs per block
overall_srt_sem = srt_overall_df.sem(0)
high_srt_sem = srt_df.sem(0)[:nr_blocks]
low_srt_sem = srt_df.sem(0)[nr_blocks:2*nr_blocks]
neut1_srt_sem = srt_df.sem(0)[2*nr_blocks:3*nr_blocks]
neut2_srt_sem = srt_df.sem(0)[3*nr_blocks:4*nr_blocks]
neutcomb_srt_sem = neutcomb_srt.sem()

# New columns for mean of each condition across reward condition (per participant)

srt_df.columns=['High_1','High_2','High_3','High_4','Low_1','Low_2','Low_3','Low_4','No1_1','No1_2','No1_3','No1_4','No2_1','No2_2','No2_3','No2_4']

srt_df['Nocomb_1'] = srt_df[['No1_1','No2_1']].mean(axis=1)
srt_df['Nocomb_2'] = srt_df[['No1_2','No2_2']].mean(axis=1)
srt_df['Nocomb_3'] = srt_df[['No1_3','No2_3']].mean(axis=1)
srt_df['Nocomb_4'] = srt_df[['No1_4','No2_4']].mean(axis=1)

srt_df['block1_mean'] = srt_df[['High_1','Low_1','No1_1','No2_1']].mean(axis=1)

srt_df['mean_high']=srt_df[['High_2','High_3','High_4']].mean(axis=1)
srt_df['mean_low']=srt_df[['Low_2','Low_3','Low_4']].mean(axis=1)
srt_df['mean_nocomb']=srt_df[['Nocomb_2','Nocomb_3','Nocomb_4']].mean(axis=1)

allsubs_mean_high = np.mean(srt_df['mean_high'])
allsubs_mean_low = np.mean(srt_df['mean_low'])
allsubs_mean_nocomb = np.mean(srt_df['mean_nocomb'])

# STD calculations
std_df = srt_df.copy(deep=False)
std_df['High_2']=srt_df['High_2']-srt_df['mean_high'] + allsubs_mean_high
std_df['High_3']=srt_df['High_3']-srt_df['mean_high'] + allsubs_mean_high
std_df['High_4']=srt_df['High_4']-srt_df['mean_high'] + allsubs_mean_high

std_df['Low_2']=srt_df['Low_2']-srt_df['mean_low'] + allsubs_mean_low
std_df['Low_3']=srt_df['Low_3']-srt_df['mean_low'] + allsubs_mean_low
std_df['Low_4']=srt_df['Low_4']-srt_df['mean_low'] + allsubs_mean_low

std_df['Nocomb_2']=srt_df['Nocomb_2']-srt_df['mean_nocomb'] + allsubs_mean_nocomb
std_df['Nocomb_3']=srt_df['Nocomb_3']-srt_df['mean_nocomb'] + allsubs_mean_nocomb
std_df['Nocomb_4']=srt_df['Nocomb_4']-srt_df['mean_nocomb'] + allsubs_mean_nocomb


# Reward data across blocks
high_list = [srt_df['High_2'].mean(),srt_df['High_3'].mean(),srt_df['High_4'].mean()]
low_list = [srt_df['Low_2'].mean(),srt_df['Low_3'].mean(),srt_df['Low_4'].mean()]
nocomb_list = [srt_df['Nocomb_2'].mean(),srt_df['Nocomb_3'].mean(),srt_df['Nocomb_4'].mean()]

# Errorbars : improvements for reward blocks : (high_2, high_3, high_4, low_1, low_2, low_3, nocomb_1, nocomb_2, nocomb_3)
std_list = [std_df['High_2'].std()/np.sqrt(len(subjects)), std_df['High_3'].std()/np.sqrt(len(subjects)),std_df['High_4'].std()/np.sqrt(len(subjects)),std_df['Low_2'].std()/np.sqrt(len(subjects)),std_df['Low_3'].std()/np.sqrt(len(subjects)),std_df['Low_4'].std()/np.sqrt(len(subjects)),std_df['Nocomb_2'].std()/np.sqrt(len(subjects)),std_df['Nocomb_3'].std()/np.sqrt(len(subjects)),std_df['Nocomb_4'].std()/np.sqrt(len(subjects))]


# Plot SRTs for reward blocks
fig = plt.figure()
ax = fig.add_subplot(111)
x=np.arange(1,nr_blocks,1)
plt.xlabel('Reward block', fontsize = '12')
plt.ylabel('Saccade Latency (ms)', fontsize = '12')
ax.yaxis.set_label_coords(-0.1, 0.5) # -.1 on x-axis, .5 = halfway along y-axis
ax.xaxis.set_label_coords(0.5, -0.09) 
sem_div = np.sqrt(len(subjects))
plt.errorbar(x,high_list, yerr = std_list[:3], linestyle = '-',capsize=10, color = 'k', label = 'High reward') 
plt.errorbar(x,low_list, yerr = std_list[3:6], linestyle = '--',capsize=10, color = 'k', label = 'Low reward') 
plt.errorbar(x,nocomb_list, yerr = std_list[6:9], linestyle = '-.',capsize=10, color = 'k', label = 'No reward')
plt.legend(loc='best')
plt.xlim(xmin=0.5, xmax = 3.5)
plt.xticks([1,2,3])
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results/figs')
plt.savefig('Training_SRT.pdf')
plt.savefig('Training_SRT.png')
plt.show()
plt.close()


### IMPROVEMENTS ###
# Get average of all trials in block 1 (baseline). Calculate improvement from this baseline for reward blocks.

improvements_df = srt_df[['High_2','High_3','High_4','Low_2','Low_3','Low_4','Nocomb_2','Nocomb_3','Nocomb_4']]

# Recalculating columns
improvements_df['High_2']=srt_df['block1_mean']-srt_df['High_2']
improvements_df['High_3']=srt_df['block1_mean']-srt_df['High_3']
improvements_df['High_4']=srt_df['block1_mean']-srt_df['High_4']
improvements_df['Low_2']=srt_df['block1_mean']-srt_df['Low_2']
improvements_df['Low_3']=srt_df['block1_mean']-srt_df['Low_3']
improvements_df['Low_4']=srt_df['block1_mean']-srt_df['Low_4']
improvements_df['Nocomb_2']=srt_df['block1_mean']-srt_df['Nocomb_2']
improvements_df['Nocomb_3']=srt_df['block1_mean']-srt_df['Nocomb_3']
improvements_df['Nocomb_4']=srt_df['block1_mean']-srt_df['Nocomb_4']

# New columns for mean of each condition across reward condition (per participant)
improvements_df['mean_improve_high']=improvements_df[['High_2','High_3','High_4']].mean(axis=1)
improvements_df['mean_improve_low']=improvements_df[['Low_2','Low_3','Low_4']].mean(axis=1)
improvements_df['mean_improve_nocomb']=improvements_df[['Nocomb_2','Nocomb_3','Nocomb_4']].mean(axis=1)

allsubs_mean_high = np.mean(improvements_df['mean_improve_high'])
allsubs_mean_low = np.mean(improvements_df['mean_improve_low'])
allsubs_mean_nocomb = np.mean(improvements_df['mean_improve_nocomb'])

# STD calculations
std_calc_df = improvements_df.copy(deep=False)
std_calc_df['High_2']=improvements_df['High_2']-improvements_df['mean_improve_high'] + allsubs_mean_high
std_calc_df['High_3']=improvements_df['High_3']-improvements_df['mean_improve_high'] + allsubs_mean_high
std_calc_df['High_4']=improvements_df['High_4']-improvements_df['mean_improve_high'] + allsubs_mean_high

std_calc_df['Low_2']=improvements_df['Low_2']-improvements_df['mean_improve_low'] + allsubs_mean_low
std_calc_df['Low_3']=improvements_df['Low_3']-improvements_df['mean_improve_low'] + allsubs_mean_low
std_calc_df['Low_4']=improvements_df['Low_4']-improvements_df['mean_improve_low'] + allsubs_mean_low

std_calc_df['Nocomb_2']=improvements_df['Nocomb_2']-improvements_df['mean_improve_nocomb'] + allsubs_mean_nocomb
std_calc_df['Nocomb_3']=improvements_df['Nocomb_3']-improvements_df['mean_improve_nocomb'] + allsubs_mean_nocomb
std_calc_df['Nocomb_4']=improvements_df['Nocomb_4']-improvements_df['mean_improve_nocomb'] + allsubs_mean_nocomb

# Reward data across blocks
high_list = [improvements_df['High_2'].mean(),improvements_df['High_3'].mean(),improvements_df['High_4'].mean()]
low_list = [improvements_df['Low_2'].mean(),improvements_df['Low_3'].mean(),improvements_df['Low_4'].mean()]
nocomb_list = [improvements_df['Nocomb_2'].mean(),improvements_df['Nocomb_3'].mean(),improvements_df['Nocomb_4'].mean()]

# Errorbars : improvements for reward blocks : (high_2, high_3, high_4, low_1, low_2, low_3, nocomb_1, nocomb_2, nocomb_3)
stdev_list = [std_calc_df['High_2'].std()/np.sqrt(len(subjects)), std_calc_df['High_3'].std()/np.sqrt(len(subjects)),std_calc_df['High_4'].std()/np.sqrt(len(subjects)),std_calc_df['Low_2'].std()/np.sqrt(len(subjects)),std_calc_df['Low_3'].std()/np.sqrt(len(subjects)),std_calc_df['Low_4'].std()/np.sqrt(len(subjects)),std_calc_df['Nocomb_2'].std()/np.sqrt(len(subjects)),std_calc_df['Nocomb_3'].std()/np.sqrt(len(subjects)),std_calc_df['Nocomb_4'].std()/np.sqrt(len(subjects))]

# Plot Improvements
fig = plt.figure()
ax = fig.add_subplot(111)
x=np.arange(1,nr_blocks,1) # 3 reward blocks
plt.xlabel('Reward block', fontsize = '12')
plt.ylabel('Saccade latency improvement (ms)', fontsize = '12')
ax.yaxis.set_label_coords(-0.1, 0.5) # -.1 on x-axis, .5 = halfway along y-axis
ax.xaxis.set_label_coords(0.5, -0.09) 
plt.errorbar(x,high_list, yerr = stdev_list[:3], linestyle = '-', capsize=10, color = 'k', label = 'High reward') 
plt.errorbar(x,low_list, yerr = stdev_list[3:6], linestyle = '--', capsize=10, color = 'k', label = 'Low reward')  
plt.errorbar(x,nocomb_list, yerr = stdev_list[6:9], linestyle = '-.', capsize=10, color = 'k', label = 'No reward')
plt.legend(loc='best')
plt.xlim(xmin=0.5, xmax = 3.5)
plt.xticks([1,2,3])
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/results/figs')
plt.savefig('Training_SRT_Improvements.pdf')
plt.savefig('Training_SRT_Improvements.png')
plt.show()
plt.close()


shell()


