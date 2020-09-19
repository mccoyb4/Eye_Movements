from __future__ import division

import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from scipy.optimize import curve_fit
from IPython import embed as shell

#All subjects
subjects = ['SRC01TR','SRC02TR','SRC03TR','SRC04TR','SRC05TR','SRC06TR','SRC07TR','SRC08TR','SRC09TR','SRC10TR','SRC11TR','SRC12TR','SRC13TR','SRC14TR','SRC15TR','SRC16TR','SRC17TR','SRC18TR','SRC19TR','SRC20TR','SRC21TR','SRC22TR','SRC23TR','SRC24TR']

# Read in Eye-tracker Saccade Report file as a data frame -> ALL participants
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_2/analysis/saccade_reports')
saccades_main_df = pd.read_csv('SRC_TR.csv')
# Can use variable RECORDING_SESSION_LABEL to get the subject name, e.g. 'S01tr1' 

#Creating single output files that will contain analysis for all subjects
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_2/results')

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

x_pixels = 136 # pixels per 3 degrees of visual angle on each side; this is usually the visual angle taken as a ROI
y_pixels = 134

x_ecc = 403 # from experimental script. Equivalent to 9 degrees of visual angle for the eccentricity of the stimulus locations
y_ecc = 391

early_percentage, late_percentage, blink_percentage, not_fixated_percentage, nontarget_percentage, target_percentage, ques_trials = [[] for i in range(7)]

###### SUBJECT LOOP ##########
for sb in range(len(subjects)):
	print subjects[sb]
	
	#Taking this subject from csv file containing all subjects
	pre_subject_df = saccades_main_df[saccades_main_df['RECORDING_SESSION_LABEL'] == subjects[sb]]

	#Taking only rows that are actual eye movements ('.' stands for slight movements of eye, but not enough to be a proper saccade)
	subject_df = pre_subject_df[np.isfinite(pre_subject_df['CURRENT_SAC_ANGLE'] != '.')] #taking all rows that are actual eye movements

	#Converting strings ('objects' in pandas terminology) that should be numeric to floats
	subject_df = subject_df.convert_objects(convert_numeric=True)
	
	# First saccade for each trial
	pre_exclusion_df = subject_df[(subject_df['CURRENT_SAC_INDEX']== 1)]

	# Early saccades (< 80ms)
	early_df = pre_exclusion_df[pre_exclusion_df['sac_latency'] < 80]
	early_percentage.append((early_df.shape[0]/nr_trials)*100)
	post_early_df = pre_exclusion_df[pre_exclusion_df['sac_latency'] > 80]
	
	# Late saccads (>500 ms)
	late_df = post_early_df[post_early_df['sac_latency'] > timeout]
	late_percentage.append((late_df.shape[0]/nr_trials)*100)
	post_late_df = post_early_df[post_early_df['sac_latency'] < timeout]
	
	# Blinks
	blinks_df = post_late_df[(post_late_df['CURRENT_SAC_CONTAINS_BLINK']==True) | (post_late_df['PREVIOUS_FIX_BLINK_AROUND'] != 'NONE')]
	blink_percentage.append((blinks_df.shape[0]/nr_trials)*100)
	post_blinks_df = post_late_df[(post_late_df['CURRENT_SAC_CONTAINS_BLINK']==False) & (post_late_df['PREVIOUS_FIX_BLINK_AROUND'] == 'NONE')]
		
	# Not fixated at start of trial
	not_fixated_df = post_blinks_df[(post_blinks_df['CURRENT_SAC_START_X'] < (xc-x_pixels)) | (post_blinks_df['CURRENT_SAC_START_X'] > (xc+x_pixels)) | (post_blinks_df['CURRENT_SAC_START_Y']< (yc-y_pixels)) | (post_blinks_df['CURRENT_SAC_START_Y']> (yc+y_pixels))]
	not_fixated_percentage.append((not_fixated_df.shape[0]/nr_trials)*100)
	post_not_fixated_df = post_blinks_df[(post_blinks_df['CURRENT_SAC_START_X'] > (xc-x_pixels)) & (post_blinks_df['CURRENT_SAC_START_X'] < (xc+x_pixels)) & (post_blinks_df['CURRENT_SAC_START_Y'] > (yc-y_pixels)) & (post_blinks_df['CURRENT_SAC_START_Y'] < (yc+y_pixels))]
	
	### Working out target information ###
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

	for i in range(len(post_not_fixated_df)):
		if post_not_fixated_df['target_position'].iloc[i]=='north_west':
			exclusion_dict[i] = {'target_x':coords_nw[0],'target_y':coords_nw[1],'target_actual_x': post_not_fixated_df['CURRENT_SAC_START_X'].iloc[i] + nw_x_diff, 'target_actual_y': post_not_fixated_df['CURRENT_SAC_START_Y'].iloc[i] + nw_y_diff}
		elif post_not_fixated_df['target_position'].iloc[i]=='north_east': 
			exclusion_dict[i] = {'target_x': coords_ne[0], 'target_y' : coords_ne[1], 'target_actual_x': post_not_fixated_df['CURRENT_SAC_START_X'].iloc[i] + ne_x_diff,'target_actual_y': post_not_fixated_df['CURRENT_SAC_START_Y'].iloc[i] + ne_y_diff}
		elif post_not_fixated_df['target_position'].iloc[i]=='south_west':
			exclusion_dict[i] = {'target_x': coords_sw[0], 'target_y' : coords_sw[1], 'target_actual_x': post_not_fixated_df['CURRENT_SAC_START_X'].iloc[i] + sw_x_diff, 'target_actual_y': post_not_fixated_df['CURRENT_SAC_START_Y'].iloc[i] + sw_y_diff}
		elif post_not_fixated_df['target_position'].iloc[i]=='south_east': 
			exclusion_dict[i] = {'target_x': coords_se[0], 'target_y' : coords_se[1], 'target_actual_x': post_not_fixated_df['CURRENT_SAC_START_X'].iloc[i] + se_x_diff,'target_actual_y': post_not_fixated_df['CURRENT_SAC_START_Y'].iloc[i] + se_y_diff}
				
	#Turn this dict into a dataframe
	exclusion_extra = pd.DataFrame.from_dict(exclusion_dict, orient = 'index', dtype = float)
	
	#Need to change the range of original exclusion_df so that it matches the new exclusion_extra dataframe. This is needed to perform the concatenation in correct way.
	post_not_fixated_df.index = range(len(post_not_fixated_df))
	#Concatenate
	post_not_fixated_df = pd.concat([post_not_fixated_df, exclusion_extra], axis=1)
	
	# Saccades not to the target ROI (this is as a percentage 'OF THE REMAINING TRIALS')
	nontarget_df = post_not_fixated_df[(post_not_fixated_df['CURRENT_SAC_END_X'] < (post_not_fixated_df['target_x'] - x_pixels)) | (post_not_fixated_df['CURRENT_SAC_END_X'] > (post_not_fixated_df['target_x'] + x_pixels)) | (post_not_fixated_df['CURRENT_SAC_END_Y'] < (post_not_fixated_df['target_y'] - y_pixels)) | (post_not_fixated_df['CURRENT_SAC_END_Y'] > (post_not_fixated_df['target_y']+ y_pixels))]
	nontarget_percentage.append((nontarget_df.shape[0]/post_not_fixated_df.shape[0])*100)
	target_df = post_not_fixated_df[(post_not_fixated_df['CURRENT_SAC_END_X'] > (post_not_fixated_df['target_x'] - x_pixels)) & (post_not_fixated_df['CURRENT_SAC_END_X'] < (post_not_fixated_df['target_x'] + x_pixels)) & (post_not_fixated_df['CURRENT_SAC_END_Y'] > (post_not_fixated_df['target_y'] - y_pixels)) & (post_not_fixated_df['CURRENT_SAC_END_Y'] < (post_not_fixated_df['target_y']+ y_pixels))]
	target_percentage.append((target_df.shape[0]/post_not_fixated_df.shape[0])*100)
	
early_mean = np.mean(early_percentage)	
late_mean = np.mean(late_percentage)
blink_mean = np.mean(blink_percentage)
not_fixated_mean = np.mean(not_fixated_percentage)
nontarget_mean = np.mean(nontarget_percentage)
target_mean = np.mean(target_percentage)
shell()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	