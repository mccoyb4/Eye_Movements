from __future__ import division

import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from scipy.optimize import curve_fit
import sympy as sym
from IPython import embed as shell

# Read in Eye-tracker Saccade Report file as a data frame -> ALL participants
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/analysis/fixation_reports')
saccades_main_df = pd.read_csv('SR01-24_TE.csv') 
# Can use variable RECORDING_SESSION_LABEL to get the subject name, e.g. 'S01tr1' 

all_or_diff = 'diff'

# In this experiment, reward is always in different hemifields	
if all_or_diff == 'diff':
#Rewards in different hemifields:
	subjects_test1 = ['SR01TE1','SR02TE1','SR03TE1','SR04TE1','SR05TE1','SR06TE1','SR07TE1','SR08TE1','SR09TE1','SR10TE1','SR11TE1','SR12TE1','SR13TE1','SR14TE1','SR15TE1','SR16TE1','SR17TE1','SR18TE1','SR19TE1','SR20TE1','SR21TE1','SR22TE1','SR23TE1','SR24TE1']
	subjects_test2 = ['SR01TE2','SR02TE2','SR03TE2','SR04TE2','SR05TE2','SR06TE2','SR07TE2','SR08TE2','SR09TE2','SR10TE2','SR11TE2','SR12TE2','SR13TE2','SR14TE2','SR15TE2','SR16TE2','SR17TE2','SR18TE2','SR19TE2','SR20TE2','SR21TE2','SR22TE2','SR23TE2','SR24TE2']

timeout = 1000
nr_trials = 140

#Carried out in K1F46
#Screen resolution: 1680 x 1050 pixels
#Screen dimensions: 47.5 X 29.5 cm
#Distance to screen: 70 cm
#Refresh rate: 120 Hz

xc = 840
yc = 525

x_pixels = 136 # pixels per 3 degrees of visual angle on each side; this is usually the visual angle taken as a ROI 
y_pixels = 134 


### PRE-TEST

early_percentage, late_percentage, blink_percentage, not_fixated_percentage, incorrect_percentage, correct_percentage = [[] for i in range(6)]

for sb in range(len(subjects_test1)):
	print subjects_test1[sb]

	subject_df = saccades_main_df[saccades_main_df['RECORDING_SESSION_LABEL'] == subjects_test1[sb]]

	subject_df = subject_df.convert_objects(convert_numeric=True)
	
	# Early response (< 200ms)
	early_df = subject_df[subject_df['response_time'] < 200]
	early_percentage.append((early_df.shape[0]/nr_trials)*100)
	post_early_df = subject_df[subject_df['response_time'] > 200]
	
	# Late response (>1000 ms)
	late_df = post_early_df[post_early_df['response_time'] > timeout]
	late_percentage.append((late_df.shape[0]/nr_trials)*100)
	post_late_df = post_early_df[post_early_df['response_time'] < timeout]
	
	# Blinks
	blinks_df = post_late_df[post_late_df['CURRENT_FIX_BLINK_AROUND'] != 'NONE']
	blink_percentage.append((blinks_df.shape[0]/nr_trials)*100)
	post_blinks_df = post_late_df[post_late_df['CURRENT_FIX_BLINK_AROUND'] == 'NONE']
	
	# Not fixated 
	not_fixated_df = post_blinks_df[(post_blinks_df['CURRENT_FIX_X'] < (xc-x_pixels)) | (post_blinks_df['CURRENT_FIX_X'] > (xc+x_pixels)) | (post_blinks_df['CURRENT_FIX_Y'] < (yc-y_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] > (yc+y_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] < (xc-x_pixels)) | (post_blinks_df['NEXT_SAC_END_X'] > (xc+x_pixels)) | (post_blinks_df['NEXT_SAC_END_Y']< (yc-y_pixels)) | (post_blinks_df['NEXT_SAC_END_Y']> (yc+y_pixels))]
	not_fixated_percentage.append((not_fixated_df.shape[0]/nr_trials)*100)
	post_not_fixated_df = post_blinks_df[(post_blinks_df['CURRENT_FIX_X'] > (xc-x_pixels)) & (post_blinks_df['CURRENT_FIX_X'] < (xc+x_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] > (yc-y_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] < (yc+y_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] > (xc-x_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] < (xc+x_pixels)) & (post_blinks_df['NEXT_SAC_END_Y'] > (yc-y_pixels)) & (post_blinks_df['NEXT_SAC_END_Y'] < (yc+y_pixels))]
		
	# Target incorrect
	incorrect_df = post_not_fixated_df[post_not_fixated_df['target_correct'] == 0]
	incorrect_percentage.append((incorrect_df.shape[0]/post_not_fixated_df.shape[0])*100)
	
	correct_df = post_not_fixated_df[post_not_fixated_df['target_correct'] == 1]
	correct_percentage.append((correct_df.shape[0]/post_not_fixated_df.shape[0])*100)
	
early_mean_1 = np.mean(early_percentage)
late_mean_1 = np.mean(late_percentage)
blink_mean_1 = np.mean(blink_percentage)
not_fixated_mean_1 = np.mean(not_fixated_percentage)
incorrect_mean_1 = np.mean(incorrect_percentage)
correct_mean_1 = np.mean(correct_percentage)


### POST-TEST

early_percentage, late_percentage, blink_percentage, not_fixated_percentage, incorrect_percentage, correct_percentage = [[] for i in range(6)]

for sb in range(len(subjects_test2)):
	print subjects_test2[sb]

	subject_df = saccades_main_df[saccades_main_df['RECORDING_SESSION_LABEL'] == subjects_test2[sb]]

	subject_df = subject_df.convert_objects(convert_numeric=True)
	
	# Early response (< 200ms)
	early_df = subject_df[subject_df['response_time'] < 200]
	early_percentage.append((early_df.shape[0]/nr_trials)*100)
	post_early_df = subject_df[subject_df['response_time'] > 200]
	
	# Late response (>1000 ms)
	late_df = post_early_df[post_early_df['response_time'] > timeout]
	late_percentage.append((late_df.shape[0]/nr_trials)*100)
	post_late_df = post_early_df[post_early_df['response_time'] < timeout]
	
	# Blinks
	blinks_df = post_late_df[post_late_df['CURRENT_FIX_BLINK_AROUND'] != 'NONE']
	blink_percentage.append((blinks_df.shape[0]/nr_trials)*100)
	post_blinks_df = post_late_df[post_late_df['CURRENT_FIX_BLINK_AROUND'] == 'NONE']
	
	# Not fixated 
	not_fixated_df = post_blinks_df[(post_blinks_df['CURRENT_FIX_X'] < (xc-x_pixels)) | (post_blinks_df['CURRENT_FIX_X'] > (xc+x_pixels)) | (post_blinks_df['CURRENT_FIX_Y'] < (yc-y_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] > (yc+y_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] < (xc-x_pixels)) | (post_blinks_df['NEXT_SAC_END_X'] > (xc+x_pixels)) | (post_blinks_df['NEXT_SAC_END_Y']< (yc-y_pixels)) | (post_blinks_df['NEXT_SAC_END_Y']> (yc+y_pixels))]
	not_fixated_percentage.append((not_fixated_df.shape[0]/nr_trials)*100)
	post_not_fixated_df = post_blinks_df[(post_blinks_df['CURRENT_FIX_X'] > (xc-x_pixels)) & (post_blinks_df['CURRENT_FIX_X'] < (xc+x_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] > (yc-y_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] < (yc+y_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] > (xc-x_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] < (xc+x_pixels)) & (post_blinks_df['NEXT_SAC_END_Y'] > (yc-y_pixels)) & (post_blinks_df['NEXT_SAC_END_Y'] < (yc+y_pixels))]
		
	# Target incorrect
	incorrect_df = post_not_fixated_df[post_not_fixated_df['target_correct'] == 0]
	incorrect_percentage.append((incorrect_df.shape[0]/post_not_fixated_df.shape[0])*100)
	
	correct_df = post_not_fixated_df[post_not_fixated_df['target_correct'] == 1]
	correct_percentage.append((correct_df.shape[0]/post_not_fixated_df.shape[0])*100)
	
early_mean_2 = np.mean(early_percentage)
late_mean_2 = np.mean(late_percentage)
blink_mean_2 = np.mean(blink_percentage)
not_fixated_mean_2 = np.mean(not_fixated_percentage)
incorrect_mean_2 = np.mean(incorrect_percentage)
correct_mean_2 = np.mean(correct_percentage)
	
shell()
