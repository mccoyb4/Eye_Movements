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
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_2/analysis/fixation_reports')
saccades_main_df = pd.read_csv('SRC_TE.csv') 
# Can use variable RECORDING_SESSION_LABEL to get the subject name, e.g. 'S01tr1' 

all_or_diff = 'diff'

# In this experiment, reward is always in different hemifields	
if all_or_diff == 'diff':
#Rewards in different hemifields:
	subjects_test1 = ['SRC01TE1','SRC02TE1','SRC03TE1','SRC04TE1','SRC05TE1','SRC06TE1','SRC07TE1','SRC08TE1','SRC09TE1','SRC10TE1','SRC11TE1','SRC12TE1','SRC13TE1','SRC14TE1','SRC15TE1','SRC16TE1','SRC17TE1','SRC18TE1','SRC19TE1','SRC20TE1','SRC21TE1','SRC22TE1','SRC23TE1','SRC24TE1']
	subjects_test2 = ['SRC01TE2','SRC02TE2','SRC03TE2','SRC04TE2','SRC05TE2','SRC06TE2','SRC07TE2','SRC08TE2','SRC09TE2','SRC10TE2','SRC11TE2','SRC12TE2','SRC13TE2','SRC14TE2','SRC15TE2','SRC16TE2','SRC17TE2','SRC18TE2','SRC19TE2','SRC20TE2','SRC21TE2','SRC22TE2','SRC23TE2','SRC24TE2']

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
	# not_fixated_df = post_blinks_df[(post_blinks_df['CURRENT_FIX_X'] < (xc-x_pixels)) | (post_blinks_df['CURRENT_FIX_X'] > (xc+x_pixels)) | (post_blinks_df['CURRENT_FIX_Y'] < (yc-y_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] > (yc+y_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] < (xc-x_pixels)) | (post_blinks_df['NEXT_SAC_END_X'] > (xc+x_pixels)) | (post_blinks_df['NEXT_SAC_END_Y']< (yc-y_pixels)) | (post_blinks_df['NEXT_SAC_END_Y']> (yc+y_pixels))]
	# not_fixated_percentage.append((not_fixated_df.shape[0]/nr_trials)*100)
	# post_not_fixated_df = post_blinks_df[(post_blinks_df['CURRENT_FIX_X'] > (xc-x_pixels)) & (post_blinks_df['CURRENT_FIX_X'] < (xc+x_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] > (yc-y_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] < (yc+y_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] > (xc-x_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] < (xc+x_pixels)) & (post_blinks_df['NEXT_SAC_END_Y'] > (yc-y_pixels)) & (post_blinks_df['NEXT_SAC_END_Y'] < (yc+y_pixels))]
		
	# Target incorrect
	incorrect_df = post_blinks_df[post_blinks_df['target_correct'] == 0]
	incorrect_percentage.append((incorrect_df.shape[0]/post_blinks_df.shape[0])*100)
	
	# Target correct
	correct_df = post_blinks_df[post_blinks_df['target_correct'] == 1]
	correct_percentage.append((correct_df.shape[0]/post_blinks_df.shape[0])*100)
	
early_mean_1 = np.mean(early_percentage)
late_mean_1 = np.mean(late_percentage)
blink_mean_1 = np.mean(blink_percentage)
# not_fixated_mean_1 = np.mean(not_fixated_percentage)
incorrect_mean_1 = np.mean(incorrect_percentage)
correct_mean_1 = np.mean(correct_percentage)

early_std_1 = np.std(early_percentage)
late_std_1 = np.std(late_percentage)
blink_std_1 = np.std(blink_percentage)
# not_fixated_std_1 = np.std(not_fixated_percentage)
incorrect_std_1 = np.std(incorrect_percentage)
correct_std_1 = np.std(correct_percentage)


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
	# not_fixated_df = post_blinks_df[(post_blinks_df['CURRENT_FIX_X'] < (xc-x_pixels)) | (post_blinks_df['CURRENT_FIX_X'] > (xc+x_pixels)) | (post_blinks_df['CURRENT_FIX_Y'] < (yc-y_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] > (yc+y_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] < (xc-x_pixels)) | (post_blinks_df['NEXT_SAC_END_X'] > (xc+x_pixels)) | (post_blinks_df['NEXT_SAC_END_Y']< (yc-y_pixels)) | (post_blinks_df['NEXT_SAC_END_Y']> (yc+y_pixels))]
	# not_fixated_percentage.append((not_fixated_df.shape[0]/nr_trials)*100)
	# post_not_fixated_df = post_blinks_df[(post_blinks_df['CURRENT_FIX_X'] > (xc-x_pixels)) & (post_blinks_df['CURRENT_FIX_X'] < (xc+x_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] > (yc-y_pixels)) & (post_blinks_df['CURRENT_FIX_Y'] < (yc+y_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] > (xc-x_pixels)) & (post_blinks_df['NEXT_SAC_END_X'] < (xc+x_pixels)) & (post_blinks_df['NEXT_SAC_END_Y'] > (yc-y_pixels)) & (post_blinks_df['NEXT_SAC_END_Y'] < (yc+y_pixels))]
		
	# Target incorrect
	incorrect_df = post_blinks_df[post_blinks_df['target_correct'] == 0]
	incorrect_percentage.append((incorrect_df.shape[0]/post_blinks_df.shape[0])*100)
	
	# Target correct
	correct_df = post_blinks_df[post_blinks_df['target_correct'] == 1]
	correct_percentage.append((correct_df.shape[0]/post_blinks_df.shape[0])*100)
	
early_mean_2 = np.mean(early_percentage)
late_mean_2 = np.mean(late_percentage)
blink_mean_2 = np.mean(blink_percentage)
# not_fixated_mean_2 = np.mean(not_fixated_percentage)
incorrect_mean_2 = np.mean(incorrect_percentage)
correct_mean_2 = np.mean(correct_percentage)

early_std_2 = np.std(early_percentage)
late_std_2 = np.std(late_percentage)
blink_std_2 = np.std(blink_percentage)
# not_fixated_std_2 = np.std(not_fixated_percentage)
incorrect_std_2 = np.std(incorrect_percentage)
correct_std_2 = np.std(correct_percentage)
	
shell()
