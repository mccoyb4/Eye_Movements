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
subjects = ['LP01','LP02','LP03', 'LP04','LP05','LP06','LP07','LP08','LP09','LP10','LP11','LP12','LP13','LP14','LP15','LP16','LP17','LP18','LP19','LP20','LP21','LP22','LP23','LP24']
subject_files = ['LP01.csv','LP02.csv','LP03.csv','LP04.csv','LP05.csv','LP06.csv','LP07.csv','LP08.csv','LP09.csv','LP10.csv','LP11.csv','LP12.csv','LP13.csv','LP14.csv','LP15.csv','LP16.csv','LP17.csv','LP18.csv','LP19.csv','LP20.csv','LP21.csv','LP22.csv','LP23.csv','LP24.csv']

os.chdir('/Users/bronagh/Documents/LePelley/LePelley_2/data/edf_reports')

blink_percentage, quick_percentage, slow_percentage, not_origin_percentage, distractor_percentage, nowhere_percentage, target_percentage = [[] for i in range(7)]

px_per_deg = 27.5
roi_check = 1.6*px_per_deg #this is 2 * size of circle stimuli

for sb in range(len(subjects)):
	
	# Read saccades file for this subject only	
	this_subject = pd.read_csv(subject_files[sb]) 
	
	list(this_subject.columns.values)
	
	#Converting strings ('objects' in pandas terminology) that should be numeric to floats
	this_subject = this_subject.convert_objects(convert_numeric=True)
	
	if subjects[sb] not in ('LP06','LP09','LP11','LP13','LP14','LP16','LP18','LP20','LP21'):
		exp_trials = this_subject[this_subject['trial_no']>75] #skipping practice
	elif subjects[sb] == 'LP06':
		exp_trials = this_subject[((this_subject['trial_no']>75) & (this_subject['trial_no']<=615)) | (this_subject['trial_no']>705)] #skip block 7
	elif subjects[sb] in ('LP09','LP11'):
		exp_trials = this_subject[(this_subject['trial_no']>165)] #skipping block 1
	elif subjects[sb] == 'LP13':
		exp_trials = this_subject[((this_subject['trial_no']>75) & (this_subject['trial_no']<=900)) | (this_subject['trial_no']>975)]# exclude block 10
	elif subjects[sb] == 'LP14':
		exp_trials = this_subject[((this_subject['trial_no']>75) & (this_subject['trial_no']<=900)) | ((this_subject['trial_no']>975) & (this_subject['trial_no']<=1335)) | (this_subject['trial_no']>1425)]
	elif subjects[sb] == 'LP16':
		exp_trials = this_subject[((this_subject['trial_no']>75) & (this_subject['trial_no']<=1245)) | (this_subject['trial_no']>1335)] # 1 block
	elif subjects[sb] == 'LP18':
		exp_trials = this_subject[((this_subject['trial_no']>75) & (this_subject['trial_no']<=1335)) | (this_subject['trial_no']>1425)]  # skipping block 15
	elif subjects[sb] == 'LP20':
		exp_trials = this_subject[((this_subject['trial_no']>75) & (this_subject['trial_no']<=1425)) | (exp_trials['trial_no']<=1515)]# skipping block 16
	elif subjects[sb] == 'LP21':
		exp_trials = this_subject[((this_subject['trial_no']>75) & (this_subject['trial_no']<=900)) | (this_subject['trial_no']>=1065)] # skipping block 10 and 11
		
	exp_trials = exp_trials[(exp_trials['CURRENT_SAC_INDEX']== 1)]
	
	# Blinks
	blinks = exp_trials[(exp_trials['CURRENT_SAC_CONTAINS_BLINK']== True)]
	blink_percent = (blinks.shape[0]/exp_trials.shape[0]) * 100
	
	# Latency < 80ms
	quick = exp_trials[(exp_trials['saccade_onset_target'] < 80) & (exp_trials['saccade_onset_distractor'] < 80)] #if eye movement is to target, distractor onset will be 0.0 -> need to account for this using '&' symbol
	quick_percent = (quick.shape[0]/exp_trials.shape[0]) * 100
	
	# Landing at target > rt_cutoff
	slow = exp_trials[(exp_trials['saccade_arrival_target'] > exp_trials['rt_cutoff']) | (exp_trials['saccade_arrival_distractor'] > exp_trials['rt_cutoff'])]
	slow_percent = (slow.shape[0]/exp_trials.shape[0]) * 100
	
	# Not starting from origin
	origin = exp_trials[(exp_trials['CURRENT_SAC_START_X'] > (exp_trials['cur_x_start']-(roi_check))) & (exp_trials['CURRENT_SAC_START_X'] < (exp_trials['cur_x_start'] + (roi_check))) & (exp_trials['CURRENT_SAC_START_Y']> (exp_trials['cur_y_start']-(roi_check))) & (exp_trials['CURRENT_SAC_START_Y']< (exp_trials['cur_y_start']+(roi_check)))]
	not_origin_percent = 100 - ((origin.shape[0]/exp_trials.shape[0]) * 100)
	
	# Landing at distractor
	distractor = exp_trials[exp_trials['which_circle']== 'd']
	distractor_percent = (distractor.shape[0]/exp_trials.shape[0]) * 100
	
	# Landing on neither target nor distractor
	target_or_distractor = exp_trials[(exp_trials['which_circle'] != 't') & (exp_trials['which_circle'] != 'd')]
	nowhere_percent = (target_or_distractor.shape[0]/exp_trials.shape[0]) * 100
	
	# Landing at target

	target = exp_trials[exp_trials['which_circle']== 't']
	
	target_percent = (target.shape[0]/exp_trials.shape[0]) * 100
	
	# All subjects
	blink_percentage.append(blink_percent)
	quick_percentage.append(quick_percent)
	slow_percentage.append(slow_percent)
	not_origin_percentage.append(not_origin_percent)
	distractor_percentage.append(distractor_percent)
	nowhere_percentage.append(nowhere_percent)
	target_percentage.append(target_percent)


# Exclusions: all means and stds
blink_mean = np.mean(blink_percentage)
blink_std = np.std(blink_percentage)

quick_mean = np.mean(quick_percent)
quick_std = np.std(quick_percent)

slow_mean = np.mean(slow_percentage)
slow_std = np.std(slow_percentage)

not_origin_mean = np.mean(not_origin_percentage)
not_origin_std = np.std(not_origin_percentage)

distractor_mean = np.mean(distractor_percentage)
distractor_std = np.std(distractor_percentage)

nowhere_mean = np.mean(nowhere_percentage)
nowhere_std = np.std(nowhere_percentage)

target_mean = np.mean(target_percentage)
target_std = np.std(target_percentage)

shell()