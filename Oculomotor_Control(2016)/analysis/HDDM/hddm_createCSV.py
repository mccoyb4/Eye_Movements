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

px_per_deg = 27.5
roi_check = 1.6*px_per_deg #This is 2 * size of circle stimuli, and for each direction (+ and -). -> 3.2 deg of the centre of the circle.

for sb in range(len(subjects)):
	
	os.chdir('/Users/bronagh/Documents/LePelley/LePelley_2/data/edf_reports')
	
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
	
	# Take all trials with saccade onset > 80 and < rt_cutoff	
	exp_trials = exp_trials[(exp_trials['CURRENT_SAC_INDEX']== 1) & (exp_trials['CURRENT_SAC_CONTAINS_BLINK']== False)]
	exp_trials = exp_trials[((exp_trials['saccade_onset_target'] > 80) & (exp_trials['saccade_arrival_target'] < exp_trials['rt_cutoff'])) | ((exp_trials['saccade_onset_distractor'] > 80) & (exp_trials['saccade_arrival_distractor'] < exp_trials['rt_cutoff']))]

	# Starting from origin
	trials = exp_trials[(exp_trials['CURRENT_SAC_START_X'] > (exp_trials['cur_x_start']-(roi_check))) & (exp_trials['CURRENT_SAC_START_X'] < (exp_trials['cur_x_start'] + (roi_check))) & (exp_trials['CURRENT_SAC_START_Y']> (exp_trials['cur_y_start']-(roi_check))) & (exp_trials['CURRENT_SAC_START_Y']< (exp_trials['cur_y_start']+(roi_check)))]
	
	# Take only trials where they looked to target or distractor (not some random place on screen)
	trials = trials[(trials['which_circle'] == 't') | (trials['which_circle'] == 'd')]
	
	hddm_file = trials[['reward_available','which_circle','distractor_angle_deg', 'd_x','t_x', 'saccade_onset_target','saccade_onset_distractor']]
		
	# Creating subject # series to add to dataframe
	sb_number = np.array(np.ones(trials.shape[0]),dtype=int)
	sb_number[:] = sb
	sb_series = pd.Series(sb_number, index=hddm_file.index)
	
	# Adding two new columns : sb_id, and hemifield_diff
	hddm_file['sb_id'] = sb_series	
	hddm_file['hemi_diff'] = abs(hddm_file.d_x - hddm_file.t_x)
	
	hddm_array = np.array(hddm_file)
	
	# Hemifield, Which circle, Angle and Reward recoding
	for i in range(len(hddm_array)):
		
		#hemifield
		if hddm_array[i][8] < 60: # converting hemifield number to same hemifield (0) or opposite hemifield (1)
			hddm_array[i][8] = 0
		else:
			hddm_array[i][8] = 1
			
		#circle looked at (this corresponds to "response" in hddm)
		if hddm_array[i][1] == 't':
			hddm_array[i][1] = 1
		else:
			hddm_array[i][1] = 0
		
		#angle
		if hddm_array[i][2] == 29: #small
			hddm_array[i][2] = 's'
		elif hddm_array[i][2] == 119: #med
			hddm_array[i][2] = 'm'
		else:
			hddm_array[i][2] = 'l' #large
		
		#reward
		if hddm_array[i][0] == 0: #high 
			hddm_array[i][0] = 'h'
		elif hddm_array[i][0] == 1: #low
			hddm_array[i][0] = 'l'
		else:
			hddm_array[i][0] = 'n' #no
		
	hddm_output = hddm_array[:,[7,0,1,2,5,6,8]]
	
	for i in range(len(hddm_output)):
		if hddm_output[i][4]==0.0:
			hddm_output[i][4] = hddm_output[i][5]
		hddm_output[i][4] = hddm_output[i][4]/1000
	
	hddm_output = hddm_output[:,[0,1,2,3,6,4]] #these refer to : [sub_id, reward_available, which_circle, distractor_angle_deg, hemifield, srt]
	
	#convert back to dataframe
	hddm_csv = pd.DataFrame(hddm_output)
	hddm_hemi_csv = hddm_csv[hddm_csv[3]=='m']

	hddm_csv.rename(columns={0:'subj_idx',1:'reward',2:'response',3:'angle',4:'hemifield',5:'rt'}, inplace=True)
	hddm_hemi_csv.rename(columns={0:'subj_idx',1:'reward',2:'response',3:'angle',4:'hemifield',5:'rt'}, inplace=True)
	
	os.chdir('/Users/bronagh/Documents/LePelley/LePelley_2/results/hddm')
	if sb == 0:
		hddm_csv.to_csv('hddm.csv', index = False)
		hddm_hemi_csv.to_csv('hddm_hemi.csv', index = False)
	else:
		with open('hddm.csv', 'a') as f:
			hddm_csv.to_csv(f,index = False, header=False)
		with open('hddm_hemi.csv', 'a') as h:
			hddm_hemi_csv.to_csv(h,index = False, header=False)

