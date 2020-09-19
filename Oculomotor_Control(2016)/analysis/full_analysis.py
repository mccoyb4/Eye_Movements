
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
subject_files = ['LP01.csv','LP02.csv','LP03.csv','LP04.csv','LP05.csv', 'LP06.csv','LP07.csv','LP08.csv','LP09.csv','LP10.csv','LP11.csv','LP12.csv','LP13.csv','LP14.csv','LP15.csv', 'LP16.csv','LP17.csv','LP18.csv','LP19.csv','LP20.csv','LP21.csv','LP22.csv','LP23.csv','LP24.csv']

#px_per_deg = 27.5

def subject_specific_exclusions(sb, this_subject):
	
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
		exp_trials = this_subject[((this_subject['trial_no']>75) & (this_subject['trial_no']<=1425)) | (this_subject['trial_no']>=1515)]# skipping block 16
	elif subjects[sb] == 'LP21':
		exp_trials = this_subject[((this_subject['trial_no']>75) & (this_subject['trial_no']<=900)) | (this_subject['trial_no']>=1065)] # skipping block 10 and 11	
	
	return exp_trials

def preprocessing(sb, which_type, analysis):
	
	os.chdir('/Users/bronagh/Documents/LePelley/LePelley_2/data/edf_reports')
	
	# Read saccades file for this subject only	
	this_subject = pd.read_csv(subject_files[sb]) 

	list(this_subject.columns.values)

	#Converting strings ('objects' in pandas terminology) that should be numeric to floats
	
	if analysis not in ('curvature_absolute'):
		
		this_subject = this_subject.convert_objects(convert_numeric=True)
	
		global trials
		trials = subject_specific_exclusions(sb, this_subject)
	
		# Take first saccade trials without any blinks		
		trials = trials[(trials['CURRENT_SAC_INDEX']== 1) & (trials['CURRENT_SAC_CONTAINS_BLINK']== False)]
		
		global pre_trials # Will use pre_trials later to calculate the number of 'allowed' trials in erroneous_saccades()
		
		# Take trials within correct timeframe (80ms - rt_cutoff)
		# Note: pre_trials becomes 'trials' again at the end of this function, so those are the trials carried to the next step of the analysis
		pre_trials = trials[((trials['saccade_onset_target'] > 80) & (trials['saccade_arrival_target'] < trials['rt_cutoff'])) | ((trials['saccade_onset_distractor'] > 80) & (trials['saccade_arrival_distractor'] < trials['rt_cutoff']))]		
		
	elif analysis in ('curvature_absolute'):

		curvature_files = ['LP01_curvature_peak.csv','LP02_curvature_peak.csv','LP03_curvature_peak.csv','LP04_curvature_peak.csv','LP05_curvature_peak.csv','LP06_curvature_peak.csv','LP07_curvature_peak.csv','LP08_curvature_peak.csv','LP09_curvature_peak.csv','LP10_curvature_peak.csv', 'LP11_curvature_peak.csv', 'LP12_curvature_peak.csv', 'LP13_curvature_peak.csv','LP14_curvature_peak.csv','LP15_curvature_peak.csv','LP16_curvature_peak.csv','LP17_curvature_peak.csv','LP18_curvature_peak.csv','LP19_curvature_peak.csv','LP20_curvature_peak.csv','LP21_curvature_peak.csv','LP22_curvature_peak.csv','LP23_curvature_peak.csv','LP24_curvature_peak.csv']
	
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
		trials = sacc_curve.drop_duplicates(subset = 'trial_no')
		trials = trials[(np.isfinite(trials['curvature'])) & (trials['curvature']!= 0)]
	
		# Take first saccade trials, with saccade onset > 80 and < rt_cutoff. These can be either to target or distractor.
		trials = trials[(trials['CURRENT_SAC_INDEX']== 1) & (trials['CURRENT_SAC_CONTAINS_BLINK']== False)]
		trials = trials[((trials['saccade_onset_target'] > 80) & (trials['saccade_arrival_target'] < trials['rt_cutoff'])) | ((trials['saccade_onset_distractor'] > 80) & (trials['saccade_arrival_distractor'] < trials['rt_cutoff']))]
	
		global pre_trials
		pre_trials = subject_specific_exclusions(sb, trials)

	if which_type == 'target':
		trials = pre_trials[pre_trials['which_circle']=='t']
	elif which_type =='distractor':
		trials = pre_trials[pre_trials['which_circle']=='d']
	else:
		trials = pre_trials

	return trials

def exclusions(sb):
	
	os.chdir('/Users/bronagh/Documents/LePelley/LePelley_2/data/edf_reports')
	
	# Read saccades file for this subject only	
	this_subject = pd.read_csv(subject_files[sb]) 

	list(this_subject.columns.values)

	#Converting strings ('objects' in pandas terminology) that should be numeric to float
	excl_this_subject = this_subject.convert_objects(convert_numeric=True)
	excl_trials = subject_specific_exclusions(sb, this_subject)
	excl_trials = excl_trials[(excl_trials['CURRENT_SAC_INDEX']== 1)]
	
	target_percentage = (excl_trials[excl_trials['which_circle']=='t'].shape[0]/excl_trials.shape[0]) * 100
	distractor_percentage = (excl_trials[excl_trials['which_circle']=='d'].shape[0]/excl_trials.shape[0]) * 100
	neither_percentage = (excl_trials[(excl_trials['which_circle']!='t') & (excl_trials['which_circle']!='d') ].shape[0]/excl_trials.shape[0]) * 100
	
	blink_percentage = (excl_trials[excl_trials['CURRENT_SAC_CONTAINS_BLINK']== True].shape[0]/excl_trials.shape[0]) * 100
	
	premature_percentage = (excl_trials[(excl_trials['saccade_onset_target'] < 80) & (excl_trials['saccade_onset_distractor'] < 80)].shape[0] /excl_trials.shape[0]) * 100
	late_percentage = (excl_trials[(excl_trials['saccade_arrival_target'] > excl_trials['rt_cutoff']) | (excl_trials['saccade_arrival_distractor'] > excl_trials['rt_cutoff'])].shape[0] /excl_trials.shape[0]) * 100
	
	if sb == 0:
		global exclusion_list
		exclusion_list = []
	
	exclusion_list.append([target_percentage,distractor_percentage,neither_percentage,blink_percentage, premature_percentage,late_percentage])
	
	if sb == 23:
		exclusion_list = np.array(exclusion_list)
		exclusion_means = [np.mean(exclusion_list[:,0]),np.mean(exclusion_list[:,1]),np.mean(exclusion_list[:,2]),np.mean(exclusion_list[:,3]),np.mean(exclusion_list[:,4]),np.mean(exclusion_list[:,5])]
		exclusion_stds = [np.std(exclusion_list[:,0]),np.std(exclusion_list[:,1]),np.std(exclusion_list[:,2]),np.std(exclusion_list[:,3]),np.std(exclusion_list[:,4]),np.std(exclusion_list[:,5])]		
		
		print 'Exlusions: '
		exlusion_df = pd.DataFrame.from_items([('Means', exclusion_means), ('SDs', exclusion_stds)], orient='index', columns=['target', 'distractor', 'neither', 'blinks', 'tooEarly', 'tooLate'])
		print exlusion_df

def dataframes(trials):
	
	global small_high
	global small_low
	global small_no
	global med_high
	global med_low
	global med_no
	global large_high
	global large_low
	global large_no
	
	global samehemi_high
	global samehemi_low
	global samehemi_no
	global opphemi_high
	global opphemi_low
	global opphemi_no
	global samehemi_all
	global opphemi_all
	
	# Angles
	small_high = trials[(trials['distractor_angle_deg']==29) & (trials['reward_available']==0)]
	small_low = trials[(trials['distractor_angle_deg']==29) & (trials['reward_available']==1)]
	small_no = trials[(trials['distractor_angle_deg']==29) & (trials['reward_available']==2)]
	
	med_high = trials[(trials['distractor_angle_deg']==119) & (trials['reward_available']==0)]
	med_low = trials[(trials['distractor_angle_deg']==119) & (trials['reward_available']==1)]
	med_no = trials[(trials['distractor_angle_deg']==119) & (trials['reward_available']==2)]
	med_all = trials[(trials['distractor_angle_deg']==119) ]
	
	large_high = trials[(trials['distractor_angle_deg']==180) & (trials['reward_available']==0)]
	large_low = trials[(trials['distractor_angle_deg']==180) & (trials['reward_available']==1)]
	large_no = trials[(trials['distractor_angle_deg']==180) & (trials['reward_available']==2)]
	
	# Hemifields (only the 120 deg trials)
	samehemi_high = med_high[abs(med_high['d_x']-med_high['t_x'])<60]
	samehemi_low = med_low[abs(med_low['d_x']-med_low['t_x'])<60]
	samehemi_no = med_no[abs(med_no['d_x']-med_no['t_x'])<60]
	
	opphemi_high = med_high[abs(med_high['d_x']-med_high['t_x'])>60]
	opphemi_low = med_low[abs(med_low['d_x']-med_low['t_x'])>60]
	opphemi_no = med_no[abs(med_no['d_x']-med_no['t_x'])>60]
	
	samehemi_all = med_all[abs(med_all['d_x']-med_all['t_x'])<60]
	opphemi_all = med_all[abs(med_all['d_x']-med_all['t_x'])>60]
	

def saccade_latency(which_type, hemifield):

	if which_type == 'target':
		onset = 'saccade_onset_target'
	elif which_type == 'distractor':
		onset = 'saccade_onset_distractor'

	# Angle
	small_high_srt = np.mean(small_high[onset])
	small_low_srt = np.mean(small_low[onset])
	small_no_srt = np.mean(small_no[onset])
	
	med_high_srt = np.mean(med_high[onset])
	med_low_srt = np.mean(med_low[onset])
	med_no_srt = np.mean(med_no[onset])

	large_high_srt = np.mean(large_high[onset])
	large_low_srt = np.mean(large_low[onset])
	large_no_srt = np.mean(large_no[onset])
	
	#Latency histograms - for Jessica Heeman
	for i in range(len(small_high)):
		latency_histogram[0].append(small_high[onset].iloc[i])
		
	for i in range(len(small_low)):
		latency_histogram[1].append(small_low[onset].iloc[i])
		
	for i in range(len(small_no)):
		latency_histogram[2].append(small_no[onset].iloc[i])
		
	for i in range(len(med_high)):
		latency_histogram[3].append(med_high[onset].iloc[i])

	for i in range(len(med_low)):
		latency_histogram[4].append(med_low[onset].iloc[i])
	
	for i in range(len(med_no)):
		latency_histogram[5].append(med_no[onset].iloc[i])

	for i in range(len(large_high)):
		latency_histogram[6].append(large_high[onset].iloc[i])

	for i in range(len(large_low)):
		latency_histogram[7].append(large_low[onset].iloc[i])

	for i in range(len(large_no)):
		latency_histogram[8].append(large_no[onset].iloc[i])

	
	# Hemifield
	samehemi_high_srt = np.mean(samehemi_high[onset])
	samehemi_low_srt = np.mean(samehemi_low[onset])
	samehemi_no_srt = np.mean(samehemi_no[onset])
	
	opphemi_high_srt = np.mean(opphemi_high[onset])
	opphemi_low_srt = np.mean(opphemi_low[onset])
	opphemi_no_srt = np.mean(opphemi_no[onset])
	
	samehemi_all_srt = np.mean(samehemi_all[onset])
	opphemi_all_srt = np.mean(opphemi_all[onset])
	
	if hemifield == 0:
		return [small_high_srt, small_low_srt, small_no_srt, med_high_srt, med_low_srt, med_no_srt, large_high_srt, large_low_srt, large_no_srt]
	elif hemifield == 1:
		return [samehemi_high_srt, samehemi_low_srt, samehemi_no_srt, opphemi_high_srt, opphemi_low_srt, opphemi_no_srt]
		
def saccade_median_latency(which_type, hemifield):

	if which_type == 'target':
		onset = 'saccade_onset_target'
	elif which_type == 'distractor':
		onset = 'saccade_onset_distractor'
	
	# Angle
	small_high_srt = np.median(small_high[onset])
	small_low_srt = np.median(small_low[onset])
	small_no_srt = np.median(small_no[onset])
	
	med_high_srt = np.median(med_high[onset])
	med_low_srt = np.median(med_low[onset])
	med_no_srt = np.median(med_no[onset])

	large_high_srt = np.median(large_high[onset])
	large_low_srt = np.median(large_low[onset])
	large_no_srt = np.median(large_no[onset])
	
	# Hemifield
	samehemi_high_srt = np.median(samehemi_high[onset])
	samehemi_low_srt = np.median(samehemi_low[onset])
	samehemi_no_srt = np.median(samehemi_no[onset])
	
	opphemi_high_srt = np.median(opphemi_high[onset])
	opphemi_low_srt = np.median(opphemi_low[onset])
	opphemi_no_srt = np.median(opphemi_no[onset])
	
	if hemifield == 0:
		return [small_high_srt, small_low_srt, small_no_srt, med_high_srt, med_low_srt, med_no_srt, large_high_srt, large_low_srt, large_no_srt]
	elif hemifield == 1:
		return [samehemi_high_srt, samehemi_low_srt, samehemi_no_srt, opphemi_high_srt, opphemi_low_srt, opphemi_no_srt]


def mean_latency(which_type):

	if which_type == 'target':
		onset = 'saccade_onset_target'
		mean_latencies = np.mean(trials[trials[onset]])
	elif which_type == 'distractor':
		onset = 'saccade_onset_distractor'
		mean_latencies = np.mean(trials[trials[onset]])
	elif which_type == 'both':
		onset_t = 'saccade_onset_target'
		onset_d = 'saccade_onset_distractor'
		
		latency_dict = {}
	
		for i in range(len(trials)):
			if trials.iloc[i]['saccade_onset_target'] > 0:
				latency_dict[i] = {'latency_target_or_distractor': trials.iloc[i]['saccade_onset_target']}
			elif trials.iloc[i]['saccade_onset_distractor'] > 0:
				latency_dict[i] = {'latency_target_or_distractor': trials.iloc[i]['saccade_onset_distractor']}
			
		# Turn dict into a dataframe
		latency_dict = pd.DataFrame.from_dict(latency_dict, orient = 'index', dtype = float)	
		trials.index = range(len(trials))
		latency_trials = pd.concat([trials, latency_dict], axis=1)
		
		mean_latencies = np.mean(latency_trials['latency_target_or_distractor'])
		
	return [mean_latencies]

def learning_latencies(which_type, hemifield):
	# Added July 2016 for resubmission to JoNeurophysiology
	# Checking evolution of errors across 5 "blocks" 

	if which_type == 'target':
		onset = 'saccade_onset_target'
	elif which_type == 'distractor':
		onset = 'saccade_onset_distractor'

	### SMALL ANGLE ###	
	
	#High reward
	small_high_bins = [small_high[(small_high['block_no'] > 0) & (small_high['block_no'] <= 4)],small_high[(small_high['block_no'] > 4) & (small_high['block_no'] <= 8)], small_high[(small_high['block_no'] > 8) & (small_high['block_no'] <= 12)], small_high[(small_high['block_no'] > 12) & (small_high['block_no'] <= 16)], small_high[(small_high['block_no'] > 16) & (small_high['block_no'] <= 20)]]
	small_high_srts = [np.mean(small_high_bins[0][onset]), np.mean(small_high_bins[1][onset]), np.mean(small_high_bins[2][onset]), np.mean(small_high_bins[3][onset]), np.mean(small_high_bins[4][onset])]
	small_high_stds = [np.std(small_high_bins[0][onset]), np.std(small_high_bins[1][onset]), np.std(small_high_bins[2][onset]), np.std(small_high_bins[3][onset]), np.std(small_high_bins[4][onset])]
	
	#Low
	small_low_bins = [small_low[(small_low['block_no'] > 0) & (small_low['block_no'] <= 4)],small_low[(small_low['block_no'] > 4) & (small_low['block_no'] <= 8)], small_low[(small_low['block_no'] > 8) & (small_low['block_no'] <= 12)], small_low[(small_low['block_no'] > 12) & (small_low['block_no'] <= 16)], small_low[(small_low['block_no'] > 16) & (small_low['block_no'] <= 20)]]
	small_low_srts = [np.mean(small_low_bins[0][onset]), np.mean(small_low_bins[1][onset]), np.mean(small_low_bins[2][onset]), np.mean(small_low_bins[3][onset]), np.mean(small_low_bins[4][onset])]
	small_low_stds = [np.std(small_low_bins[0][onset]), np.std(small_low_bins[1][onset]), np.std(small_low_bins[2][onset]), np.std(small_low_bins[3][onset]), np.std(small_low_bins[4][onset])]
	
	#No
	small_no_bins = [small_no[(small_no['block_no'] > 0) & (small_no['block_no'] <= 4)],small_no[(small_no['block_no'] > 4) & (small_no['block_no'] <= 8)], small_no[(small_no['block_no'] > 8) & (small_no['block_no'] <= 12)], small_no[(small_no['block_no'] > 12) & (small_no['block_no'] <= 16)], small_no[(small_no['block_no'] > 16) & (small_no['block_no'] <= 20)]]
	small_no_srts = [np.mean(small_no_bins[0][onset]), np.mean(small_no_bins[1][onset]), np.mean(small_no_bins[2][onset]), np.mean(small_no_bins[3][onset]), np.mean(small_no_bins[4][onset])]
	small_no_stds = [np.std(small_no_bins[0][onset]), np.std(small_no_bins[1][onset]), np.std(small_no_bins[2][onset]), np.std(small_no_bins[3][onset]), np.std(small_no_bins[4][onset])]
	
	small_high_no_diff_srts = list(np.array(small_high_srts) - np.array(small_no_srts))
	small_high_no_diff_stds = list(np.mean([small_high_stds,small_no_stds], axis=0))
	
	### MEDIUM ANGLE
	
	#High reward
	med_high_bins = [med_high[(med_high['block_no'] > 0) & (med_high['block_no'] <= 4)],med_high[(med_high['block_no'] > 4) & (med_high['block_no'] <= 8)], med_high[(med_high['block_no'] > 8) & (med_high['block_no'] <= 12)], med_high[(med_high['block_no'] > 12) & (med_high['block_no'] <= 16)], med_high[(med_high['block_no'] > 16) & (med_high['block_no'] <= 20)]]
	med_high_srts = [np.mean(med_high_bins[0][onset]), np.mean(med_high_bins[1][onset]), np.mean(med_high_bins[2][onset]), np.mean(med_high_bins[3][onset]), np.mean(med_high_bins[4][onset])]
	med_high_stds = [np.std(med_high_bins[0][onset]), np.std(med_high_bins[1][onset]), np.std(med_high_bins[2][onset]), np.std(med_high_bins[3][onset]), np.std(med_high_bins[4][onset])]

	#Low
	med_low_bins = [med_low[(med_low['block_no'] > 0) & (med_low['block_no'] <= 4)],med_low[(med_low['block_no'] > 4) & (med_low['block_no'] <= 8)], med_low[(med_low['block_no'] > 8) & (med_low['block_no'] <= 12)], med_low[(med_low['block_no'] > 12) & (med_low['block_no'] <= 16)], med_low[(med_low['block_no'] > 16) & (med_low['block_no'] <= 20)]]
	med_low_srts = [np.mean(med_low_bins[0][onset]), np.mean(med_low_bins[1][onset]), np.mean(med_low_bins[2][onset]), np.mean(med_low_bins[3][onset]), np.mean(med_low_bins[4][onset])]
	med_low_stds = [np.std(med_low_bins[0][onset]), np.std(med_low_bins[1][onset]), np.std(med_low_bins[2][onset]), np.std(med_low_bins[3][onset]), np.std(med_low_bins[4][onset])]
	
	#No
	med_no_bins = [med_no[(med_no['block_no'] > 0) & (med_no['block_no'] <= 4)],med_no[(med_no['block_no'] > 4) & (med_no['block_no'] <= 8)], med_no[(med_no['block_no'] > 8) & (med_no['block_no'] <= 12)], med_no[(med_no['block_no'] > 12) & (med_no['block_no'] <= 16)], med_no[(med_no['block_no'] > 16) & (med_no['block_no'] <= 20)]]
	med_no_srts = [np.mean(med_no_bins[0][onset]), np.mean(med_no_bins[1][onset]), np.mean(med_no_bins[2][onset]), np.mean(med_no_bins[3][onset]), np.mean(med_no_bins[4][onset])]
	med_no_stds = [np.std(med_no_bins[0][onset]), np.std(med_no_bins[1][onset]), np.std(med_no_bins[2][onset]), np.std(med_no_bins[3][onset]), np.std(med_no_bins[4][onset])]

	med_high_no_diff_srts = list(np.array(med_high_srts) - np.array(med_no_srts))
	med_high_no_diff_stds = list(np.mean([med_high_stds,med_no_stds], axis=0))	

	### LARGE ANGLE
	
	#High reward
	large_high_bins = [large_high[(large_high['block_no'] > 0) & (large_high['block_no'] <= 4)],large_high[(large_high['block_no'] > 4) & (large_high['block_no'] <= 8)], large_high[(large_high['block_no'] > 8) & (large_high['block_no'] <= 12)], large_high[(large_high['block_no'] > 12) & (large_high['block_no'] <= 16)], large_high[(large_high['block_no'] > 16) & (large_high['block_no'] <= 20)]]
	large_high_srts = [np.mean(large_high_bins[0][onset]), np.mean(large_high_bins[1][onset]), np.mean(large_high_bins[2][onset]), np.mean(large_high_bins[3][onset]), np.mean(large_high_bins[4][onset])]
	large_high_stds = [np.std(large_high_bins[0][onset]), np.std(large_high_bins[1][onset]), np.std(large_high_bins[2][onset]), np.std(large_high_bins[3][onset]), np.std(large_high_bins[4][onset])]

	#Low
	large_low_bins = [large_low[(large_low['block_no'] > 0) & (large_low['block_no'] <= 4)],large_low[(large_low['block_no'] > 4) & (large_low['block_no'] <= 8)], large_low[(large_low['block_no'] > 8) & (large_low['block_no'] <= 12)], large_low[(large_low['block_no'] > 12) & (large_low['block_no'] <= 16)], large_low[(large_low['block_no'] > 16) & (large_low['block_no'] <= 20)]]
	large_low_srts = [np.mean(large_low_bins[0][onset]), np.mean(large_low_bins[1][onset]), np.mean(large_low_bins[2][onset]), np.mean(large_low_bins[3][onset]), np.mean(large_low_bins[4][onset])]
	large_low_stds = [np.std(large_high_bins[0][onset]), np.std(large_high_bins[1][onset]), np.std(large_high_bins[2][onset]), np.std(large_high_bins[3][onset]), np.std(large_high_bins[4][onset])]
		
	#No
	large_no_bins = [large_no[(large_no['block_no'] > 0) & (large_no['block_no'] <= 4)],large_no[(large_no['block_no'] > 4) & (large_no['block_no'] <= 8)], large_no[(large_no['block_no'] > 8) & (large_no['block_no'] <= 12)], large_no[(large_no['block_no'] > 12) & (large_no['block_no'] <= 16)], large_no[(large_no['block_no'] > 16) & (large_no['block_no'] <= 20)]]
	large_no_srts = [np.mean(large_no_bins[0][onset]), np.mean(large_no_bins[1][onset]), np.mean(large_no_bins[2][onset]), np.mean(large_no_bins[3][onset]), np.mean(large_no_bins[4][onset])]
	large_no_stds = [np.std(large_no_bins[0][onset]), np.std(large_no_bins[1][onset]), np.std(large_no_bins[2][onset]), np.std(large_no_bins[3][onset]), np.std(large_no_bins[4][onset])]

	large_high_no_diff_srts = list(np.array(large_high_srts) - np.array(large_no_srts))
	large_high_no_diff_stds = list(np.mean([large_high_stds,large_no_stds], axis=0))

	### PLOTS ###
	 
	x=np.arange(1,6)
	plt.xlim(xmin=0.5, xmax = 5.5)
	plt.title('Saccade latency difference between high and no reward across bins')
	plt.xlabel('Block')
	plt.ylabel('Saccade latency difference (ms)')
	plt.plot(x,small_high_no_diff_srts,'g', label = 'Small angle')
	plt.errorbar(x,small_high_no_diff_srts, yerr = small_high_no_diff_stds, color = 'g')
	plt.plot(x,med_high_no_diff_srts,'b',label = 'Medium angle')
	plt.errorbar(x,med_high_no_diff_srts, yerr = med_high_no_diff_stds, color = 'b')
	plt.plot(x,large_high_no_diff_srts,'r',label = 'Large angle')
	plt.errorbar(x,large_high_no_diff_srts, yerr = large_high_no_diff_stds, color = 'r')
	plt.legend(loc='best')
	plt.savefig('latency_bins_angles.png')
	#plt.show()
	plt.close()

	# Hemifield
	
	#Same hemi
	#High reward
	samehemi_high_bins = [samehemi_high[(samehemi_high['block_no'] > 0) & (samehemi_high['block_no'] <= 4)],samehemi_high[(samehemi_high['block_no'] > 4) & (samehemi_high['block_no'] <= 8)], samehemi_high[(samehemi_high['block_no'] > 8) & (samehemi_high['block_no'] <= 12)], samehemi_high[(samehemi_high['block_no'] > 12) & (samehemi_high['block_no'] <= 16)], samehemi_high[(samehemi_high['block_no'] > 16) & (samehemi_high['block_no'] <= 20)]]
	samehemi_high_srts = [np.mean(samehemi_high_bins[0][onset]), np.mean(samehemi_high_bins[1][onset]), np.mean(samehemi_high_bins[2][onset]), np.mean(samehemi_high_bins[3][onset]), np.mean(samehemi_high_bins[4][onset])]
	samehemi_high_stds = [np.std(samehemi_high_bins[0][onset]), np.std(samehemi_high_bins[1][onset]), np.std(samehemi_high_bins[2][onset]), np.std(samehemi_high_bins[3][onset]), np.std(samehemi_high_bins[4][onset])]
	
	#Low
	samehemi_low_bins = [samehemi_low[(samehemi_low['block_no'] > 0) & (samehemi_low['block_no'] <= 4)],samehemi_low[(samehemi_low['block_no'] > 4) & (samehemi_low['block_no'] <= 8)], samehemi_low[(samehemi_low['block_no'] > 8) & (samehemi_low['block_no'] <= 12)], samehemi_low[(samehemi_low['block_no'] > 12) & (samehemi_low['block_no'] <= 16)], samehemi_low[(samehemi_low['block_no'] > 16) & (samehemi_low['block_no'] <= 20)]]
	samehemi_low_srts = [np.mean(samehemi_low_bins[0][onset]), np.mean(samehemi_low_bins[1][onset]), np.mean(samehemi_low_bins[2][onset]), np.mean(samehemi_low_bins[3][onset]), np.mean(samehemi_low_bins[4][onset])]
	samehemi_low_stds = [np.std(samehemi_low_bins[0][onset]), np.std(samehemi_low_bins[1][onset]), np.std(samehemi_low_bins[2][onset]), np.std(samehemi_low_bins[3][onset]), np.std(samehemi_low_bins[4][onset])]
	
	#No
	samehemi_no_bins = [samehemi_no[(samehemi_no['block_no'] > 0) & (samehemi_no['block_no'] <= 4)],samehemi_no[(samehemi_no['block_no'] > 4) & (samehemi_no['block_no'] <= 8)], samehemi_no[(samehemi_no['block_no'] > 8) & (samehemi_no['block_no'] <= 12)], samehemi_no[(samehemi_no['block_no'] > 12) & (samehemi_no['block_no'] <= 16)], samehemi_no[(samehemi_no['block_no'] > 16) & (samehemi_no['block_no'] <= 20)]]
	samehemi_no_srts = [np.mean(samehemi_no_bins[0][onset]), np.mean(samehemi_no_bins[1][onset]), np.mean(samehemi_no_bins[2][onset]), np.mean(samehemi_no_bins[3][onset]), np.mean(samehemi_no_bins[4][onset])]
	samehemi_no_stds = [np.std(samehemi_no_bins[0][onset]), np.std(samehemi_no_bins[1][onset]), np.std(samehemi_no_bins[2][onset]), np.std(samehemi_no_bins[3][onset]), np.std(samehemi_no_bins[4][onset])]
	
	samehemi_high_no_diff_srts = list(np.array(samehemi_high_srts) - np.array(samehemi_no_srts))
	samehemi_high_no_diff_stds = list(np.mean([samehemi_high_stds,samehemi_no_stds], axis=0))

	#Opp hemi
	#High reward
	opphemi_high_bins = [opphemi_high[(opphemi_high['block_no'] > 0) & (opphemi_high['block_no'] <= 4)],opphemi_high[(opphemi_high['block_no'] > 4) & (opphemi_high['block_no'] <= 8)], opphemi_high[(opphemi_high['block_no'] > 8) & (opphemi_high['block_no'] <= 12)], opphemi_high[(opphemi_high['block_no'] > 12) & (opphemi_high['block_no'] <= 16)], opphemi_high[(opphemi_high['block_no'] > 16) & (opphemi_high['block_no'] <= 20)]]
	opphemi_high_srts = [np.mean(opphemi_high_bins[0][onset]), np.mean(opphemi_high_bins[1][onset]), np.mean(opphemi_high_bins[2][onset]), np.mean(opphemi_high_bins[3][onset]), np.mean(opphemi_high_bins[4][onset])]
	opphemi_high_stds = [np.std(opphemi_high_bins[0][onset]), np.std(opphemi_high_bins[1][onset]), np.std(opphemi_high_bins[2][onset]), np.std(opphemi_high_bins[3][onset]), np.std(opphemi_high_bins[4][onset])]
	
	#Low
	opphemi_low_bins = [opphemi_low[(opphemi_low['block_no'] > 0) & (opphemi_low['block_no'] <= 4)],opphemi_low[(opphemi_low['block_no'] > 4) & (opphemi_low['block_no'] <= 8)], opphemi_low[(opphemi_low['block_no'] > 8) & (opphemi_low['block_no'] <= 12)], opphemi_low[(opphemi_low['block_no'] > 12) & (opphemi_low['block_no'] <= 16)], opphemi_low[(opphemi_low['block_no'] > 16) & (opphemi_low['block_no'] <= 20)]]
	opphemi_low_srts = [np.mean(opphemi_low_bins[0][onset]), np.mean(opphemi_low_bins[1][onset]), np.mean(opphemi_low_bins[2][onset]), np.mean(opphemi_low_bins[3][onset]), np.mean(opphemi_low_bins[4][onset])]
	opphemi_low_stds = [np.std(opphemi_low_bins[0][onset]), np.std(opphemi_low_bins[1][onset]), np.std(opphemi_low_bins[2][onset]), np.std(opphemi_low_bins[3][onset]), np.std(opphemi_low_bins[4][onset])]
	
	#No
	opphemi_no_bins = [opphemi_no[(opphemi_no['block_no'] > 0) & (opphemi_no['block_no'] <= 4)],opphemi_no[(opphemi_no['block_no'] > 4) & (opphemi_no['block_no'] <= 8)], opphemi_no[(opphemi_no['block_no'] > 8) & (opphemi_no['block_no'] <= 12)], opphemi_no[(opphemi_no['block_no'] > 12) & (opphemi_no['block_no'] <= 16)], opphemi_no[(opphemi_no['block_no'] > 16) & (opphemi_no['block_no'] <= 20)]]
	opphemi_no_srts = [np.mean(opphemi_no_bins[0][onset]), np.mean(opphemi_no_bins[1][onset]), np.mean(opphemi_no_bins[2][onset]), np.mean(opphemi_no_bins[3][onset]), np.mean(opphemi_no_bins[4][onset])]
	opphemi_no_stds = [np.std(opphemi_no_bins[0][onset]), np.std(opphemi_no_bins[1][onset]), np.std(opphemi_no_bins[2][onset]), np.std(opphemi_no_bins[3][onset]), np.std(opphemi_no_bins[4][onset])]
	
	opphemi_high_no_diff_srts = list(np.array(opphemi_high_srts) - np.array(opphemi_no_srts))
	opphemi_high_no_diff_stds = list(np.mean([opphemi_high_stds,opphemi_no_stds], axis=0))

	x=np.arange(1,6)
	plt.xlim(xmin=0.5, xmax = 5.5)
	plt.title('Saccade latency difference between high and no reward across bins')
	plt.xlabel('Block')
	plt.ylabel('Saccade latency difference (ms)')
	plt.plot(x,samehemi_high_no_diff_srts,'g', label = 'Ipsi hemi')
	plt.errorbar(x,samehemi_high_no_diff_srts, yerr = samehemi_high_no_diff_stds, color = 'g')
	plt.plot(x,opphemi_high_no_diff_srts,'b', label = 'Contra hemi')
	plt.errorbar(x,opphemi_high_no_diff_srts, yerr = opphemi_high_no_diff_stds, color = 'b')
	plt.legend(loc='best')
	plt.savefig('latency_bins_hemi.png')
	#plt.show()
	plt.close()

	if hemifield == 0:
		return [small_high_srts[0],small_high_srts[1],small_high_srts[2],small_high_srts[3],small_high_srts[4], small_low_srts[0],small_low_srts[1],small_low_srts[2],small_low_srts[3],small_low_srts[4], small_no_srts[0],small_no_srts[1],small_no_srts[2],small_no_srts[3],small_no_srts[4],med_high_srts[0],med_high_srts[1],med_high_srts[2],med_high_srts[3],med_high_srts[4], med_low_srts[0], med_low_srts[1],med_low_srts[2],med_low_srts[3],med_low_srts[4], med_no_srts[0],med_no_srts[1],med_no_srts[2],med_no_srts[3],med_no_srts[4], large_high_srts[0], large_high_srts[1],large_high_srts[2],large_high_srts[3],large_high_srts[4],large_low_srts[0], large_low_srts[1],large_low_srts[2],large_low_srts[3],large_low_srts[4], large_no_srts[0], large_no_srts[1],large_no_srts[2],large_no_srts[3],large_no_srts[4]]
	elif hemifield == 1:
		return [samehemi_high_srts[0],samehemi_high_srts[1],samehemi_high_srts[2],samehemi_high_srts[3] ,samehemi_high_srts[4],samehemi_low_srts[0],samehemi_low_srts[1],samehemi_low_srts[2],samehemi_low_srts[3],samehemi_low_srts[4],samehemi_no_srts[0],samehemi_no_srts[1],samehemi_no_srts[2],samehemi_no_srts[3],samehemi_no_srts[4], opphemi_high_srts[0], opphemi_high_srts[1],opphemi_high_srts[2],opphemi_high_srts[3],opphemi_high_srts[4], opphemi_low_srts[0],opphemi_low_srts[1],opphemi_low_srts[2],opphemi_low_srts[3],opphemi_low_srts[4], opphemi_no_srts[0],opphemi_no_srts[1],opphemi_no_srts[2],opphemi_no_srts[3],opphemi_no_srts[4]]

def saccade_amplitude(hemifield):
	
	# Angle
	small_high_amp = np.mean(small_high['CURRENT_SAC_AMPLITUDE'])
	small_low_amp = np.mean(small_low['CURRENT_SAC_AMPLITUDE'])
	small_no_amp = np.mean(small_no['CURRENT_SAC_AMPLITUDE'])
	
	med_high_amp = np.mean(med_high['CURRENT_SAC_AMPLITUDE']) 
	med_low_amp = np.mean(med_low['CURRENT_SAC_AMPLITUDE'])
	med_no_amp = np.mean(med_no['CURRENT_SAC_AMPLITUDE'])
	
	large_high_amp = np.mean(large_high['CURRENT_SAC_AMPLITUDE']) 
	large_low_amp = np.mean(large_low['CURRENT_SAC_AMPLITUDE']) 
	large_no_amp = np.mean(large_no['CURRENT_SAC_AMPLITUDE']) 
	
	# Amplitude
	samehemi_high_amp = np.mean(samehemi_high['CURRENT_SAC_AMPLITUDE'])
	samehemi_low_amp = np.mean(samehemi_low['CURRENT_SAC_AMPLITUDE'])
	samehemi_no_amp = np.mean(samehemi_no['CURRENT_SAC_AMPLITUDE'])
	
	opphemi_high_amp = np.mean(opphemi_high['CURRENT_SAC_AMPLITUDE'])
	opphemi_low_amp = np.mean(opphemi_low['CURRENT_SAC_AMPLITUDE'])
	opphemi_no_amp = np.mean(opphemi_no['CURRENT_SAC_AMPLITUDE'])
	
	if hemifield == 0:
		return [small_high_amp, small_low_amp, small_no_amp, med_high_amp, med_low_amp, med_no_amp, large_high_amp, large_low_amp, large_no_amp]
	elif hemifield == 1:
		return [samehemi_high_amp, samehemi_low_amp, samehemi_no_amp, opphemi_high_amp, opphemi_low_amp, opphemi_no_amp]
		
	
def erroneous_saccades(hemifield):
	
	total_nr_allowed_trials = pre_trials.shape[0]
	
	count_distractor = trials.shape[0]
	
	# Calculate percentage of all allowed trials that participants looked to distractor
	percent_looked_distractor = (count_distractor/total_nr_allowed_trials)*100
	if sb == 0:
		global percent_distractor_trials
		percent_distractor_trials = []
	percent_distractor_trials.append(percent_looked_distractor)

	
	# Angle
	percent_small_high = (small_high.shape[0]/count_distractor)*100
	percent_small_low = (small_low.shape[0]/count_distractor)*100
	percent_small_no = (small_no.shape[0]/count_distractor)*100
	
	percent_med_high = (med_high.shape[0]/count_distractor)*100
	percent_med_low = (med_low.shape[0]/count_distractor)*100
	percent_med_no = (med_no.shape[0]/count_distractor)*100
	
	percent_large_high = (large_high.shape[0]/count_distractor)*100
	percent_large_low = (large_low.shape[0]/count_distractor)*100
	percent_large_no = (large_no.shape[0]/count_distractor)*100
	
	# Hemifield
	percent_samehemi_high = (samehemi_high.shape[0]/count_distractor)*100
	percent_samehemi_low = (samehemi_low.shape[0]/count_distractor)*100
	percent_samehemi_no = (samehemi_no.shape[0]/count_distractor)*100
	
	percent_opphemi_high = (opphemi_high.shape[0]/count_distractor)*100
	percent_opphemi_low = (opphemi_low.shape[0]/count_distractor)*100
	percent_opphemi_no = (opphemi_no.shape[0]/count_distractor)*100
	

	if hemifield == 0:
		return [percent_small_high, percent_small_low, percent_small_no, percent_med_high, percent_med_low, percent_med_no, percent_large_high, percent_large_low, percent_large_no]
	elif hemifield == 1:
		return [percent_samehemi_high, percent_samehemi_low, percent_samehemi_no, percent_opphemi_high, percent_opphemi_low, percent_opphemi_no]


def learning_errors(hemifield):
	# Added July 2016 for resubmission to JoNeurophysiology
	# Checking evolution of errors across 5 "blocks" 

	total_nr_allowed_trials = pre_trials.shape[0]

	count_distractor = trials.shape[0]

	# Calculate percentage of all allowed trials that participants looked to distractor
	percent_looked_distractor = (count_distractor/total_nr_allowed_trials)*100
	if sb == 0:
		global percent_distractor_trials
		percent_distractor_trials = []
	percent_distractor_trials.append(percent_looked_distractor)
	
	### SMALL ANGLE
	
	# High
	small_high_bins = [small_high[(small_high['block_no'] > 0) & (small_high['block_no'] <= 4)],small_high[(small_high['block_no'] > 4) & (small_high['block_no'] <= 8)], small_high[(small_high['block_no'] > 8) & (small_high['block_no'] <= 12)], small_high[(small_high['block_no'] > 12) & (small_high['block_no'] <= 16)], small_high[(small_high['block_no'] > 16) & (small_high['block_no'] <= 20)]]
	small_high_errors = [(small_high_bins[0].shape[0]/count_distractor)*100,  (small_high_bins[1].shape[0]/count_distractor)*100,  (small_high_bins[2].shape[0]/count_distractor)*100,  (small_high_bins[3].shape[0]/count_distractor)*100,  (small_high_bins[4].shape[0]/count_distractor)*100]
	
	#Low
	small_low_bins = [small_low[(small_low['block_no'] > 0) & (small_low['block_no'] <= 4)],small_low[(small_low['block_no'] > 4) & (small_low['block_no'] <= 8)], small_low[(small_low['block_no'] > 8) & (small_low['block_no'] <= 12)], small_low[(small_low['block_no'] > 12) & (small_low['block_no'] <= 16)], small_low[(small_low['block_no'] > 16) & (small_low['block_no'] <= 20)]]
	small_low_errors = [(small_low_bins[0].shape[0]/count_distractor)*100,  (small_low_bins[1].shape[0]/count_distractor)*100,  (small_low_bins[2].shape[0]/count_distractor)*100,  (small_low_bins[3].shape[0]/count_distractor)*100,  (small_low_bins[4].shape[0]/count_distractor)*100]	
	
	#No
	small_no_bins = [small_no[(small_no['block_no'] > 0) & (small_no['block_no'] <= 4)],small_no[(small_no['block_no'] > 4) & (small_no['block_no'] <= 8)], small_no[(small_no['block_no'] > 8) & (small_no['block_no'] <= 12)], small_no[(small_no['block_no'] > 12) & (small_no['block_no'] <= 16)], small_no[(small_no['block_no'] > 16) & (small_no['block_no'] <= 20)]]
	small_no_errors = [(small_no_bins[0].shape[0]/count_distractor)*100,  (small_no_bins[1].shape[0]/count_distractor)*100,  (small_no_bins[2].shape[0]/count_distractor)*100,  (small_no_bins[3].shape[0]/count_distractor)*100,  (small_no_bins[4].shape[0]/count_distractor)*100]	

	small_high_no_diff_errors = list(np.array(small_high_errors) - np.array(small_no_errors))
	small_high_low_diff_errors = list(np.array(small_high_errors) - np.array(small_low_errors))
	
	### MEDIUM ANGLE
	
	#High reward
	med_high_bins = [med_high[(med_high['block_no'] > 0) & (med_high['block_no'] <= 4)],med_high[(med_high['block_no'] > 4) & (med_high['block_no'] <= 8)], med_high[(med_high['block_no'] > 8) & (med_high['block_no'] <= 12)], med_high[(med_high['block_no'] > 12) & (med_high['block_no'] <= 16)], med_high[(med_high['block_no'] > 16) & (med_high['block_no'] <= 20)]]
	med_high_errors = [(med_high_bins[0].shape[0]/count_distractor)*100,  (med_high_bins[1].shape[0]/count_distractor)*100,  (med_high_bins[2].shape[0]/count_distractor)*100,  (med_high_bins[3].shape[0]/count_distractor)*100,  (med_high_bins[4].shape[0]/count_distractor)*100]
	
	#Low
	med_low_bins = [med_low[(med_low['block_no'] > 0) & (med_low['block_no'] <= 4)],med_low[(med_low['block_no'] > 4) & (med_low['block_no'] <= 8)], med_low[(med_low['block_no'] > 8) & (med_low['block_no'] <= 12)], med_low[(med_low['block_no'] > 12) & (med_low['block_no'] <= 16)], med_low[(med_low['block_no'] > 16) & (med_low['block_no'] <= 20)]]
	med_low_errors = [(med_low_bins[0].shape[0]/count_distractor)*100,  (med_low_bins[1].shape[0]/count_distractor)*100,  (med_low_bins[2].shape[0]/count_distractor)*100,  (med_low_bins[3].shape[0]/count_distractor)*100,  (med_low_bins[4].shape[0]/count_distractor)*100]
	
	#No
	med_no_bins = [med_no[(med_no['block_no'] > 0) & (med_no['block_no'] <= 4)],med_no[(med_no['block_no'] > 4) & (med_no['block_no'] <= 8)], med_no[(med_no['block_no'] > 8) & (med_no['block_no'] <= 12)], med_no[(med_no['block_no'] > 12) & (med_no['block_no'] <= 16)], med_no[(med_no['block_no'] > 16) & (med_no['block_no'] <= 20)]]
	med_no_errors = [(med_no_bins[0].shape[0]/count_distractor)*100,  (med_no_bins[1].shape[0]/count_distractor)*100,  (med_no_bins[2].shape[0]/count_distractor)*100,  (med_no_bins[3].shape[0]/count_distractor)*100,  (med_no_bins[4].shape[0]/count_distractor)*100]

	med_high_no_diff_errors = list(np.array(med_high_errors) - np.array(med_no_errors))
	med_high_low_diff_errors = list(np.array(med_high_errors) - np.array(med_low_errors))

	### LARGE ANGLE
	
	#High reward
	large_high_bins = [large_high[(large_high['block_no'] > 0) & (large_high['block_no'] <= 4)],large_high[(large_high['block_no'] > 4) & (large_high['block_no'] <= 8)], large_high[(large_high['block_no'] > 8) & (large_high['block_no'] <= 12)], large_high[(large_high['block_no'] > 12) & (large_high['block_no'] <= 16)], large_high[(large_high['block_no'] > 16) & (large_high['block_no'] <= 20)]]
	large_high_errors = [(large_high_bins[0].shape[0]/count_distractor)*100,  (large_high_bins[1].shape[0]/count_distractor)*100,  (large_high_bins[2].shape[0]/count_distractor)*100,  (large_high_bins[3].shape[0]/count_distractor)*100,  (large_high_bins[4].shape[0]/count_distractor)*100]

	#Low
	large_low_bins = [large_low[(large_low['block_no'] > 0) & (large_low['block_no'] <= 4)],large_low[(large_low['block_no'] > 4) & (large_low['block_no'] <= 8)], large_low[(large_low['block_no'] > 8) & (large_low['block_no'] <= 12)], large_low[(large_low['block_no'] > 12) & (large_low['block_no'] <= 16)], large_low[(large_low['block_no'] > 16) & (large_low['block_no'] <= 20)]]
	large_low_errors = [(large_low_bins[0].shape[0]/count_distractor)*100,  (large_low_bins[1].shape[0]/count_distractor)*100,  (large_low_bins[2].shape[0]/count_distractor)*100,  (large_low_bins[3].shape[0]/count_distractor)*100,  (large_low_bins[4].shape[0]/count_distractor)*100]
		
	#No
	large_no_bins = [large_no[(large_no['block_no'] > 0) & (large_no['block_no'] <= 4)],large_no[(large_no['block_no'] > 4) & (large_no['block_no'] <= 8)], large_no[(large_no['block_no'] > 8) & (large_no['block_no'] <= 12)], large_no[(large_no['block_no'] > 12) & (large_no['block_no'] <= 16)], large_no[(large_no['block_no'] > 16) & (large_no['block_no'] <= 20)]]
	large_no_errors = [(large_no_bins[0].shape[0]/count_distractor)*100,  (large_no_bins[1].shape[0]/count_distractor)*100,  (large_no_bins[2].shape[0]/count_distractor)*100,  (large_no_bins[3].shape[0]/count_distractor)*100,  (large_no_bins[4].shape[0]/count_distractor)*100]
	
	large_high_no_diff_errors = list(np.array(large_high_errors) - np.array(large_no_errors))
	large_high_low_diff_errors = list(np.array(large_high_errors) - np.array(large_low_errors))
		
	# PLOTS (no errorbars - percentage within individuals is an absolute value)	
	# High-No difference
	x=np.arange(1,6)
	plt.xlim(xmin=0.5, xmax = 5.5)
	plt.title('Saccade error difference between high and no reward across bins')
	plt.xlabel('Block')
	plt.ylabel('Saccade error difference (%)')
	plt.plot(x,small_high_no_diff_errors,'g', label = 'Small angle')
	plt.plot(x,med_high_no_diff_errors,'b',label = 'Medium angle')
	plt.plot(x,large_high_no_diff_errors,'r',label = 'Large angle')
	plt.legend(loc='best')
	plt.savefig('errors_bins_hn_angles.png')
	#plt.show()
	plt.close()
	
	# High-Low difference
	x=np.arange(1,6)
	plt.xlim(xmin=0.5, xmax = 5.5)
	plt.title('Saccade error difference between high and low reward across bins')
	plt.xlabel('Block')
	plt.ylabel('Saccade error difference (%)')
	plt.plot(x,small_high_low_diff_errors,'g', label = 'Small angle')
	plt.plot(x,med_high_low_diff_errors,'b',label = 'Medium angle')
	plt.plot(x,large_high_low_diff_errors,'r',label = 'Large angle')
	plt.legend(loc='best')
	plt.savefig('errors_bins_hl_angles.png')
	#plt.show()
	plt.close()
		
	# HEMIFIELD
	
	# SAME 
	
	#High reward
	samehemi_high_bins = [samehemi_high[(samehemi_high['block_no'] > 0) & (samehemi_high['block_no'] <= 4)],samehemi_high[(samehemi_high['block_no'] > 4) & (samehemi_high['block_no'] <= 8)], samehemi_high[(samehemi_high['block_no'] > 8) & (samehemi_high['block_no'] <= 12)], samehemi_high[(samehemi_high['block_no'] > 12) & (samehemi_high['block_no'] <= 16)], samehemi_high[(samehemi_high['block_no'] > 16) & (samehemi_high['block_no'] <= 20)]]
	samehemi_high_errors = [(samehemi_high_bins[0].shape[0]/count_distractor)*100,  (samehemi_high_bins[1].shape[0]/count_distractor)*100,  (samehemi_high_bins[2].shape[0]/count_distractor)*100,  (samehemi_high_bins[3].shape[0]/count_distractor)*100,  (samehemi_high_bins[4].shape[0]/count_distractor)*100]

	#Low
	samehemi_low_bins = [samehemi_low[(samehemi_low['block_no'] > 0) & (samehemi_low['block_no'] <= 4)],samehemi_low[(samehemi_low['block_no'] > 4) & (samehemi_low['block_no'] <= 8)], samehemi_low[(samehemi_low['block_no'] > 8) & (samehemi_low['block_no'] <= 12)], samehemi_low[(samehemi_low['block_no'] > 12) & (samehemi_low['block_no'] <= 16)], samehemi_low[(samehemi_low['block_no'] > 16) & (samehemi_low['block_no'] <= 20)]]
	samehemi_low_errors = [(samehemi_low_bins[0].shape[0]/count_distractor)*100,  (samehemi_low_bins[1].shape[0]/count_distractor)*100,  (samehemi_low_bins[2].shape[0]/count_distractor)*100,  (samehemi_low_bins[3].shape[0]/count_distractor)*100,  (samehemi_low_bins[4].shape[0]/count_distractor)*100]
		
	#No
	samehemi_no_bins = [samehemi_no[(samehemi_no['block_no'] > 0) & (samehemi_no['block_no'] <= 4)],samehemi_no[(samehemi_no['block_no'] > 4) & (samehemi_no['block_no'] <= 8)], samehemi_no[(samehemi_no['block_no'] > 8) & (samehemi_no['block_no'] <= 12)], samehemi_no[(samehemi_no['block_no'] > 12) & (samehemi_no['block_no'] <= 16)], samehemi_no[(samehemi_no['block_no'] > 16) & (samehemi_no['block_no'] <= 20)]]
	samehemi_no_errors = [(samehemi_no_bins[0].shape[0]/count_distractor)*100,  (samehemi_no_bins[1].shape[0]/count_distractor)*100,  (samehemi_no_bins[2].shape[0]/count_distractor)*100,  (samehemi_no_bins[3].shape[0]/count_distractor)*100,  (samehemi_no_bins[4].shape[0]/count_distractor)*100]
	
	# OPPOSITE
	
	#High reward
	opphemi_high_bins = [opphemi_high[(opphemi_high['block_no'] > 0) & (opphemi_high['block_no'] <= 4)],opphemi_high[(opphemi_high['block_no'] > 4) & (opphemi_high['block_no'] <= 8)], opphemi_high[(opphemi_high['block_no'] > 8) & (opphemi_high['block_no'] <= 12)], opphemi_high[(opphemi_high['block_no'] > 12) & (opphemi_high['block_no'] <= 16)], opphemi_high[(opphemi_high['block_no'] > 16) & (opphemi_high['block_no'] <= 20)]]
	opphemi_high_errors = [(opphemi_high_bins[0].shape[0]/count_distractor)*100,  (opphemi_high_bins[1].shape[0]/count_distractor)*100,  (opphemi_high_bins[2].shape[0]/count_distractor)*100,  (opphemi_high_bins[3].shape[0]/count_distractor)*100,  (opphemi_high_bins[4].shape[0]/count_distractor)*100]

	#Low
	opphemi_low_bins = [opphemi_low[(opphemi_low['block_no'] > 0) & (opphemi_low['block_no'] <= 4)],opphemi_low[(opphemi_low['block_no'] > 4) & (opphemi_low['block_no'] <= 8)], opphemi_low[(opphemi_low['block_no'] > 8) & (opphemi_low['block_no'] <= 12)], opphemi_low[(opphemi_low['block_no'] > 12) & (opphemi_low['block_no'] <= 16)], opphemi_low[(opphemi_low['block_no'] > 16) & (opphemi_low['block_no'] <= 20)]]
	opphemi_low_errors = [(opphemi_low_bins[0].shape[0]/count_distractor)*100,  (opphemi_low_bins[1].shape[0]/count_distractor)*100,  (opphemi_low_bins[2].shape[0]/count_distractor)*100,  (opphemi_low_bins[3].shape[0]/count_distractor)*100,  (opphemi_low_bins[4].shape[0]/count_distractor)*100]
		
	#No
	opphemi_no_bins = [opphemi_no[(opphemi_no['block_no'] > 0) & (opphemi_no['block_no'] <= 4)],opphemi_no[(opphemi_no['block_no'] > 4) & (opphemi_no['block_no'] <= 8)], opphemi_no[(opphemi_no['block_no'] > 8) & (opphemi_no['block_no'] <= 12)], opphemi_no[(opphemi_no['block_no'] > 12) & (opphemi_no['block_no'] <= 16)], opphemi_no[(opphemi_no['block_no'] > 16) & (opphemi_no['block_no'] <= 20)]]
	opphemi_no_errors = [(opphemi_no_bins[0].shape[0]/count_distractor)*100,  (opphemi_no_bins[1].shape[0]/count_distractor)*100,  (opphemi_no_bins[2].shape[0]/count_distractor)*100,  (opphemi_no_bins[3].shape[0]/count_distractor)*100,  (opphemi_no_bins[4].shape[0]/count_distractor)*100]

	if hemifield == 0:
		return [small_high_errors[0],small_high_errors[1],small_high_errors[2],small_high_errors[3],small_high_errors[4],small_low_errors[0],small_low_errors[1],small_low_errors[2],small_low_errors[3],small_low_errors[4],small_no_errors[0],small_no_errors[1],small_no_errors[2],small_no_errors[3],small_no_errors[4],med_high_errors[0],med_high_errors[1],med_high_errors[2],med_high_errors[3],med_high_errors[4],med_low_errors[0],med_low_errors[1],med_low_errors[2],med_low_errors[3],med_low_errors[4],med_no_errors[0],med_no_errors[1],med_no_errors[2],med_no_errors[3],med_no_errors[4],large_high_errors[0],large_high_errors[1],large_high_errors[2],large_high_errors[3],large_high_errors[4],large_low_errors[0],large_low_errors[1],large_low_errors[2],large_low_errors[3],large_low_errors[4],large_no_errors[0],large_no_errors[1],large_no_errors[2],large_no_errors[3],large_no_errors[4]]
	elif hemifield == 1:
		return [samehemi_high_errors[0],samehemi_high_errors[1],samehemi_high_errors[2],samehemi_high_errors[3],samehemi_high_errors[4],samehemi_low_errors[0],samehemi_low_errors[1],samehemi_low_errors[2],samehemi_low_errors[3],samehemi_low_errors[4],samehemi_no_errors[0],samehemi_no_errors[1],samehemi_no_errors[2],samehemi_no_errors[3],samehemi_no_errors[4],opphemi_high_errors[0],opphemi_high_errors[1],opphemi_high_errors[2],opphemi_high_errors[3],opphemi_high_errors[4],opphemi_low_errors[0],opphemi_low_errors[1],opphemi_low_errors[2],opphemi_low_errors[3],opphemi_low_errors[4],opphemi_no_errors[0],opphemi_no_errors[1],opphemi_no_errors[2],opphemi_no_errors[3],opphemi_no_errors[4]]
	
def distance_from_cog():
	# For all correct trials, calculates the percentage closer to the midpoint (cog)
	
	# Create dictionaries to append to dataframe (this method is quicker than adding two new Series for each of the cogs)
	cog_dict = {}
	dist_cog_dict = {} 
	dist_eye_dict = {}
	
	# Create columns cog_x, cog_y (this is the midpoint of target and distractor)
	for i in range(len(trials)):
		cog_dict[i] = {'cog_x': np.mean([trials.iloc[i]['t_x'],trials.iloc[i]['d_x']]), 'cog_y' : np.mean([trials.iloc[i]['t_y'],trials.iloc[i]['d_y']])}
			
	# Turn dict into a dataframe
	cog_dict = pd.DataFrame.from_dict(cog_dict, orient = 'index', dtype = float)	
	trials.index = range(len(trials))
	cog_trials = pd.concat([trials, cog_dict], axis=1)
	
	for i in range(len(cog_trials)):
		dist_cog_dict[i] = {'dist_target_cog' : np.sqrt((cog_trials.iloc[i]['cog_x'] - cog_trials.iloc[i]['t_x'])**2 + (cog_trials.iloc[i]['cog_y'] - cog_trials.iloc[i]['t_y'])**2)}

	dist_cog_dict = pd.DataFrame.from_dict(dist_cog_dict, orient = 'index', dtype = float)
	cog_trials.index = range(len(cog_trials))
	cog_trials = pd.concat([cog_trials, dist_cog_dict], axis=1)
	
	for i in range(len(cog_trials)):
		dist_eye_dict[i] = {'dist_eye_cog' : np.sqrt((cog_trials.iloc[i]['cog_x'] - cog_trials.iloc[i]['saccade_endx_target'])**2 + (cog_trials.iloc[i]['cog_y'] - cog_trials.iloc[i]['saccade_endy_target'])**2)}

	dist_eye_dict = pd.DataFrame.from_dict(dist_eye_dict, orient = 'index', dtype = float)
	cog_trials.index = range(len(cog_trials))
	cog_trials = pd.concat([cog_trials, dist_eye_dict], axis=1)
	
	# Only taking 30 degree distractor trials (would not expect global effects for the larger angles)
	cog_trials = cog_trials[cog_trials['distractor_angle_deg']==29]
	
	median_latency = np.median(cog_trials['saccade_onset_target'])
	early_cog_trials = cog_trials[cog_trials['saccade_onset_target']< median_latency]
	late_cog_trials = cog_trials[cog_trials['saccade_onset_target']> median_latency]
	
	#High
	high_cog_1 = early_cog_trials[early_cog_trials['reward_available']==0]
	high_cog_2 = late_cog_trials[late_cog_trials['reward_available']==0]
	
	percent_high_1 = np.mean(((high_cog_1['dist_target_cog'] - high_cog_1['dist_eye_cog'])/high_cog_1['dist_target_cog'])*100)
	percent_high_2 = np.mean(((high_cog_2['dist_target_cog'] - high_cog_2['dist_eye_cog'])/high_cog_2['dist_target_cog'])*100)
	
	#Low
	low_cog_1 = early_cog_trials[early_cog_trials['reward_available']==1]
	low_cog_2 = late_cog_trials[late_cog_trials['reward_available']==1]
	
	percent_low_1 = np.mean(((low_cog_1['dist_target_cog'] - low_cog_1['dist_eye_cog'])/low_cog_1['dist_target_cog'])*100)
	percent_low_2 = np.mean(((low_cog_2['dist_target_cog'] - low_cog_2['dist_eye_cog'])/low_cog_2['dist_target_cog'])*100)
	
	#No
	no_cog_1 = early_cog_trials[early_cog_trials['reward_available']==2]
	no_cog_2 = late_cog_trials[late_cog_trials['reward_available']==2]
	
	percent_no_1 = np.mean(((no_cog_1['dist_target_cog'] - no_cog_1['dist_eye_cog'])/no_cog_1['dist_target_cog'])*100)
	percent_no_2 = np.mean(((no_cog_2['dist_target_cog'] - no_cog_2['dist_eye_cog'])/no_cog_2['dist_target_cog'])*100)
	
	# Percentage closer to midpoint (cog) : timebin (2) x reward (3)
	return [percent_high_1, percent_low_1, percent_no_1, percent_high_2, percent_low_2, percent_no_2]
	
def saccade_evolution():
	# Takes all correct and error trials, and finds percentage to each target and distractor ROI, across latency quartiles.
	
	# all trials
	new_trials = trials
	
	latency_dict = {}
	
	for i in range(len(new_trials)):
		if new_trials.iloc[i]['saccade_onset_target'] > 0:
			latency_dict[i] = {'latency_target_or_distractor': new_trials.iloc[i]['saccade_onset_target']}
		elif new_trials.iloc[i]['saccade_onset_distractor'] > 0:
			latency_dict[i] = {'latency_target_or_distractor': new_trials.iloc[i]['saccade_onset_distractor']}
			
	# Turn dict into a dataframe
	latency_dict = pd.DataFrame.from_dict(latency_dict, orient = 'index', dtype = float)	
	new_trials.index = range(len(new_trials))
	latency_trials = pd.concat([new_trials, latency_dict], axis=1)
	
	distractor_latencies = latency_trials[latency_trials['which_circle'] == 'd']
	distractor_total = distractor_latencies.shape[0]	
	
	# High distractor
	high_distractor = distractor_latencies[distractor_latencies['reward_available']==0]
	
	if high_distractor.shape[0]!= 0:
		
		cutoff_high_1 = np.percentile(high_distractor['latency_target_or_distractor'], 25)
		cutoff_high_2 = np.percentile(high_distractor['latency_target_or_distractor'], 50)
		cutoff_high_3 = np.percentile(high_distractor['latency_target_or_distractor'], 75)

	# Low distractor
	low_distractor = distractor_latencies[distractor_latencies['reward_available']==1]
	
	if low_distractor.shape[0]!= 0:
		cutoff_low_1 = np.percentile(low_distractor['latency_target_or_distractor'], 25)
		cutoff_low_2 = np.percentile(low_distractor['latency_target_or_distractor'], 50)
		cutoff_low_3 = np.percentile(low_distractor['latency_target_or_distractor'], 75)	
	
	# No distractor
	no_distractor = distractor_latencies[distractor_latencies['reward_available']==2]
	
	if no_distractor.shape[0]!= 0:
		cutoff_no_1 = np.percentile(no_distractor['latency_target_or_distractor'], 25)
		cutoff_no_2 = np.percentile(no_distractor['latency_target_or_distractor'], 50)
		cutoff_no_3 = np.percentile(no_distractor['latency_target_or_distractor'], 75)	
	
	print 'distractor total: ', [high_distractor.shape[0]+low_distractor.shape[0]+no_distractor.shape[0]]
	
	# High - latency quartiles
	if high_distractor.shape[0]!= 0:
		high_latbin_1 = high_distractor[high_distractor['latency_target_or_distractor'] < cutoff_high_1] # 25th percentile
		high_latbin_2 = high_distractor[(high_distractor['latency_target_or_distractor'] < cutoff_high_2) & (high_distractor['latency_target_or_distractor'] >= cutoff_high_1)] # 50th
		high_latbin_3 = high_distractor[(high_distractor['latency_target_or_distractor'] < cutoff_high_3) & (high_distractor['latency_target_or_distractor'] >= cutoff_high_2)] #75th
		high_latbin_4 = high_distractor[high_distractor['latency_target_or_distractor'] >= cutoff_high_3] # 100th percentile
	else:
		# Create empty dataframe so that .shape[0] will be 0 later
		high_latbin_1, high_latbin_2, high_latbin_3, high_latbin_4 = [pd.DataFrame([]) for i in range(4)]

	# Low - latency quartiles
	if low_distractor.shape[0]!= 0:
		low_latbin_1 = low_distractor[low_distractor['latency_target_or_distractor'] < cutoff_low_1] # 25th percentile
		low_latbin_2 = low_distractor[(low_distractor['latency_target_or_distractor'] < cutoff_low_2) & (low_distractor['latency_target_or_distractor'] >= cutoff_low_1)] # 50th
		low_latbin_3 = low_distractor[(low_distractor['latency_target_or_distractor'] < cutoff_low_3) & (low_distractor['latency_target_or_distractor'] >= cutoff_low_2)] #75th
		low_latbin_4 = low_distractor[low_distractor['latency_target_or_distractor'] >= cutoff_low_3] # 100th percentile	
	else:
		# Create empty dataframe so that .shape[0] will be 0 later
		low_latbin_1, low_latbin_2, low_latbin_3, low_latbin_4 = [pd.DataFrame([]) for i in range(4)]		

	# No - latency quartiles
	if no_distractor.shape[0]!= 0:
		no_latbin_1 = no_distractor[no_distractor['latency_target_or_distractor'] < cutoff_no_1] # 25th percentile
		no_latbin_2 = no_distractor[(no_distractor['latency_target_or_distractor'] < cutoff_no_2) & (no_distractor['latency_target_or_distractor'] >= cutoff_no_1)] # 50th
		no_latbin_3 = no_distractor[(no_distractor['latency_target_or_distractor'] < cutoff_no_3) & (no_distractor['latency_target_or_distractor'] >= cutoff_no_2)] #75th
		no_latbin_4 = no_distractor[no_distractor['latency_target_or_distractor'] >= cutoff_no_3] # 100th percentile	
	else:
		# Create empty dataframe so that .shape[0] will be 0 later
		no_latbin_1, no_latbin_2, no_latbin_3, no_latbin_4 = [pd.DataFrame([]) for i in range(4)]		

	# Total percentages per reward level and latency bin
	distractor_percentage_high_1 = (high_latbin_1.shape[0]/distractor_total) * 100
	distractor_percentage_high_2 = (high_latbin_2.shape[0]/distractor_total) * 100
	distractor_percentage_high_3 = (high_latbin_3.shape[0]/distractor_total) * 100
	distractor_percentage_high_4 = (high_latbin_4.shape[0]/distractor_total) * 100
	
	distractor_percentage_low_1 = (low_latbin_1.shape[0]/distractor_total) * 100
	distractor_percentage_low_2 = (low_latbin_2.shape[0]/distractor_total) * 100
	distractor_percentage_low_3 = (low_latbin_3.shape[0]/distractor_total) * 100
	distractor_percentage_low_4 = (low_latbin_4.shape[0]/distractor_total) * 100
	
	distractor_percentage_no_1 = (no_latbin_1.shape[0]/distractor_total) * 100
	distractor_percentage_no_2 = (no_latbin_2.shape[0]/distractor_total) * 100
	distractor_percentage_no_3 = (no_latbin_3.shape[0]/distractor_total) * 100
	distractor_percentage_no_4 = (no_latbin_4.shape[0]/distractor_total) * 100
	
	# Mean latencies per reward level and latency bin
	if high_distractor.shape[0]!= 0:
		distractor_mean_high_1 = np.mean(high_latbin_1['latency_target_or_distractor'])
		distractor_mean_high_2 = np.mean(high_latbin_2['latency_target_or_distractor'])
		distractor_mean_high_3 = np.mean(high_latbin_3['latency_target_or_distractor'])
		distractor_mean_high_4 = np.mean(high_latbin_4['latency_target_or_distractor'])
	else:
		distractor_mean_high_1, distractor_mean_high_2, distractor_mean_high_3, distractor_mean_high_4 = [0 for i in range(4)]
	
	if low_distractor.shape[0]!= 0:
		distractor_mean_low_1 = np.mean(low_latbin_1['latency_target_or_distractor'])
		distractor_mean_low_2 = np.mean(low_latbin_2['latency_target_or_distractor'])
		distractor_mean_low_3 = np.mean(low_latbin_3['latency_target_or_distractor'])
		distractor_mean_low_4 = np.mean(low_latbin_4['latency_target_or_distractor'])	
	else:
		distractor_mean_low_1, distractor_mean_low_2, distractor_mean_low_3, distractor_mean_low_4 = [0 for i in range(4)]
	
	if no_distractor.shape[0]!= 0:
		distractor_mean_no_1 = np.mean(no_latbin_1['latency_target_or_distractor'])
		distractor_mean_no_2 = np.mean(no_latbin_2['latency_target_or_distractor'])
		distractor_mean_no_3 = np.mean(no_latbin_3['latency_target_or_distractor'])
		distractor_mean_no_4 = np.mean(no_latbin_4['latency_target_or_distractor'])		
	else:
		distractor_mean_no_1, distractor_mean_no_2, distractor_mean_no_3, distractor_mean_no_4 = [0 for i in range(4)]		
		
	print 'total #: ', [distractor_percentage_high_1+distractor_percentage_high_2+distractor_percentage_high_3+distractor_percentage_high_4+distractor_percentage_low_1+distractor_percentage_low_2+distractor_percentage_low_3+distractor_percentage_low_4 + distractor_percentage_no_1+distractor_percentage_no_2+distractor_percentage_no_3+distractor_percentage_no_4]
	
	return [distractor_percentage_high_1, distractor_percentage_low_1, distractor_percentage_no_1, distractor_percentage_high_2, distractor_percentage_low_2, distractor_percentage_no_2, distractor_percentage_high_3, distractor_percentage_low_3, distractor_percentage_no_3, distractor_percentage_high_4, distractor_percentage_low_4, distractor_percentage_no_4, distractor_mean_high_1, distractor_mean_low_1,distractor_mean_no_1,distractor_mean_high_2,distractor_mean_low_2,distractor_mean_no_2,distractor_mean_high_3,distractor_mean_low_3,distractor_mean_no_3,distractor_mean_high_4,distractor_mean_low_4,distractor_mean_no_4]
	

def curvature_absolute(sb,hemifield):
	
	# Angle
	mean_small_high = np.mean(med_high['curvature'])
	mean_small_low = np.mean(small_low['curvature'])
	mean_small_no = np.mean(small_no['curvature'])
	
	mean_med_high = np.mean(med_high['curvature'])
	mean_med_low = np.mean(med_low['curvature'])
	mean_med_no = np.mean(med_no['curvature'])
	
	mean_large_high = np.mean(large_high['curvature'])
	mean_large_low = np.mean(large_low['curvature'])
	mean_large_no = np.mean(large_no['curvature'])
	
	# Hemifield
	mean_samehemi_high = np.mean(samehemi_high['curvature'])
	mean_samehemi_low = np.mean(samehemi_low['curvature'])
	mean_samehemi_no = np.mean(samehemi_no['curvature'])
	
	mean_opphemi_high = np.mean(opphemi_high['curvature'])
	mean_opphemi_low = np.mean(opphemi_low['curvature'])
	mean_opphemi_no = np.mean(opphemi_no['curvature'])	
	
	if hemifield == 0:
		return [mean_small_high, mean_small_low, mean_small_no,  mean_med_high, mean_med_low, mean_med_no, mean_large_high, mean_large_low, mean_large_no]
	elif hemifield == 1:
		return [mean_samehemi_high, mean_samehemi_low, mean_samehemi_no, mean_opphemi_high, mean_opphemi_low, mean_opphemi_no]
		return [mean_samehemi_high, mean_samehemi_low, mean_samehemi_no, mean_opphemi_high, mean_opphemi_low, mean_opphemi_no]

def peak_velocity(hemifield):

	# Angle
	small_high_vel = np.mean(small_high['CURRENT_SAC_PEAK_VELOCITY'])
	small_low_vel = np.mean(small_low['CURRENT_SAC_PEAK_VELOCITY'])
	small_no_vel = np.mean(small_no['CURRENT_SAC_PEAK_VELOCITY'])
	
	med_high_vel = np.mean(med_high['CURRENT_SAC_PEAK_VELOCITY']) 
	med_low_vel = np.mean(med_low['CURRENT_SAC_PEAK_VELOCITY'])
	med_no_vel = np.mean(med_no['CURRENT_SAC_PEAK_VELOCITY'])
	
	large_high_vel = np.mean(large_high['CURRENT_SAC_PEAK_VELOCITY']) 
	large_low_vel = np.mean(large_low['CURRENT_SAC_PEAK_VELOCITY']) 
	large_no_vel = np.mean(large_no['CURRENT_SAC_PEAK_VELOCITY']) 
	
	# Hemifield
	samehemi_high_vel = np.mean(samehemi_high['CURRENT_SAC_PEAK_VELOCITY'])
	samehemi_low_vel = np.mean(samehemi_low['CURRENT_SAC_PEAK_VELOCITY'])
	samehemi_no_vel = np.mean(samehemi_no['CURRENT_SAC_PEAK_VELOCITY'])
	
	opphemi_high_vel = np.mean(opphemi_high['CURRENT_SAC_PEAK_VELOCITY'])
	opphemi_low_vel = np.mean(opphemi_low['CURRENT_SAC_PEAK_VELOCITY'])
	opphemi_no_vel = np.mean(opphemi_no['CURRENT_SAC_PEAK_VELOCITY'])
	
	if hemifield == 0:
		return [small_high_vel, small_low_vel, small_no_vel, med_high_vel, med_low_vel, med_no_vel, large_high_vel, large_low_vel, large_no_vel]
	elif hemifield == 1:
		return [samehemi_high_vel, samehemi_low_vel, samehemi_no_vel, opphemi_high_vel, opphemi_low_vel, opphemi_no_vel]


def intertrial_priming(hemifield, angle = 'all'):
	
	# All trials per reward level repetition
	all_trials = trials
	
	all_hh_rep = []
	all_hl_rep = []
	all_hn_rep = []
	
	all_lh_rep = []
	all_ll_rep = []
	all_ln_rep = []
	
	all_nh_rep = []
	all_nl_rep = []
	all_nn_rep = []
	
	for i in range(len(all_trials)):
		if ((all_trials['trial_no'].iloc[i] == (all_trials['trial_no'].iloc[i-1]+1)) & (all_trials['which_circle'].iloc[i-1] == 't')):
			if all_trials['reward_available'].iloc[i-1] == 0: # if previous trial was high reward
				if all_trials['reward_available'].iloc[i] == 0: #if current trial is high reward
					all_hh_rep.append([all_trials['trial_no'].iloc[i],all_trials['saccade_onset_target'].iloc[i], all_trials['reward_available'].iloc[i], all_trials['distractor_angle_deg'].iloc[i]])
				elif (all_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					all_hl_rep.append([all_trials['trial_no'].iloc[i],all_trials['saccade_onset_target'].iloc[i], all_trials['reward_available'].iloc[i], all_trials['distractor_angle_deg'].iloc[i]])
				elif (all_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					all_hn_rep.append([all_trials['trial_no'].iloc[i],all_trials['saccade_onset_target'].iloc[i], all_trials['reward_available'].iloc[i], all_trials['distractor_angle_deg'].iloc[i]])
			elif all_trials['reward_available'].iloc[i-1] == 1: # if previous trial was low reward
				if (all_trials['reward_available'].iloc[i] == 0): #if current trial is high reward
					all_lh_rep.append([all_trials['trial_no'].iloc[i],all_trials['saccade_onset_target'].iloc[i], all_trials['reward_available'].iloc[i], all_trials['distractor_angle_deg'].iloc[i]])
				elif (all_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					all_ll_rep.append([all_trials['trial_no'].iloc[i],all_trials['saccade_onset_target'].iloc[i], all_trials['reward_available'].iloc[i], all_trials['distractor_angle_deg'].iloc[i]])
				elif (all_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					all_ln_rep.append([all_trials['trial_no'].iloc[i],all_trials['saccade_onset_target'].iloc[i], all_trials['reward_available'].iloc[i], all_trials['distractor_angle_deg'].iloc[i]])
			elif all_trials['reward_available'].iloc[i-1] == 2: # if previous trial was no reward
				if (all_trials['reward_available'].iloc[i] == 0): #if current trial is high reward
					all_nh_rep.append([all_trials['trial_no'].iloc[i],all_trials['saccade_onset_target'].iloc[i], all_trials['reward_available'].iloc[i], all_trials['distractor_angle_deg'].iloc[i]])
				elif (all_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					all_nl_rep.append([all_trials['trial_no'].iloc[i],all_trials['saccade_onset_target'].iloc[i], all_trials['reward_available'].iloc[i], all_trials['distractor_angle_deg'].iloc[i]])
				elif (all_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					all_nn_rep.append([all_trials['trial_no'].iloc[i],all_trials['saccade_onset_target'].iloc[i], all_trials['reward_available'].iloc[i], all_trials['distractor_angle_deg'].iloc[i]])

	#30 deg distractor trials per reward repetition
	small_trials = trials[trials['distractor_angle_deg']==29]
	
	small_hh_rep = []
	small_hl_rep = []
	small_hn_rep = []
	
	small_lh_rep = []
	small_ll_rep = []
	small_ln_rep = []
	
	small_nh_rep = []
	small_nl_rep = []
	small_nn_rep = []
	
	for i in range(len(small_trials)):
		if ((small_trials['trial_no'].iloc[i] == (small_trials['trial_no'].iloc[i-1]+1)) & (small_trials['which_circle'].iloc[i-1] == 't')):
			if small_trials['reward_available'].iloc[i-1] == 0: # if previous trial was high reward
				if small_trials['reward_available'].iloc[i] == 0: #if current trial is high reward
					small_hh_rep.append([small_trials['trial_no'].iloc[i],small_trials['saccade_onset_target'].iloc[i], small_trials['reward_available'].iloc[i], small_trials['distractor_angle_deg'].iloc[i]])
				elif (small_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					small_hl_rep.append([small_trials['trial_no'].iloc[i],small_trials['saccade_onset_target'].iloc[i], small_trials['reward_available'].iloc[i], small_trials['distractor_angle_deg'].iloc[i]])
				elif (small_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					small_hn_rep.append([small_trials['trial_no'].iloc[i],small_trials['saccade_onset_target'].iloc[i], small_trials['reward_available'].iloc[i], small_trials['distractor_angle_deg'].iloc[i]])
			elif small_trials['reward_available'].iloc[i-1] == 1: # if previous trial was low reward
				if (small_trials['reward_available'].iloc[i] == 0): #if current trial is high reward
					small_lh_rep.append([small_trials['trial_no'].iloc[i],small_trials['saccade_onset_target'].iloc[i], small_trials['reward_available'].iloc[i], small_trials['distractor_angle_deg'].iloc[i]])
				elif (small_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					small_ll_rep.append([small_trials['trial_no'].iloc[i],small_trials['saccade_onset_target'].iloc[i], small_trials['reward_available'].iloc[i], small_trials['distractor_angle_deg'].iloc[i]])
				elif (small_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					small_ln_rep.append([small_trials['trial_no'].iloc[i],small_trials['saccade_onset_target'].iloc[i], small_trials['reward_available'].iloc[i], small_trials['distractor_angle_deg'].iloc[i]])
			elif small_trials['reward_available'].iloc[i-1] == 2: # if previous trial was no reward
				if (small_trials['reward_available'].iloc[i] == 0): #if current trial is high reward
					small_nh_rep.append([small_trials['trial_no'].iloc[i],small_trials['saccade_onset_target'].iloc[i], small_trials['reward_available'].iloc[i], small_trials['distractor_angle_deg'].iloc[i]])
				elif (small_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					small_nl_rep.append([small_trials['trial_no'].iloc[i],small_trials['saccade_onset_target'].iloc[i], small_trials['reward_available'].iloc[i], small_trials['distractor_angle_deg'].iloc[i]])
				elif (small_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					small_nn_rep.append([small_trials['trial_no'].iloc[i],small_trials['saccade_onset_target'].iloc[i], small_trials['reward_available'].iloc[i], small_trials['distractor_angle_deg'].iloc[i]])

	#120 deg distractor trials per reward repetition
	med_trials = trials[trials['distractor_angle_deg']==119]
	
	med_hh_rep = []
	med_hl_rep = []
	med_hn_rep = []
	
	med_lh_rep = []
	med_ll_rep = []
	med_ln_rep = []
	
	med_nh_rep = []
	med_nl_rep = []
	med_nn_rep = []
	
	for i in range(len(med_trials)):
		if ((med_trials['trial_no'].iloc[i] == (med_trials['trial_no'].iloc[i-1]+1)) & (med_trials['which_circle'].iloc[i-1] == 't')):
			if med_trials['reward_available'].iloc[i-1] == 0: # if previous trial was high reward
				if med_trials['reward_available'].iloc[i] == 0: #if current trial is high reward
					med_hh_rep.append([med_trials['trial_no'].iloc[i],med_trials['saccade_onset_target'].iloc[i], med_trials['reward_available'].iloc[i], med_trials['distractor_angle_deg'].iloc[i]])
				elif (med_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					med_hl_rep.append([med_trials['trial_no'].iloc[i],med_trials['saccade_onset_target'].iloc[i], med_trials['reward_available'].iloc[i], med_trials['distractor_angle_deg'].iloc[i]])
				elif (med_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					med_hn_rep.append([med_trials['trial_no'].iloc[i],med_trials['saccade_onset_target'].iloc[i], med_trials['reward_available'].iloc[i], med_trials['distractor_angle_deg'].iloc[i]])
			elif med_trials['reward_available'].iloc[i-1] == 1: # if previous trial was low reward
				if (med_trials['reward_available'].iloc[i] == 0): #if current trial is high reward
					med_lh_rep.append([med_trials['trial_no'].iloc[i],med_trials['saccade_onset_target'].iloc[i], med_trials['reward_available'].iloc[i], med_trials['distractor_angle_deg'].iloc[i]])
				elif (med_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					med_ll_rep.append([med_trials['trial_no'].iloc[i],med_trials['saccade_onset_target'].iloc[i], med_trials['reward_available'].iloc[i], med_trials['distractor_angle_deg'].iloc[i]])
				elif (med_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					med_ln_rep.append([med_trials['trial_no'].iloc[i],med_trials['saccade_onset_target'].iloc[i], med_trials['reward_available'].iloc[i], med_trials['distractor_angle_deg'].iloc[i]])
			elif med_trials['reward_available'].iloc[i-1] == 2: # if previous trial was no reward
				if (med_trials['reward_available'].iloc[i] == 0): #if current trial is high reward
					med_nh_rep.append([med_trials['trial_no'].iloc[i],med_trials['saccade_onset_target'].iloc[i], med_trials['reward_available'].iloc[i], med_trials['distractor_angle_deg'].iloc[i]])
				elif (med_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					med_nl_rep.append([med_trials['trial_no'].iloc[i],med_trials['saccade_onset_target'].iloc[i], med_trials['reward_available'].iloc[i], med_trials['distractor_angle_deg'].iloc[i]])
				elif (med_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					med_nn_rep.append([med_trials['trial_no'].iloc[i],med_trials['saccade_onset_target'].iloc[i], med_trials['reward_available'].iloc[i], med_trials['distractor_angle_deg'].iloc[i]])
	
	#180 deg distractor trials per reward repetition
	large_trials = trials[trials['distractor_angle_deg']==180]
	
	large_hh_rep = []
	large_hl_rep = []
	large_hn_rep = []
	
	large_lh_rep = []
	large_ll_rep = []
	large_ln_rep = []
	
	large_nh_rep = []
	large_nl_rep = []
	large_nn_rep = []
	
	for i in range(len(large_trials)):
		if ((large_trials['trial_no'].iloc[i] == (large_trials['trial_no'].iloc[i-1]+1)) & (large_trials['which_circle'].iloc[i-1] == 't')):
			if large_trials['reward_available'].iloc[i-1] == 0: # if previous trial was high reward
				if large_trials['reward_available'].iloc[i] == 0: #if current trial is high reward
					large_hh_rep.append([large_trials['trial_no'].iloc[i],large_trials['saccade_onset_target'].iloc[i], large_trials['reward_available'].iloc[i], large_trials['distractor_angle_deg'].iloc[i]])
				elif (large_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					large_hl_rep.append([large_trials['trial_no'].iloc[i],large_trials['saccade_onset_target'].iloc[i], large_trials['reward_available'].iloc[i], large_trials['distractor_angle_deg'].iloc[i]])
				elif (large_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					large_hn_rep.append([large_trials['trial_no'].iloc[i],large_trials['saccade_onset_target'].iloc[i], large_trials['reward_available'].iloc[i], large_trials['distractor_angle_deg'].iloc[i]])
			elif large_trials['reward_available'].iloc[i-1] == 1: # if previous trial was low reward
				if (large_trials['reward_available'].iloc[i] == 0): #if current trial is high reward
					large_lh_rep.append([large_trials['trial_no'].iloc[i],large_trials['saccade_onset_target'].iloc[i], large_trials['reward_available'].iloc[i], large_trials['distractor_angle_deg'].iloc[i]])
				elif (large_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					large_ll_rep.append([large_trials['trial_no'].iloc[i],large_trials['saccade_onset_target'].iloc[i], large_trials['reward_available'].iloc[i], large_trials['distractor_angle_deg'].iloc[i]])
				elif (large_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					large_ln_rep.append([large_trials['trial_no'].iloc[i],large_trials['saccade_onset_target'].iloc[i], large_trials['reward_available'].iloc[i], large_trials['distractor_angle_deg'].iloc[i]])
			elif large_trials['reward_available'].iloc[i-1] == 2: # if previous trial was no reward
				if (large_trials['reward_available'].iloc[i] == 0): #if current trial is high reward
					large_nh_rep.append([large_trials['trial_no'].iloc[i],large_trials['saccade_onset_target'].iloc[i], large_trials['reward_available'].iloc[i], large_trials['distractor_angle_deg'].iloc[i]])
				elif (large_trials['reward_available'].iloc[i] == 1): #if current trial is low reward
					large_nl_rep.append([large_trials['trial_no'].iloc[i],large_trials['saccade_onset_target'].iloc[i], large_trials['reward_available'].iloc[i], large_trials['distractor_angle_deg'].iloc[i]])
				elif (large_trials['reward_available'].iloc[i] == 2): #if current trial is no reward
					large_nn_rep.append([large_trials['trial_no'].iloc[i],large_trials['saccade_onset_target'].iloc[i], large_trials['reward_available'].iloc[i], large_trials['distractor_angle_deg'].iloc[i]])
	
	# All means
	mean_all_hh = np.mean(all_hh_rep, axis=0)[1]
	mean_all_hl = np.mean(all_hl_rep, axis=0)[1]
	mean_all_hn = np.mean(all_hn_rep, axis=0)[1]
	
	mean_all_lh = np.mean(all_lh_rep, axis=0)[1]
	mean_all_ll = np.mean(all_ll_rep, axis=0)[1]
	mean_all_ln = np.mean(all_ln_rep, axis=0)[1]	
	
	mean_all_nh = np.mean(all_nh_rep, axis=0)[1]
	mean_all_nl = np.mean(all_nl_rep, axis=0)[1]
	mean_all_nn = np.mean(all_nn_rep, axis=0)[1]	
	
	# Small means
	mean_small_hh = np.mean(small_hh_rep, axis=0)[1]
	mean_small_hl = np.mean(small_hl_rep, axis=0)[1]
	mean_small_hn = np.mean(small_hn_rep, axis=0)[1]
	
	mean_small_lh = np.mean(small_lh_rep, axis=0)[1]
	mean_small_ll = np.mean(small_ll_rep, axis=0)[1]
	mean_small_ln = np.mean(small_ln_rep, axis=0)[1]	
	
	mean_small_nh = np.mean(small_nh_rep, axis=0)[1]
	mean_small_nl = np.mean(small_nl_rep, axis=0)[1]
	mean_small_nn = np.mean(small_nn_rep, axis=0)[1]
	
	# Med means
	mean_med_hh = np.mean(med_hh_rep, axis=0)[1]
	mean_med_hl = np.mean(med_hl_rep, axis=0)[1]
	mean_med_hn = np.mean(med_hn_rep, axis=0)[1]
	
	mean_med_lh = np.mean(med_lh_rep, axis=0)[1]
	mean_med_ll = np.mean(med_ll_rep, axis=0)[1]
	mean_med_ln = np.mean(med_ln_rep, axis=0)[1]	
	
	mean_med_nh = np.mean(med_nh_rep, axis=0)[1]
	mean_med_nl = np.mean(med_nl_rep, axis=0)[1]
	mean_med_nn = np.mean(med_nn_rep, axis=0)[1]
	
	# Large means
	mean_large_hh = np.mean(large_hh_rep, axis=0)[1]
	mean_large_hl = np.mean(large_hl_rep, axis=0)[1]
	mean_large_hn = np.mean(large_hn_rep, axis=0)[1]
	
	mean_large_lh = np.mean(large_lh_rep, axis=0)[1]
	mean_large_ll = np.mean(large_ll_rep, axis=0)[1]
	mean_large_ln = np.mean(large_ln_rep, axis=0)[1]	
	
	mean_large_nh = np.mean(large_nh_rep, axis=0)[1]
	mean_large_nl = np.mean(large_nl_rep, axis=0)[1]
	mean_large_nn = np.mean(large_nn_rep, axis=0)[1]
	
	if angle == 'all':
		return [mean_all_hh, mean_all_hl, mean_all_hn, mean_all_lh,mean_all_ll, mean_all_ln, mean_all_nh, mean_all_nl, mean_all_nn]
	else:
		return [mean_small_hh, mean_small_hl, mean_small_hn, mean_small_lh,mean_small_ll, mean_small_ln, mean_small_nh, mean_small_nl, mean_small_nn, mean_med_hh, mean_med_hl, mean_med_hn, mean_med_lh, mean_med_ll, mean_med_ln, mean_med_nh, mean_med_nl, mean_med_nn, mean_large_hh, mean_large_hl, mean_large_hn, mean_large_lh, mean_large_ll,mean_large_ln, mean_large_nh, mean_large_nl, mean_large_nn]
	
	
def saccade_vigor():
	# Peak velocity as a function of saccade amplitude
	
	# Amplitude and Peak Velocity plot per subject
	xdata = trials['CURRENT_SAC_AMPLITUDE']
	ydata = trials['CURRENT_SAC_PEAK_VELOCITY']

	#Change x and y data into arrays so curve_fit will work
	x = np.array(xdata, dtype=float)
	y = np.array(ydata, dtype = float)

	plt.plot(xdata,ydata, 'ro', label = 'Original Data')
	plt.xlabel('Amplitude (deg)')
	plt.ylabel('Peak Velocity (deg/s)')
		
	# Fit a linear function to the data	
	def linear_func(x, a, c):
		return (a*x) + c

	popt, pcov = curve_fit(linear_func, x, y)
	print "slope = %s, intercept = %s" %(popt[0],popt[1])

	xs = sym.Symbol('\lambda')
	tex = sym.latex(linear_func(xs,*popt)).replace('$','')
	#plt.title(r'$f(\lambda = %s$' %(tex), fontsize = 16))
	plt.plot(x, linear_func(x, *popt), label = 'Fitted Curve')
	plt.legend(loc='upper left')
	plt.show() # shows both original data and the fit

	expected_v = []
	for i in range(trials.shape[0]):
		expected_v.append([linear_func(trials['CURRENT_SAC_AMPLITUDE'].iloc[i], popt[0],popt[1]), trials['trial_no'].iloc[i]])

	expected_peakvel = pd.DataFrame(expected_v, columns = ['expected_peakvel','trial_no'])

	peakvel_trials = pd.merge(trials, expected_peakvel, how = 'outer', on= 'trial_no')

	# Calculate saccade vigor : ratio peak_vel/expected_peak_vel
	sac_vigor = []

	for i in range(peakvel_trials.shape[0]):
		sac_vigor.append([peakvel_trials.iloc[i]['CURRENT_SAC_PEAK_VELOCITY']/peakvel_trials.iloc[i]['expected_peakvel'], peakvel_trials['trial_no'].iloc[i]])

	saccade_vigor = pd.DataFrame(sac_vigor, columns = ['saccade_vigor','trial_no'])

	vigor_trials = pd.merge(peakvel_trials, saccade_vigor, how = 'outer', on= 'trial_no')
		
def all_subjects(analysis, data, output_filename):
	
	subjects_output.append(data)	
	
	if sb == 0:
		f = open(output_filename,'w')
		writer = csv.writer(f, delimiter = ' ')
		f.close()
			
	return subjects_output

def write_file(output, output_filename):
	with open(output_filename, "wb") as f:
		writer = csv.writer(f)
		writer.writerows(output)
	

# Run these functions

which_analysis = ['latency']#['latency', 'median_latency','mean_latency', 'learning_latencies', 'amplitude', 'errors', 'learning_errors', 'peak_velocity', 'cog', 'saccade_evolution', 'curvature_absolute', 'intertrial_priming'] # can do one or multiple analyses

which_type = 'target' # can be 'target', 'distractor', or 'both'

hemifield = 0 # 1 for hemifield analysis, 0 for standard analysis

intertrial_angle = ''
global latency_histogram # for Jessica Heeman
latency_histogram = [[],[],[],[],[],[],[],[],[]]


# Analysis loop
for analysis in which_analysis:
	
	print analysis
	
	subjects_output = []

	# Subject loop
	for sb in range(len(subjects)):
		
		print sb
		
		exclusions(sb)
	
		preprocessed_trials = preprocessing(sb, which_type = which_type, analysis = analysis)
	
		dataframe_trials = dataframes(preprocessed_trials) # get dataframes
		
		os.chdir('/Users/bronagh/Documents/LePelley/LePelley_2/results/24_subjects')
		
		global output_filename
		
		if analysis == 'latency':
			latencies = saccade_latency(which_type = which_type, hemifield = hemifield)
			if which_type == 'target':
				if hemifield == 0:
					output_filename = 'latency_target.csv'
				else:
					output_filename = 'latency_target_hemi.csv'
			elif which_type == 'distractor': 
				if hemifield == 0:
					output_filename = 'latency_distractor.csv'
				else:
					output_filename = 'latency_distractor_hemi.csv'
			
			# append to array of all subjects
			subjects_output = all_subjects(analysis = analysis, data = latencies, output_filename = output_filename)	

		elif analysis == 'median_latency':
			latencies = saccade_median_latency(which_type = which_type, hemifield = hemifield)
			if which_type == 'target':
				if hemifield == 0:
					output_filename = 'latency_median_target.csv'
				else:
					output_filename = 'latency_median_target_hemi.csv'
			elif which_type == 'distractor': 
				if hemifield == 0:
					output_filename = 'latency_median_distractor.csv'
				else:
					output_filename = 'latency_median_distractor_hemi.csv'
			
			# append to array of all subjects
			subjects_output = all_subjects(analysis = analysis, data = latencies, output_filename = output_filename)			
			
		elif analysis == 'mean_latency':
			mean_latencies = mean_latency(which_type = which_type)
			
			if which_type == 'target':
				output_filename = 'mean_latency_target.csv'
			elif which_type == 'distractor':
				output_filename = 'mean_latency_distractor.csv'
			elif which_type == 'both':
				output_filename = 'mean_latency_all.csv'
				
			# append to array of all subjects
			subjects_output = all_subjects(analysis =  analysis, data = mean_latencies, output_filename = output_filename)

		elif analysis == 'learning_latencies':
			learning = learning_latencies(which_type = which_type, hemifield = hemifield)
			if hemifield == 0:
				output_filename = 'learning_latencies.csv'	
			else:
				output_filename = 'learning_latencies_hemi.csv'		

			# append to array of all subjects
			subjects_output = all_subjects(analysis =  analysis, data = learning, output_filename = output_filename)
			
		elif analysis == 'amplitude':
			amplitudes = saccade_amplitude(hemifield = hemifield)
			if hemifield == 0:
				output_filename = 'amplitudes.csv'
			else:
				output_filename = 'amplitudes_hemi.csv'
			
			# append to array of all subjects
			subjects_output = all_subjects(analysis =  analysis, data = amplitudes, output_filename = output_filename)
		
		elif analysis == 'errors': #make sure which_type = 'distractor'
			distractors = erroneous_saccades(hemifield = hemifield)
			if hemifield == 0:	
				output_filename = 'errors.csv'
			else:
				output_filename = 'errors_hemi.csv'
			
			# append to array of all subjects
			subjects_output = all_subjects(analysis =  analysis, data = distractors, output_filename = output_filename)
			
		elif analysis == 'learning_errors': #make sure which_type = 'distractor'
			distractors = learning_errors(hemifield = hemifield)
			if hemifield == 0:	
				output_filename = 'learning_errors.csv'
			else:
				output_filename = 'learning_errors_hemi.csv'
			
			# append to array of all subjects
			subjects_output = all_subjects(analysis =  analysis, data = distractors, output_filename = output_filename)
			
		elif analysis == 'peak_velocity':
			peak_velocities = peak_velocity(hemifield = hemifield)
			
			if hemifield == 0:	
				output_filename = 'peak_velocity.csv'
			else:
				output_filename = 'peak_velocity_hemi.csv'
			
			# append to array of all subjects
			subjects_output = all_subjects(analysis =  analysis, data = peak_velocities, output_filename = output_filename)
		
		elif analysis == 'cog':
			distances = distance_from_cog()
			output_filename = 'cog.csv'
			
			# append to array of all subjects
			subjects_output = all_subjects(analysis =  analysis, data = distances, output_filename = output_filename)

		elif analysis == 'saccade_evolution':
			rois_quartiles = saccade_evolution()
			output_filename = 'saccade_evolution.csv'
			
			# append to array of all subjects
			subjects_output = all_subjects(analysis =  analysis, data = rois_quartiles, output_filename = output_filename)
		
		elif analysis == 'curvature_absolute':
			absolute_curvatures = curvature_absolute(sb, hemifield = hemifield)
			
			if hemifield == 0:
				output_filename = 'curvature_absolute_peak.csv'
			else:
				output_filename = 'curvature_absolute_peak_hemi.csv'

			# append to array of all subjects
			subjects_output = all_subjects(analysis =  analysis, data = absolute_curvatures, output_filename = output_filename)
		
		elif analysis == 'intertrial_priming':
			priming = intertrial_priming(hemifield = hemifield, angle = intertrial_angle)
			if hemifield == 0:
				output_filename = 'intertrial_priming.csv'	
			else:
				output_filename = 'intertrial_priming_hemi.csv'		
			
			# append to array of all subjects
			subjects_output = all_subjects(analysis =  analysis, data = priming, output_filename = output_filename)

	write_file(subjects_output, output_filename)

shell()

