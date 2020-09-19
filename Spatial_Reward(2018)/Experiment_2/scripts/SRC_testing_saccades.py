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
import sys
from IPython import embed as shell

# Read in Eye-tracker Saccade Report file as a data frame -> ALL participants
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_2/analysis/saccade_reports')
saccades_main_df = pd.read_csv('SRC_TE_saccades.csv') 

# In this experiment, reward is always in different hemifields	
subjects_test1 = ['SRC01TE1','SRC02TE1','SRC03TE1','SRC04TE1','SRC05TE1','SRC06TE1','SRC07TE1','SRC08TE1','SRC09TE1','SRC10TE1','SRC11TE1','SRC12TE1','SRC13TE1','SRC14TE1','SRC15TE1', 'SRC16TE1','SRC17TE1','SRC18TE1','SRC19TE1','SRC20TE1','SRC21TE1','SRC22TE1','SRC23TE1','SRC24TE1'] 
subjects_test2 = ['SRC01TE2','SRC02TE2','SRC03TE2','SRC04TE2','SRC05TE2','SRC06TE2','SRC07TE2','SRC08TE2','SRC09TE2','SRC10TE2','SRC11TE2','SRC12TE2','SRC13TE2','SRC14TE2','SRC15TE2','SRC16TE2','SRC17TE2','SRC18TE2','SRC19TE2','SRC20TE2','SR21TE2','SRC22TE2','SRC23TE2','SRC24TE2'] 

#'SRC04TE1'
#'SRC04TE2'

# High v Low = 180 degrees apart (different hemifield in both vertical and horizontal axis)
# subjects_test1 = ['SRC01TE1','SRC02TE1','SRC03TE1','SRC04TE1','SRC09TE1','SRC10TE1','SRC11TE1','SRC12TE1','SRC17TE1','SRC18TE1','SRC19TE1','SRC20TE1']
# subjects_test2 = ['SRC01TE2','SRC02TE2','SRC03TE2','SRC04TE2','SRC09TE2','SRC10TE2','SRC11TE2','SRC12TE2','SRC17TE2','SRC18TE2','SRC19TE2','SRC20TE2']

# High v Low = 90 degrees apart (different hemifield in vertical only)
# subjects_test1 = ['SRC05TE1','SRC06TE1','SRC07TE1','SRC08TE1','SRC13TE1','SRC14TE1','SRC15TE1','SRC16TE1','SRC21TE1','SRC22TE1','SRC23TE1','SRC24TE1']
# subjects_test2 = ['SRC05TE2','SRC06TE2','SRC07TE2','SRC08TE2','SRC13TE2','SRC14TE2','SRC15TE2','SRC16TE2','SRC21TE2','SRC22TE2','SRC23TE2','SRC24TE2']


analysis = 'target_location' # target_location or any_location. This is for analyzing saccades in relation to whether the target was at that location or regardless of where the target was.

timeout = 1000
nr_blocks = 1
trials_per_condition = 30
total_nr_trials = 120

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

all_subjects_saccades_1 = []
all_subjects_total_saccades_1 = []
 
for sb in range(len(subjects_test1)):
	print(subjects_test1[sb])

	subject_df = saccades_main_df[saccades_main_df['RECORDING_SESSION_LABEL'] == subjects_test1[sb]]
	subject_df = subject_df.convert_objects(convert_numeric=True)
	all_subjects_total_saccades_1.append((subject_df.shape[0]/total_nr_trials)*100)

	saccades_window = subject_df[(subject_df['CURRENT_SAC_START_TIME']>(subject_df['fix_to_stim'])) & (subject_df['CURRENT_SAC_START_TIME']<(subject_df['fix_to_stim']+subject_df['response_time']))]
	saccades_window=saccades_window.drop_duplicates('trial_number')

	print('saccades', saccades_window.shape[0])
	#all_subjects_total_saccades_1.append((saccades_window.shape[0]/total_nr_trials)*100)

	# Direction of (micro)saccades made
	saccade_direction = []
	north_west_trials, south_west_trials, north_east_trials, south_east_trials = [[] for i in range(4)]

	if saccades_window.shape[0]>0:

		dflength = len(saccades_window['RECORDING_SESSION_LABEL'])
		saccade_dir= pd.Series(np.zeros(dflength))
		saccades_window = saccades_window.assign(saccade_dir=saccade_dir.values)

		for i in range(saccades_window.shape[0]):
			if ((saccades_window['CURRENT_SAC_END_X'].iloc[i] - saccades_window['CURRENT_SAC_START_X'].iloc[i] < 0) & (saccades_window['CURRENT_SAC_END_X'].iloc[i] < xc)): # if end of saccade has negative x-value
				if ((saccades_window['CURRENT_SAC_END_Y'].iloc[i] - saccades_window['CURRENT_SAC_START_Y'].iloc[i] < 0) & (saccades_window['CURRENT_SAC_END_Y'].iloc[i] < yc)) : # if change in y is negative AND saccade ends in negative y
					saccade_direction.append('north_west')
					saccades_window['saccade_dir'].iloc[i]='north_west'
					north_west_trials.append(saccades_window.iloc[i])					
				elif ((saccades_window['CURRENT_SAC_END_Y'].iloc[i] - saccades_window['CURRENT_SAC_START_Y'].iloc[i] > 0) & (saccades_window['CURRENT_SAC_END_Y'].iloc[i] > yc)): # if change in y is positive AND saccade ends in positive y
					saccade_direction.append('south_west')
					saccades_window['saccade_dir'].iloc[i]='south_west'
					south_west_trials.append(saccades_window.iloc[i])
			elif ((saccades_window['CURRENT_SAC_END_X'].iloc[i] - saccades_window['CURRENT_SAC_START_X'].iloc[i] > 0) & (saccades_window['CURRENT_SAC_END_X'].iloc[i] > xc)): # if change in x value is positive
				if ((saccades_window['CURRENT_SAC_END_Y'].iloc[i] - saccades_window['CURRENT_SAC_START_Y'].iloc[i] < 0) & (saccades_window['CURRENT_SAC_END_Y'].iloc[i] < yc)): # if change in y is negative
					saccade_direction.append('north_east')
					saccades_window['saccade_dir'].iloc[i]='north_east'
					north_east_trials.append(saccades_window.iloc[i])
				elif ((saccades_window['CURRENT_SAC_END_Y'].iloc[i] - saccades_window['CURRENT_SAC_START_Y'].iloc[i] > 0) & (saccades_window['CURRENT_SAC_END_Y'].iloc[i] > yc)): # if change in y is positive
					saccade_direction.append('south_east')
					saccades_window['saccade_dir'].iloc[i]='south_east'
					south_east_trials.append(saccades_window.iloc[i])

		#Getting neutral positions: neut1 = same hemisphere as high, neut2 = same hemisphere as low
		if saccades_window['high_location'].iloc[0] == 'north_west':
			neut1 = 'south_west'
		elif saccades_window['high_location'].iloc[0] == 'north_east':
			neut1 = 'south_east'
		elif saccades_window['high_location'].iloc[0] == 'south_west':
			neut1 = 'north_west'
		elif saccades_window['high_location'].iloc[0] == 'south_east':
			neut1 = 'north_east'
		
		if saccades_window['low_location'].iloc[0] == 'north_west':
			neut2 = 'south_west'
		elif saccades_window['low_location'].iloc[0] == 'north_east':
			neut2 = 'south_east'
		elif saccades_window['low_location'].iloc[0] == 'south_west':
			neut2 = 'north_west'
		elif saccades_window['low_location'].iloc[0] == 'south_east':
			neut2 = 'north_east'

		high_saccade = low_saccade = neut1_saccade = neut2_saccade = 0

		if analysis=='target_location':	

			for i in range(saccades_window.shape[0]):
				if ((saccades_window['high_location'].iloc[i]==saccades_window['target_position'].iloc[i]) & (saccades_window['high_location'].iloc[i]==saccades_window['saccade_dir'].iloc[i])):
					high_saccade+=1
				elif ((saccades_window['low_location'].iloc[i]==saccades_window['target_position'].iloc[i]) & (saccades_window['low_location'].iloc[i]==saccades_window['saccade_dir'].iloc[i])):
					low_saccade+=1
				elif ((saccades_window['target_position'].iloc[i]==neut1) & (saccades_window['saccade_dir'].iloc[i]==neut1)):
					neut1_saccade+=1
				elif ((saccades_window['target_position'].iloc[i]==neut2) & (saccades_window['saccade_dir'].iloc[i]==neut2)):
					neut2_saccade+=1
		
		elif analysis=='any_location':
		
			for i in range(len(saccade_direction)):
				if saccade_direction[i]==saccades_window['high_location'].iloc[i]:
					high_saccade+=1
				elif saccade_direction[i]==saccades_window['low_location'].iloc[i]:
					low_saccade+=1
				elif saccade_direction[i]==neut1:
					neut1_saccade+=1
				elif saccade_direction[i]==neut2:
					neut2_saccade+=1

		high_saccade_percentage = (high_saccade/total_nr_trials) * 100
		low_saccade_percentage = (low_saccade/total_nr_trials) * 100
		neut1_saccade_percentage = (neut1_saccade/total_nr_trials) * 100
		neut2_saccade_percentage = (neut2_saccade/total_nr_trials) * 100
		neutcomb_saccade_percentage = (neut1_saccade_percentage+neut2_saccade_percentage)/2

		all_subjects_saccades_1.append([high_saccade_percentage,low_saccade_percentage,neut1_saccade_percentage, neut2_saccade_percentage, neutcomb_saccade_percentage])

	elif saccades_window.shape[0]==0:
		all_subjects_saccades_1.append([0,0,0,0,0])


### POST-TEST

all_subjects_saccades_2 = []
all_subjects_total_saccades_2 = []

for sb in range(len(subjects_test2)):
	print(subjects_test2[sb])

	subject_df = saccades_main_df[saccades_main_df['RECORDING_SESSION_LABEL'] == subjects_test2[sb]]
	subject_df = subject_df.convert_objects(convert_numeric=True)
	all_subjects_total_saccades_2.append((subject_df.shape[0]/total_nr_trials)*100)

	saccades_window = subject_df[(subject_df['CURRENT_SAC_START_TIME']>(subject_df['fix_to_stim'])) & (subject_df['CURRENT_SAC_START_TIME']<(subject_df['fix_to_stim']+subject_df['response_time']))]
	saccades_window=saccades_window.drop_duplicates('trial_number')

	print('saccades', saccades_window.shape[0])
	#all_subjects_total_saccades_2.append((saccades_window.shape[0]/total_nr_trials)*100)

	# Direction of (micro)saccades made
	saccade_direction = []
	north_west_trials, south_west_trials, north_east_trials, south_east_trials = [[] for i in range(4)]

	if saccades_window.shape[0]>0:

		dflength = len(saccades_window['RECORDING_SESSION_LABEL'])
		saccade_dir= pd.Series(np.zeros(dflength))
		saccades_window = saccades_window.assign(saccade_dir=saccade_dir.values)

		for i in range(saccades_window.shape[0]):
			if ((saccades_window['CURRENT_SAC_END_X'].iloc[i] - saccades_window['CURRENT_SAC_START_X'].iloc[i] < 0) & (saccades_window['CURRENT_SAC_END_X'].iloc[i] < xc)): # if end of saccade has negative x-value
				if ((saccades_window['CURRENT_SAC_END_Y'].iloc[i] - saccades_window['CURRENT_SAC_START_Y'].iloc[i] < 0) & (saccades_window['CURRENT_SAC_END_Y'].iloc[i] < yc)) : # if change in y is negative AND saccade ends in negative y
					saccade_direction.append('north_west')
					saccades_window['saccade_dir'].iloc[i]='north_west'
					north_west_trials.append(saccades_window.iloc[i])					
				elif ((saccades_window['CURRENT_SAC_END_Y'].iloc[i] - saccades_window['CURRENT_SAC_START_Y'].iloc[i] > 0) & (saccades_window['CURRENT_SAC_END_Y'].iloc[i] > yc)): # if change in y is positive AND saccade ends in positive y
					saccade_direction.append('south_west')
					saccades_window['saccade_dir'].iloc[i]='south_west'
					south_west_trials.append(saccades_window.iloc[i])
			elif ((saccades_window['CURRENT_SAC_END_X'].iloc[i] - saccades_window['CURRENT_SAC_START_X'].iloc[i] > 0) & (saccades_window['CURRENT_SAC_END_X'].iloc[i] > xc)): # if change in x value is positive
				if ((saccades_window['CURRENT_SAC_END_Y'].iloc[i] - saccades_window['CURRENT_SAC_START_Y'].iloc[i] < 0) & (saccades_window['CURRENT_SAC_END_Y'].iloc[i] < yc)): # if change in y is negative
					saccade_direction.append('north_east')
					saccades_window['saccade_dir'].iloc[i]='north_east'
					north_east_trials.append(saccades_window.iloc[i])
				elif ((saccades_window['CURRENT_SAC_END_Y'].iloc[i] - saccades_window['CURRENT_SAC_START_Y'].iloc[i] > 0) & (saccades_window['CURRENT_SAC_END_Y'].iloc[i] > yc)): # if change in y is positive
					saccade_direction.append('south_east')
					saccades_window['saccade_dir'].iloc[i]='south_east'
					south_east_trials.append(saccades_window.iloc[i])

		#Getting neutral positions: neut1 = same hemisphere as high, neut2 = same hemisphere as low
		if saccades_window['high_location'].iloc[0] == 'north_west':
			neut1 = 'south_west'
		elif saccades_window['high_location'].iloc[0] == 'north_east':
			neut1 = 'south_east'
		elif saccades_window['high_location'].iloc[0] == 'south_west':
			neut1 = 'north_west'
		elif saccades_window['high_location'].iloc[0] == 'south_east':
			neut1 = 'north_east'
		
		if saccades_window['low_location'].iloc[0] == 'north_west':
			neut2 = 'south_west'
		elif saccades_window['low_location'].iloc[0] == 'north_east':
			neut2 = 'south_east'
		elif saccades_window['low_location'].iloc[0] == 'south_west':
			neut2 = 'north_west'
		elif saccades_window['low_location'].iloc[0] == 'south_east':
			neut2 = 'north_east'

		high_saccade = low_saccade = neut1_saccade = neut2_saccade = 0

		if analysis=='target_location':	

			for i in range(saccades_window.shape[0]):
				if ((saccades_window['high_location'].iloc[i]==saccades_window['target_position'].iloc[i]) & (saccades_window['high_location'].iloc[i]==saccades_window['saccade_dir'].iloc[i])):
					high_saccade+=1
				elif ((saccades_window['low_location'].iloc[i]==saccades_window['target_position'].iloc[i]) & (saccades_window['low_location'].iloc[i]==saccades_window['saccade_dir'].iloc[i])):
					low_saccade+=1
				elif ((saccades_window['target_position'].iloc[i]==neut1) & (saccades_window['saccade_dir'].iloc[i]==neut1)):
					neut1_saccade+=1
				elif ((saccades_window['target_position'].iloc[i]==neut2) & (saccades_window['saccade_dir'].iloc[i]==neut2)):
					neut2_saccade+=1
		
		elif analysis=='any_location':
		
			for i in range(len(saccade_direction)):
				if saccade_direction[i]==saccades_window['high_location'].iloc[i]:
					high_saccade+=1
				elif saccade_direction[i]==saccades_window['low_location'].iloc[i]:
					low_saccade+=1
				elif saccade_direction[i]==neut1:
					neut1_saccade+=1
				elif saccade_direction[i]==neut2:
					neut2_saccade+=1

		high_saccade_percentage = (high_saccade/total_nr_trials) * 100
		low_saccade_percentage = (low_saccade/total_nr_trials) * 100
		neut1_saccade_percentage = (neut1_saccade/total_nr_trials) * 100
		neut2_saccade_percentage = (neut2_saccade/total_nr_trials) * 100
		neutcomb_saccade_percentage = (neut1_saccade_percentage+neut2_saccade_percentage)/2

		all_subjects_saccades_2.append([high_saccade_percentage,low_saccade_percentage,neut1_saccade_percentage, neut2_saccade_percentage, neutcomb_saccade_percentage])

	elif saccades_window.shape[0]==0:
		all_subjects_saccades_2.append([0,0,0,0,0])


saccades1_df = pd.DataFrame(all_subjects_saccades_1, columns=['high','low','neut1','neut2','neutcomb'])
saccades2_df = pd.DataFrame(all_subjects_saccades_2, columns=['high','low','neut1','neut2','neutcomb'])

saccades_diff_df = saccades2_df-saccades1_df

stats.ttest_rel(saccades_diff_df['high'],saccades_diff_df['low'])
stats.ttest_rel(saccades_diff_df['high'],saccades_diff_df['neutcomb'])

os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_2/results')

if analysis=='target_location':
	saccades_diff_df.to_csv('saccades_change_target.csv', index=False)
	saccades1_df.to_csv('saccades_pre_change_target.csv', index=False)
	saccades2_df.to_csv('saccades_post_change_target.csv', index=False)
elif analysis=='any_location':
	saccades_diff_df.to_csv('saccades_change_any.csv', index=False)
	saccades1_df.to_csv('saccades_pre_change_any.csv', index=False)
	saccades2_df.to_csv('saccades_post_change_any.csv', index=False)


os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_2/results/figs')
# Bar plot
width = 0.08

fig, ax = plt.subplots()
rects1 = ax.bar(0.15, np.mean(saccades_diff_df['high']), width, hatch='/', color='white', edgecolor='k', yerr=np.std(saccades_diff_df['high'])/np.sqrt(len(subjects_test2)), ecolor = 'k')
rects2 = ax.bar(0.35, np.mean(saccades_diff_df['low']), width, hatch='/',color='white', edgecolor='k',yerr=np.std(saccades_diff_df['low'])/np.sqrt(len(subjects_test2)), ecolor = 'k')
rects3 = ax.bar(0.55, np.mean(saccades_diff_df['neutcomb']), width, hatch='/',color='white', edgecolor='k', yerr=np.std(saccades_diff_df['neutcomb'])/np.sqrt(len(subjects_test2)), ecolor = 'k')
ax.set_ylabel('Change in saccades (%)', fontsize = '12')
ax.set_xlabel('Reward Location', fontsize = '12')
ax.yaxis.set_label_coords(-0.10, 0.5) # -.1 on x-axis, .5 = halfway along y-axis
ax.xaxis.set_label_coords(0.5, -0.10)
ax.axhline(y=0, color='k')
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(0, 0.7)
ax.set_xticks([0.15, 0.35, 0.55])
ax.set_xticklabels( ('High','Low', 'No') )
# Use following 2 lines to put significance marker between high and low reward location
#ax.axhline(y=1.2, color='k', xmin=0.21, xmax=0.78)
#ax.text(0.34, 1.3, '*', fontweight='bold')
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_2/results/figs')
if analysis=='target_location':
	plt.savefig('Saccades_change_target.pdf')
	plt.savefig('Saccades_change_target.png')
elif analysis=='any_location':
	plt.savefig('Saccades_change_any.pdf')
	plt.savefig('Saccades_change_any.png')	
plt.show()
plt.close()

shell()


