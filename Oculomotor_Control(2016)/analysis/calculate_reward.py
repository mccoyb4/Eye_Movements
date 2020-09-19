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

all_subjects = []

for sb in range(len(subjects)):
	
	total_points = []
	
	# Read saccades file for this subject only	
	this_subject = pd.read_csv(subject_files[sb]) 
	
	list(this_subject.columns.values)
	
	#Converting strings ('objects' in pandas terminology) that should be numeric to floats
	this_subject = this_subject.convert_objects(convert_numeric=True)
	
	exp_trials = this_subject[(this_subject['CURRENT_SAC_INDEX']== 1) & (this_subject['CURRENT_SAC_CONTAINS_BLINK']== False)]

	for tr in range(165,1876,90): # 165 = last trial in block 1, then go up in steps of 90 trials
		for i in range(len(exp_trials)):
			if exp_trials['trial_no'].iloc[i] == tr:
				total_points.append(exp_trials['points_accumulated'].iloc[i])
	
	money_received = (np.sum(total_points)/1000) * 1.5
	#print subjects[sb],': ',money_received
	
	all_subjects.append(money_received)

mean_reward = np.mean(all_subjects)
std_reward = np.std(all_subjects)
print mean_reward
print std_reward