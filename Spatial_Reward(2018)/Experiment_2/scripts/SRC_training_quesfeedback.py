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
subject_files = ['SRC01_TR.csv','SRC02_TR.csv','SRC03_TR.csv','SRC04_TR.csv','SRC05_TR.csv','SRC06_TR.csv','SRC07_TR.csv','SRC08_TR.csv','SRC09_TR.csv','SRC10_TR.csv','SRC11_TR.csv','SRC12_TR.csv','SRC13_TR.csv','SRC14_TR.csv','SRC15_TR.csv','SRC17_TR.csv','SRC18_TR.csv','SRC19_TR.csv','SRC20_TR.csv','SRC21_TR.csv','SRC23_TR.csv','SRC24_TR.csv'] # Need to get 'SRC22_TR.csv'

# Read in Eye-tracker Saccade Report file as a data frame -> ALL participants
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_2/data/csv')

nr_ques_trials, correct_ques_trials = [[] for i in range(2)]

###### SUBJECT LOOP ##########
for sb in range(len(subject_files)):
	print subject_files[sb]
	
	subject_df = pd.read_csv(subject_files[sb])
	
	ques_trials = subject_df[(subject_df['ques_feedback'] == 'correct') | (subject_df['ques_feedback']=='incorrect')]
	nr_correct = ques_trials[(ques_trials['ques_feedback'] == 'correct')].shape[0]
	
	#Group
	nr_ques_trials.append(ques_trials.shape[0])
	correct_ques_trials.append((nr_correct/ques_trials.shape[0])*100)

shell()