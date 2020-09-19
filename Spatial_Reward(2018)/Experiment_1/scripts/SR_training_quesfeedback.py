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
subject_files = ['SR01_TR.csv','SR02_TR.csv','SR03_TR.csv','SR04_TR.csv','SR05_TR.csv','SR06_TR.csv','SR07_TR.csv','SR08_TR.csv','SR09_TR.csv','SR10_TR.csv','SR11_TR.csv','SR12_TR.csv','SR13_TR.csv','SR14_TR.csv','SR15_TR.csv','SR16_TR.csv','SR17_TR.csv','SR18_TR.csv','SR19_TR.csv','SR20_TR.csv','SR21_TR.csv','SR22_TR.csv','SR23_TR.csv','SR24_TR.csv']

# Read in Eye-tracker Saccade Report file as a data frame -> ALL participants
os.chdir('/Users/bronagh/Documents/Spatial_Reward_Toolbox/experiment_1/data/csv')

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