import math
import os
import sys
import pandas as pd
import csv


from parserPB import Parser


# BM notes: 
# To run: need to put participant filename in parserPB.py parse() and __init__ functions, 
# (lines ~512 and ~626).
# Then run from this script. There is an asc_LP** folder in data/asc for each participant 
# e.g. asc_LP01 folder contains ascii file LP01.asc.
# Output is to results/curvature folder, per participant. 
# Can adjust curvature calculations in function stop_curvature() in parserPb.py (starts at line 312),
# and can change the output of that function on 405 to either average curvature or median curavature.
# If changing between average and median, it's best to change name of output files, lines
# 112 - 114 at the bottom of this script.
# Use curvature_concat.py analysis file to merge curvature details with standard dataframe from edf.

class Analyze(Parser):
    
    # Specifiy which lines should be interpreted as the start of a trial
    def start_of_trial(self, word):
         return ((len(word) == 3 or len(word) == 4) and word[2] == "start_trial")
        

    # Specify which lines should be interpreted as the end of a trial
    def end_of_trial(self, word):
        return len(word) == 3 and word[2] == "stop_trial"

    # Extract the trial id from a string
    def trial_id(self, word):
        return int(word[1])
            
    def description(self):
        
        # Identifies your experiment
        return "ASC Parser Template"

    def criteria(self):

        return []

    def attrstats(self, attr):
        
        return False

        
    def parse_variables(self, trial, word):
        if len(word) == 6  and word[2] == "!V" and word[3] != "TRIAL_VAR":
            trial[word[4]] = word[5]
            
    def parse_displays(self, trial, word):
        if len(word) >3  and word[3] == "Begin" and word[0] == "MSG":
            trial[word[2]] = word[1]
            
        # Extract the trial id from a string
    def trial_number(self, word):
        print word
        if len(word) == 6 and word[2] == "!V" and word[4] == "trial_no":
            return int(word[5])
  

    def parseline(self, trial, word):

        # This function is called with each line in the asc file. The contents of
        # each line are split and stored in the 'word' list (see Python manual).

        #self.parse_variables(trial,word)
        #self.parse_displays(trial, word)
      
        # Saccade start
        if len(word) > 0 and word[0] == "SSACC":
            self.start_curvature()
        
            
        # Parse a saccade. Note that this corresponds ESACC, so it does
        # not respond to the start of a saccade (SSACC).
        saccade = self.saccade(word)
        
        # saccade contains important information about the saccade
        if saccade != False: 
            #print saccade
            if len(word) > 0 and word[0] == "ESACC":
                curvature_ref = [saccade["start_x_deg"], saccade["start_y_deg"],saccade["end_x_deg"], saccade["end_y_deg"]]
                curvature_thisTrial = self.stop_curvature(trial,curvature_ref) 
                #if curvature_thisTrial != 0:
                self.curvature_file(self.nroftrials,curvature_thisTrial)
            
    def px_per_deg(self):   

        # How many pixels correspond to one visual degree?  
        # Default in K2D-38 is 32
        # Default in K2D-35 is 37
        # Default in K2d-37 is 43
        return 27.5 # BM : If distance to screen is 60cm, then there are actually 27.5 px in 1 degree

    def resolution(self):       

        # Default in K2D-37 is
        return (1024, 768)

    def fixation_point(self):

        # Coordinates of the fixation point
        return (512, 384) 
        
    def curvature_file(self, trial_nr, curvature):
        os.chdir("/Users/bronagh/Documents/LePelley/LePelley_2/results/curvature")
        if trial_nr == 1:
            if os.path.exists(self.root + '_curvature_peak.csv'):
                os.remove(self.root + '_curvature_peak.csv')
        file = open(self.root + '_curvature_peak.csv', 'a')
        writer = csv.writer(file)
        writer.writerow([int(trial_nr), float(curvature)])
        file.close()

if __name__ == "__main__":
    analyze = Analyze()