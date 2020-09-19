"""
Analysis script for experiment DS_EXO

Paul Boon

"""


#!/usr/bin/env python
import math
import os
import sys
import pandas as pd

#sys.path.append("/Users/knapen/Documents/LePelley/LePelley_2/data/csv")

from parserPB import Parser

#
# IMPORTANT STUFF BELOW
#

class Analyze(Parser):

    # Specifiy which lines should be interpreted as the start of a trial
    def start_of_trial(self, word):
        return len(word) == 3 and word[2] == "start_trial"
      #  return len(word) == 21 and word[3] == "TRIAL" 
        

    # Specify which lines should be interpreted as the end of a trial
    def end_of_trial(self, word):
        return len(word) == 3 and word[2] == "stop_trial"
        # return len(word) == 4 and (word[2] == "TRIAL" or word[2] == "BLOCK") and word[3] in ("OK", "ERROR", "ABORTED")

    # Extract the trial id from a string
    def trial_id(self, word):
        return int(word[1])
            
    def description(self):
        
        # Identifies your experiment
        return "ASC Parser Template"

    def criteria(self):

        #return ["TRIALID < 100"]
        return []

    def attrstats(self, attr):

      #  if attr == "saccade_rt":
      #    return ["FILE", "SOA", "DistrMatch"]
        
        return False
     

    def attrtype(self, attr):

        if "bin" in attr:
            return "double"

        if "sacc_angle_within_target" in attr:
            return "varchar(30)"

        if "saccade_rt_within_bounds" in attr:
            return "varchar(30)"

        if "saccade_too_small" in attr:
            return "varchar(30)"

        if "curvature_done" in attr:
            return "varchar(30)"

        if "early_saccades_present" in attr:
            return "varchar(30)"

        if "sacc_angle_within_target" in attr:
            return "varchar(30)"

        if "saccade_size" in attr:
            return "double"

        if attr == "thisisastring":
            return "varchar(100)"

        return Parser.attrtype(self, attr)

   
    
    # Parse memory correctness messages (memory expts only), by default lines of the form MSG __timestamp__ CORRECT __value__TotScore __value __
    def parse_ROI(self, trial, word):
        if len(word) == 11  and word[3] == "IAREA" and word[0] == "MSG":
            trial[word[10]+"x"] = (word[8]+word[6])/2
            trial[word[10]+"y"] = (word[9]+word[7])/2

    def parse_displays(self, trial, word):
        if len(word) >3  and word[3] == "search" and word[0] == "MSG":
            trial[word[2]] = word[1]
            
    def parse_leavingfixation(self, trial, word):
        if 21.82 >= self.distance(512, 384, word[1], word[2]):
            T1 = math.sqrt((512 - trial["targetx"])**2 + (384 - trial["targety"])**2)
            T2 = math.sqrt((512 - word[1])**2 + (384 - word[2])**2)
            T3 = math.sqrt((trial["targetx"] - word[1])**2 + (trial["targety"] - word[2])**2)
                
            earlytargetangle = math.acos((T1**2 + T2**2 - T3**2) / (2 * T1 * T2)) 
                
            D1 = math.sqrt((512 - trial["distractorx"])**2 + (384 - trial["distractory"])**2)
            D2 = math.sqrt((512 - word[1])**2 + (384 - word[2])**2)
            D3 = math.sqrt((trial["distractorx"] - word[1])**2 + (trial["distractory"] - word[2])**2)
                
            earlydistractorangle = math.acos((D1**2 + D2**2 - D3**2) / (2 * D1 * D2))
            
            if math.degrees(earlytargetangle) < 30:
                trial["early2target"] = 1
                
            if math.degrees(earlydistractorangle) < 30:
                trial["early2distractor"] = 1
                
            trial["earlyDONE"] = 1
            
            trial["firstsaccadeonset"] = word[0] - trial["SHOWING"]

    
    def initialize(self, trial):
        
        # As soon as a new trial starts, this function is
        # called. This allows you to set up some stuff.
        
        # The trial variable is a dictionary (see Python manual). All
        # content that you store there is automatically stored
        # in the data.csv file.
       
        self.d2target = 0
        
        self.nr_bins = 10      
        for i in range(self.nr_bins):
            trial["bin%d" % i] = []
            

        trial["saccade_found"] = 0
        trial["fixation_found"] = 0
        trial["sample_found"] = 0
        trial["saccade_too_small"] = "no"
        trial["saccade_start"] = 0 
        trial["saccade_rt"] = 0

        self.saccade_detected = False
        trial["SHOWING"] = "NA"
        trial["tondistractor"] = "NA"
        trial["curvature1_done"] = False
        trial["curvature2_done"] = False
        trial["early_saccades_present"]= False
        trial["distractorangle"] = "NA"
        trial["targetangle"] = "NA"
        trial["early_angle"] = "NA"
        trial["firstondistractor"] = 0
        trial["firstOnDistOLD"] = 0
        trial["firstdirectiondistractor"] = 0
        trial["firstdirectiontarget"] = 0
        trial["firstontarget"] = 0
        trial["secondontarget"] = 0
        trial["DirectionNeartarget"] = 0
        trial["DirectionTarget"] = 0
        trial["DirectionIntermediate"] = 0
        trial["DirectionDistractor"] = 0
        trial["DirectionNeardistractor"] = 0
        
        trial["EarlyX"] = "NA"
        trial["EarlyY"] = "NA"
        
        trial["firstsaccadeonset"] = "NA"

        trial["early2target"] = 0
        trial["early2distractor"] = 0
        
        trial["earlyTerror"] = 0
        trial["earlyDerror"] = 0

        
       
    def parseline(self, trial, word):

        # This function is called with each line in the asc file. The contents of
        # each line are split and stored in the 'word' list (see Python manual).

        self.parse_displays(trial, word)
        self.parse_ROI(trial, word)
        
            
    
      
        
        # Here we detect the start of a saccade to start recording curvature.
        # Note that we do not check for multiple saccades, so basically we always
        # end up with the last saccade (which is often a small useless corrective saccade).
        if len(word) > 0 and word[0] == "SSACC":
            self.start_curvature()
            
        # THE EASY WAY
        #   
        # However, there is an easier way built in to the script.
        # You can automatically parse saccades, fixations and samples
        # into a dictionary.
        # Take a look at parser.py line 182 - 264. There you can see
        # which attributes (duration etc) are stored in the dictionary.
        
        # Autoparse a saccade. Note that this corresponds ESACC, so it does
        # not respond to the start of a saccade (SSACC).
        saccade = self.saccade(word)
        
        
        
        if saccade != False:            
              
            if trial["SHOWING"] > 0 and saccade["size_deg"] > 2 and trial["curvature1_done"] == False:
                # Stop recording curvature end specify which points are to be used as the start and endpoints of the saccade.
                # Curvature is stored as trial["curvature"] as average deviation in radians, clockwise < 0.
                self.saccade1 = self.saccade(word)
                
                T1 = math.sqrt((512 - trial["targetx"])**2 + (384 - trial["targety"])**2)
                T2 = math.sqrt((512 - self.saccade1["end_x_px"])**2 + (384 - self.saccade1["end_y_px"])**2)
                T3 = math.sqrt((trial["targetx"] - self.saccade1["end_x_px"])**2 + (trial["targety"] - self.saccade1["end_y_px"])**2)
                
                trial["targetangle"] = math.acos((T1**2 + T2**2 - T3**2) / (2 * T1 * T2)) 
                
                D1 = math.sqrt((512 - trial["distractorx"])**2 + (384 - trial["distractory"])**2)
                D2 = math.sqrt((512 - self.saccade1["end_x_px"])**2 + (384 - self.saccade1["end_y_px"])**2)
                D3 = math.sqrt((trial["distractorx"] - self.saccade1["end_x_px"])**2 + (trial["distractory"] - self.saccade1["end_y_px"])**2)
                
                trial["distractorangle"] = math.acos((D1**2 + D2**2 - D3**2) / (2 * D1 * D2))
                
                ## Richard Godijn 2002 Distance = neartarget, target,...
                
                if math.degrees(trial["targetangle"]) < 15:
                    trial["DirectionTarget"] = 1
                elif math.degrees(trial["targetangle"]) < 45:
                    if math.degrees(trial["distractorangle"]) < 45:
                        trial["DirectionIntermediate"] = 1
                    else: trial["DirectionNeartarget"] = 1
                if math.degrees(trial["distractorangle"]) < 15:
                    trial["DirectionDistractor"] = 1
                elif math.degrees(trial["distractorangle"]) < 45 and trial["DirectionIntermediate"] == 0:
                    trial["DirectionNeardistractor"] = 1
                        
                
                
                
                
                
                
                
                
                
                ## Standaard direction of first saccade
                
                if math.degrees(trial["targetangle"]) < 30:
                        trial["firstdirectiontarget"] = 1
                        
                self.stop_curvature(trial, (self.saccade1["start_x_deg"], self.saccade1["start_y_deg"], self.saccade1["end_x_deg"], self.saccade1["end_y_deg"]))
                trial["sacc1_curvature"]  = trial["sacc_curvature"]
                        
                ## Early angle (16th sample) direction:
                        
                TE1 = math.sqrt((512 - trial["targetx"])**2 + (384 - trial["targety"])**2)
                TE2 = math.sqrt((512 - trial["EarlyX"])**2 + (384 - trial["EarlyY"])**2)
                TE3 = math.sqrt((trial["targetx"] - trial["EarlyX"])**2 + (trial["targety"] - trial["EarlyY"])**2)
                
                try:
                    earlyTangle = math.acos((TE1**2 + TE2**2 - TE3**2) / (2 * TE1 * TE2)) 
                    if math.degrees(earlyTangle) < 30:
                        trial["early2target"] = 1
                except:
                    trial["earlyTerror"] = 1
                        
                
                            
                if trial["early2target"] == 0:
                
                    DE1 = math.sqrt((512 - trial["distractorx"])**2 + (384 - trial["distractory"])**2)
                    DE2 = math.sqrt((512 - trial["EarlyX"])**2 + (384 - trial["EarlyY"])**2)
                    DE3 = math.sqrt((trial["distractorx"] - trial["EarlyX"])**2 + (trial["distractory"] - trial["EarlyY"])**2)
                    
                    try:
                        earlyDangle = math.acos((DE1**2 + DE2**2 - DE3**2) / (2 * DE1 * DE2))
                        
                        if math.degrees(earlyDangle) < 30:
                            trial["early2distractor"] = 1
                    except:
                         trial["earlyDerror"] = 1
                        
                
                if 76.5 >= self.distance(trial["targetx"], trial["targety"], self.saccade1["end_x_px"], self.saccade1["end_y_px"]):
                    trial["firstontarget"] = 1
                    
                    
 
                else:
                    #Calculate angle between vector(fixation-saccadeEnd) and vector (fixation-target/distractorlocation)
                    
                    if trial["distractorangle"] <= 0.1713:
                        trial["firstOnDistOLD"] = 1
                    if math.degrees(trial["distractorangle"]) < 30:
                        trial["firstdirectiondistractor"] = 1
                        
                    if 111.5 >= self.distance(trial["distractorx"], trial["distractory"], self.saccade1["end_x_px"], self.saccade1["end_y_px"]):
                        trial["firstondistractor"] = 1
                
                
                
                
                trial["curvature1_done"] = True
            
            elif trial["SHOWING"] > 0 and saccade["size_deg"] > 2 and trial["curvature2_done"]== False and trial["curvature1_done"] == True:
                # Stop recording curvature end specify which points are to be used as the start and endpoints of the saccade.
                # Curvature is stored as trial["curvature"] as average deviation in radians, clockwise < 0.
                self.saccade2 = self.saccade(word)
                
                #saccade end within target ROI?
                if 76.5 >= self.distance(trial["targetx"], trial["targety"], self.saccade2["end_x_px"], self.saccade2["end_y_px"]):
                    trial["secondontarget"] = 1
                    if trial["firstondistractor"] == 1:
                    #calculate time between saccade towards distractor and consecutive saccade toward target
                        trial["tondistractor"] = self.saccade2["start_time"] - self.saccade1["end_time"]
                
                

                
                trial["curvature2_done"] = True  

            

                
            # Mark if there are large saccades that occur before saccade target display    
            if trial["SHOWING"] == 0 and saccade["size_deg"] > 2:
                trial["early_saccades_present"] = True
            
            
        fixation = self.fixation(word) # Autoparse a fixation
        if fixation != False:
            trial["fixation_found"] = 1
            
        sample = self.sample(word)
        if sample != False:
            trial["sample_found"] = 1
        
        #print trial["TRIAL_START"] 
        #print trial["TRIAL_END"]  
        #if 0 in trial:
        #    print trial
        #    quit()
       
    def finalize(self, trial):
            
        # As soon as a trial is finished, this function is called.
        # This allows you to do some finalizing of your analysis.
          

        '''            
        self.d2target = (math.sqrt((trial["XSacc1"] - self.fixation_point()[0])**2 + (trial["YSacc1"] - self.fixation_point()[1])**2)) / self.px_per_deg()
        trial["dist2target"] = round(self.d2target)
        
        trial["saccade1_start_time"] = self.saccade1["start_time"]
        trial["saccade1_rt"] = self.saccade1["start_time"] - trial["saccade_cue1"]
        '''
        if trial["curvature2_done"]== True:
            trial["saccade1_end_time"] = self.saccade1["end_time"]
        
        trial["saccade1_found"] = 1
        trial["saccade1_dur"] = self.saccade1["duration"]
        trial["saccade1_size"] = self.saccade1["size_deg"]
        trial["saccade1_start_x_deg"] = self.saccade1["start_x_deg"]
        trial["saccade1_start_y_deg"] = self.saccade1["start_y_deg"]
        trial["saccade1_end_x_deg"] = self.saccade1["end_x_deg"]
        trial["saccade1_end_y_deg"] = self.saccade1["end_y_deg"]
        trial["sacc1_endpoint_angle"] = 180/math.pi * self.angle(self.fixation_point()[0], self.fixation_point()[1], self.saccade1["end_x_px"], self.saccade1["end_y_px"])
       
                # E.g., we may not want to count saccades which are too small
        if "saccade1_size" in trial and trial["saccade1_size"] < 5:
            trial["saccade1_too_small"] = "yes"
        else:
            trial["saccade1_too_small"] = "no"
            
        if trial["curvature1_done"] == True:    
            trial["saccade1_rt"] = self.saccade1["start_time"] - trial["SHOWING"]
            
        if "cue" in trial:
            trial["saccade1_timing"] = self.saccade1["start_time"] - trial["distractor"]
        
        # Label saccades that do go directly to the target
        if "sacc1_endpoint_angle" in trial and "target1_endpoint_angle" in trial:
            if trial["sacc1_endpoint_angle"] - trial["target1_endpoint_angle"] < 30 and trial["sacc1_endpoint_angle"] - trial["target1_endpoint_angle"] > -30:
                trial["sacc1_angle_within_target"] = "yes"
            elif trial["sacc1_endpoint_angle"] - trial["target1_endpoint_angle"] > 330 or trial["sacc1_endpoint_angle"] - trial["target1_endpoint_angle"] < -330:
                trial["sacc1_angle_within_target"] = "yes"
            else: 
                trial["sacc1_angle_within_target"] = "no"

        # Label saccades that are too fast or too slow
        if "saccade1_rt" in trial and trial["saccade1_rt"] > 150:
            trial["saccade1_rt_within_bounds"] = "yes"
        else:
            trial["saccade1_rt_within_bounds"] = "no"

        # Label saccades that start at the initial fixation 
        if "saccade1_start_x_deg" in trial and ((self.saccade1["start_x_px"]-512) > 129 or (self.saccade1["start_x_px"]-512) < -129 or (self.saccade1["start_y_px"]-384) > 129 or (self.saccade1["start_y_px"]-384) < -129):
            trial["saccade1_within_initial_fixation"] = "no"
        else:
            trial["saccade1_within_initial_fixation"] = "yes"    
             
        
        if trial["curvature2_done"]== True:
            #print trial["sacc_curvature"]
            #trial["sacc1_curvature"]  = trial["sacc_curvature"]
            
            '''        
            self.d2target = (math.sqrt((trial["XSacc1"] - self.fixation_point()[0])**2 + (trial["YSacc2"] - self.fixation_point()[1])**2)) / self.px_per_deg()
            trial["dist2target"] = round(self.d2target)
            '''
            trial["saccade2_found"] = 1
            trial["saccade2_dur"] = self.saccade2["duration"]
            trial["saccade2_size"] = self.saccade2["size_deg"]
            trial["saccade2_start_x_deg"] = self.saccade2["start_x_deg"]
            trial["saccade2_start_y_deg"] = self.saccade2["start_y_deg"]
            trial["saccade2_end_x_deg"] = self.saccade2["end_x_deg"]
            trial["saccade2_end_y_deg"] = self.saccade2["end_y_deg"]
           
                            
 
        
            # E.g., we may not want to count saccades which are too small
            if "saccade2_size" in trial and trial["saccade2_size"] < 5:
                trial["saccade2_too_small"] = "yes"

            # Label saccades that are too fast or too slow
            if "saccade2_rt" in trial and trial["saccade2_rt"] < 600:
                trial["saccade2_rt_within_bounds"] = "yes"
            else:
                trial["saccade2_rt_within_bounds"] = "no"
        

            # Label saccades that do go directly to the target
            if "sacc2_endpoint_angle" in trial and "target2_endpoint_angle" in trial:
                if trial["sacc2_endpoint_angle"] - trial["target2_endpoint_angle"] < 30 and trial["sacc2_endpoint_angle"] - trial["target2_endpoint_angle"] > -30:
                    trial["sacc2_angle_within_target"] = "yes"
                elif trial["sacc2_endpoint_angle"] - trial["target2_endpoint_angle"] > 330 or trial["sacc2_endpoint_angle"] - trial["target2_endpoint_angle"] < -330:
                    trial["sacc2_angle_within_target"] = "yes"
                else: 
                    trial["sacc2_angle_within_target"] = "no"
	# Average the data in the bins of sample points        
        for i in range(self.nr_bins):
            if len(trial["bin%d" % i]) > 0:
                trial["bin%d" % i] = 1.0 * sum(trial["bin%d" % i])/ len(trial["bin%d" % i])
            else:
                trial["bin%d" % i] = 0

        #print "Finalize: %d" % trial["TRIALID"]
        
	# Label saccades that have distractor clockwise or counterclockwise
        '''
        if trial["saccade"] == "up":
                if trial["distractor_location"] == "Left":
			trial["DistrRelatTarget"] = "0"
		else: 
			trial["DistrRelatTarget"] = "1"

	if trial["saccade"] == "down":
		if trial["distractor_location"] == "Right":
			trial["DistrRelatTarget"] = "0"
		else: 
			trial["DistrRelatTarget"] = "1"

	# Label saccades that move past retinotopic or spatiotopic coordinates

	if trial["saccade"] == "up":
                if trial["box_movement"] == "up":
			trial["spatiotopic_retinotopic"] = "spatiotopic"
		else: 
			trial["spatiotopic_retinotopic"] = "retinotopic"

	elif trial["saccade"] == "down":
                if trial["box_movement"] == "up":
			trial["spatiotopic_retinotopic"] = "retinotopic"
		else: 
			trial["spatiotopic_retinotopic"] = "spatiotopic"

        '''

       
       
   # This function can be used to process curvature points
    def process_curvature_points(self, trial, norm, sample, start, end, dsample, norm_asample):

        #length_sacc_deg = self.d2target
        length_sacc_deg = 9.0
        bin_size = (length_sacc_deg * self.px_per_deg()) / self.nr_bins
        trial["bin_size"] = bin_size /  self.px_per_deg()

        for i in range(self.nr_bins):
            trial["distbin%d" % i] = i * trial["bin_size"]  + trial["bin_size"] / 2

        
        # Get the quotent (the main part)
        bin_number = dsample //  bin_size

                
        # Collect the normalized angles in appropriate bins
        if bin_number < self.nr_bins:
            trial["bin%d" % bin_number].append(norm_asample)
           # print "bin number: %d, dsample: %d, norm_a: %f" % (bin_number, dsample, norm_asample)  
            
        
    def px_per_deg(self):   

        # How many pixels correspond to one visual degree?  
        # Default in K2D-38 is 32
        # Default in K2D-35 is 37
        # Default in K2d-37 is 43
        return 32

    def resolution(self):       

        # Default in K2D-37 is
        return (1024, 768)

    def fixation_point(self):

        # Coordinates of the fixation point
        return (512, 384) 

      

        # These are some default mysql connection parameters, which can be overriden if necessary
    def user(self):
        return "abelopolsky"

    def password(self):
        return "0Haff9em"

    def database(self):
        return "abelopolsky"

    def table(self):
        return "data"

    def host(self):
        return "vpc-0010.psy.vu.nl"   

# This line simply starts the analysis

if __name__ == "__main__":
    analyze = Analyze()
    os.chdir('/Users/knapen/Documents/LePelley/LePelley_2/data/csv')
    datframe = pd.read_csv("LP01.csv")