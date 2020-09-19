#!/usr/bin/env python

"""
Experimental Data Parser

Sebastiaan Mathot

Changes:

22-12-2010: Artem Belopolsky
- Edited sample() to accept either one eye or two

25-06-2010: Artem Belopolsky
- Edited process_curvature_points() to assign points to bins

24-06-2010: Artem Belopolsky
- Added parse_memory_scoreAVB() for getting memory scores in memory expts into Analyze()
- Added recording of too early saccades to parseline()

01-06-2010: Artem Belopolsky

- Edited the start_trial and end_trial detection
- Edited saccade() function to look for the left eye only (in case of binocular recording)
- Edited sample() function: in Eyelink2 samples have 7 words on the line, in 1000 only 4
- Edited the minimal length variable in the stop_curvature() from 20 to 10 points
- Added parsedisplaysAVB() and entered a clause to calculate curvature only for large saccades after saccade target display (DISPLAY 4)
- Moved calling parsedisplaysAVB() from parse() to parseline()

25-02-2010
- Changed to platform independent os.path.join()
- Changed it so the curvature_ref can be passed to stop_curvature() as well

26-01-2010
- Now keeping track of start_trial and end_trial timestamps as TRIAL_START and TRIAL_END
- Added functions to work with angles: angle(), d_deg() and d_rad()

15-07-2009
- Added stop_curvature() function
- Renamed calculate_curvature() to collect_samples() as the actual calculation is now done in stop_curvature()

14-07-2009
- Added curvature calculation methods start_curvature() and determine_curvature()
- Added deg_to_px_x() and deg_to_px_y() functions
- Added get_median() function

30-06-2009
- Added description function
- Fixed a bug which caused median scores to be performed based on string comparisons, rather than numerical where possible

24-03-2009
- Saccade, fixation and sample functions return values in degrees as well
- Attributes starttime and endtime are now start_time and end_time
- pxdeg() is now px_to_deg()
- added resolution() function

21-04-2009
- Fixed import _mysql in skip parsing mode
- Added the print trial ids command line option

17-04-2009
- Fixed a bug where the fixation method contained references to a non-existent saccade variable
- Corrected typo: intiliaize -> initialize
- Changed default criteria return value from empty tuple to empty list


"""

import pandas as pd
import os
import sys
import math
import operator
import copy
import time
from IPython import embed as shell

from PIL import Image, ImageDraw

class Parser:

    ###
    ### OVERRIDE THESE METHODS
    ###

    def description(self):
        return "Generic experiment description.\nPlease override Parser.description() for a more useful description."

    # This function does nothing, but needs to be overloaded
    def parseline(self, trial, word):
        return

    # This function may do some extra processing when an entire trial is read
    def finalize(self, trial):
        return

    # This function may do some processing before the trial is read
    def initialize(self, trial):
        return

    # Returns the type that is to be used in the mysql database for an attribute
    def attrtype(self, attr):
        return "int"

    # Determines if percentile and median scores are calculated for an attribute. Percentile scores are stored as "ATTRNAME"_PERC
    # Scores are always grouped by FILE, an additional attribute can be specified as a return value, or None for no additional grouping
    def attrstats(self, attr):
        return False

    # this function can be used to hide some of the attributes, so the data becomes more readable
    def hideattr(self, attr):
        return False

    # this function can be used to hide some of the trials, defaults to hiding practice trials
    def hidetrial(self, trial):
        return False

    # Indicates if you want to call the parseline() function during blinks
    def parseblinks(self):
        return False

    # Indicates the number of pixels per degree. This must be overridden, because it could lead to gross errors in the analysis.
    def px_per_deg(self):
        print "Please override Parser.px_per_deg()"
        assert False

    # Returns a tuple containing the resolution
    def resolution(self):
        print "Please override Parser.resolution()"
        assert False

    # This function specifies a number of criteria in SQL syntax which sould be used to filter the data
    def criteria(self):
        return []

    # This function may mark trials as outliers based on arbitrary criteria (such as RT being 2.5 x STD from the mean of some group)
    def mark_outliers(self, db, sql):
        return

    # This function specifices a number of SQL strings to be used for summarizing the data
    def summary(self):
        if os.path.exists("summary.sql"):
            f = open("summary.sql")
            return f.read().split(";")
        else:
            return ("select FILE, count(*) as N from %s group by FILE" % self.table(),)

    # Specifiy which lines should be interpreted as the start of a trial
    def start_of_trial(self, word):
        return ((len(word) == 3 or len(word) == 4) and word[2] == "start_trial");

    # Extract the trial id from a string
    def trial_id(self, word):
        return int(word[1])

    # Specifiy which lines should be interpreted as the end of a trial
    def end_of_trial(self, word):
        return len(word) == 3 and word[2] == "stop_trial";

    # These are some default mysql connection parameters, which can be overriden if necessary
    def user(self):
        return "eyelink"

    def password(self):
        return "eyelink"

    def database(self):
        return "EyelinkData"

    def table(self):
        return "data"

    def host(self):
        return "localhost"

    # Parse variables, by default lines of the form MSG __timestamp__ VARIABLE __varname__ __value__
    def parsevariables(self, trial, word):
        if len(word) == 6  and word[2] == "!V" and word[3] != "TRIAL_VAR":
            trial[word[4]] = word[5]
        elif len(word) == 5  and word[2] == "variable":
            trial[word[3]] = word[4]
    # Parse display messages, by default lines of the form MSG __timestamp__ 0 __DISPLAY...__ __ __
    def parsedisplays(self, trial, word):
        if len(word) >3  and word[3] == "search" and word[0] == "MSG":
            trial[word[2]] = word[1]




    # This function can be used to process curvature points
    def process_curvature_points(self, trial, norm, sample, start, end, dsample, norm_asample):
      None

    ###
    ### DO NOT OVERRIDE THESE METHODS
    ###

    def px_to_deg_x(self, x):
        return (float(x) - self.resolution()[0] / 2) / self.px_per_deg()

    def px_to_deg_y(self, y):
        return (float(y) - self.resolution()[1] / 2 ) / self.px_per_deg()

    def deg_to_px_x(self, x):
        return round(x * self.px_per_deg() + self.resolution()[0] / 2)

    def deg_to_px_y(self, y):
        return round(y * self.px_per_deg() + self.resolution()[1] / 2)

    # See: http://codecomments.wordpress.com/2008/03/17/simple-method-to-calculate-median-in-python/
    def get_median(self, numericValues):
        theValues = sorted(numericValues)

        if len(theValues) % 2 == 1:
            return theValues[(len(theValues)+1)/2-1]
        else:
            lower = theValues[len(theValues)/2-1]
            upper = theValues[len(theValues)/2]

        return (float(lower + upper)) / 2
        
    # Bronagh experiment
    def get_peak_deviance(self, numericValues):       
        theValues = numericValues
        
        pos_peak = max(theValues)
        neg_peak = min(theValues)
        
        if pos_peak > abs(neg_peak):
            return float(pos_peak)
        elif pos_peak < abs(neg_peak):
            return float(neg_peak)
        else:
            return 0
        

    # Checks if a given line represents a saccade and returns a dictionary with the attributes of the saccade
    def saccade(self, word):
        if len(word) == 0 or not word[0] == "ESACC" or not word[1] == "R":
             return False

        try:

            saccade = {}

            # The eyelink has two different ways to encode saccades
            if len(word) == 15:

                saccade["start_x_px"] = int(word[9])
                saccade["start_y_px"] = int(word[10])
                saccade["end_x_px"] = int(word[11])
                saccade["end_y_px"] = int(word[12])

            elif len(word) == 11:

                saccade["start_x_px"] = int(word[5])
                saccade["start_y_px"] = int(word[6])
                saccade["end_x_px"] = int(word[7])
                saccade["end_y_px"] = int(word[8])
                
            else:
                print "Unknown format for saccade"
                assert False

            saccade["start_x_deg"] = self.px_to_deg_x(saccade["start_x_px"])
            saccade["start_y_deg"] = self.px_to_deg_y(saccade["start_y_px"])
            saccade["end_x_deg"] = self.px_to_deg_x(saccade["end_x_px"])
            saccade["end_y_deg"] = self.px_to_deg_y(saccade["end_y_px"])

            saccade["size_px"] = self.distance(saccade["start_x_px"], saccade["start_y_px"], saccade["end_x_px"], saccade["end_y_px"])
            saccade["size_deg"] = self.distance(saccade["start_x_deg"], saccade["start_y_deg"], saccade["end_x_deg"], saccade["end_y_deg"])

            saccade["start_time"] = int(word[2])
            saccade["end_time"] = int(word[3])
            saccade["duration"] = saccade["end_time"] - saccade["start_time"]
            
            return saccade

        # If an exception occurs, this most likely reflects the occurrence of a "." instead of a coordinate, due to whatever error that the eyelink encountered.
        except ValueError:
            return False

        return False

    # Checks if a given line represents a fixation and returns a dictionary with the attributes of the fixation
    def fixation(self, word):
        if not len(word) == 8 or not word[0] == "EFIX":
            return False

        fixation = {}

        fixation["x_px"] = int(word[5])
        fixation["y_px"] = int(word[6])

        fixation["x_deg"] = self.px_to_deg_x(fixation["x_px"])
        fixation["y_deg"] = self.px_to_deg_y(fixation["y_px"])

        fixation["start_time"] = int(word[2])
        fixation["end_time"] = int(word[3])
        fixation["duration"] = fixation["end_time"] - fixation["start_time"]

        return fixation

    # Checks if a given line respresents a sample
    def sample(self, word):
        if not len(word) == 8 and not len(word) == 5:
            return False
        try:

            sample = {}
            sample["time"] = int(word[0])

            sample["x_px"] = int(word[1])
            sample["y_px"] = int(word[2])

            sample["x_deg"] = self.px_to_deg_x(sample["x_px"])
            sample["y_deg"] = self.px_to_deg_y(sample["y_px"])

            return sample
        except:
            return False

    # Indicate that curvature should be determined starting now
    def start_curvature(self, curvature_ref = None):
        self.determine_curvature = True
        self.curvature_ref = curvature_ref
        self.stored_samples = []

    def stop_curvature(self, trial, curvature_ref = None):


        """
        Curvature is defined as the median angle between the line from the
        point
        to the saccade end point and the line from the
        point to the gaze sample.
        The saccade start and end points can be specified or left open, in which case the
        positions as reported by the eyelink are used.
        """
        
        if curvature_ref != None:
            self.curvature_ref = curvature_ref

        d_threshold = 0 #BM: cuts this much off start and end of saccade to analyze. Originally was self.px_per_deg() / 2.
        min_length = 10 #BM: Minimum # of samples required in saccade, in order to analyze it.
        
        self.determine_curvature = False

        x = self.deg_to_px_x(self.curvature_ref[0])
        y = self.deg_to_px_y(self.curvature_ref[1])
        tx = self.deg_to_px_x(self.curvature_ref[2])
        ty = self.deg_to_px_y(self.curvature_ref[3])

        a = math.atan2(ty - y, tx - x) # BM : angle (in radians) between saccade start and end with respect to due north
        da_store = []
        da_deviance = [] # BM : created for looking at peak deviance
        
        i = 0
        
        jump_size = float(math.sqrt( (tx - x)**2 + (ty - y)**2 )) # BM : distance from saccade start to end
        
        # Make an image with the saccade start and end point, connected by a line

        im = Image.new("RGB", (int(jump_size) + 100, 100), (0, 0, 0))
        dr = ImageDraw.Draw(im)
        dr.ellipse( (48, 48, 52, 52), fill = (255, 0, 0) )
        dr.ellipse( (jump_size + 48, 48, jump_size + 52, 52), fill = (255, 0, 0) )
        dr.line( (50, 50, 50 + jump_size, 50), fill = (255, 0, 0) )

        for sx, sy in self.stored_samples:
            
            i += 1
            
            d = math.sqrt( (sx - x)**2 + (sy - y)**2 ) # BM : distance from current sample to starting point
            dt = math.sqrt( (sx - tx)**2 + (sy - ty)**2 ) # BM : distance from current sample to target
            
            t = math.atan2(sy - y, sx - x) # BM: angle (in radians) between current sample and starting point
            da = t - a # BM : relative angle (in radians) between current sample and straight line joining saccade start and endoint
            
            while da < -math.pi:
                da += 2 * math.pi

            while t > math.pi:
                da -= 2 * math.pi

            norm_x = d * math.cos(da) # current distance along the straight line joining starting point to target, for the given sample
            norm_y = d * math.sin(da) # current deviance from the straight line, for the given sample.

            if norm_x > d_threshold and norm_x < jump_size - d_threshold:
                
                dr.point( (50 + norm_x, 50 + norm_y), fill = (0, 255, 0) )
                da_store.append(da)
                da_deviance.append(norm_y) # BM : created this for looking at peak deviance
                self.process_curvature_points(trial, (norm_x, norm_y), (sx, sy), (x, y), (tx, ty), d, da)
                
            if i == 16:
                trial["EarlyX"] = sx
                trial["EarlyY"] = sy

        #print len(da_store)
        if len(da_store) > min_length:
            print 'da_store: ',len(da_store)
            # Median curvature
            #curvature = math.degrees(self.get_median(da_store))  #BM : Get median of whole saccade
            # Peak deviance curvature            
            curvature = math.degrees(self.get_peak_deviance(da_deviance))/math.degrees(jump_size) # BM : Get peak deviance in either pos or neg direction, then divide by the saccade amplitude (Nummenmaaa, 2006)
            # Average curvature
            curvature_avg = math.degrees(sum(da_store[:(len(da_store)/2)]) / (len(da_store)/2)) #BM : Get average of first half of every saccade
            
            
            while (curvature < -180):
                curvature += 360                

            while (curvature > 180):
                curvature -= 180               

            while (curvature_avg < -180):               
                curvature_avg += 360

            while (curvature_avg > 180):                
                curvature_avg -= 180 

        else:
            curvature = 0
            curvature_avg = 0
    
        if curvature != 0:
            print 'curvature: ',curvature
            
        return curvature
            
     #   trial["sacc_curvature_avg"] = curvature_avg

        #dr.text( (10, 10), "Curvature = %.2f" % curvature, (255, 255, 255) )
       # im.save(os.path.join("png", "curvature_%s_%d.png" % (trial["FILE"], trial["TRIALID"])), "PNG")


    # Collect samples until the end of the saccade and determine curvature
    def collect_samples(self, trial, word):

        # Collect samples
        sample = self.sample(word)
        if sample != False:
            self.stored_samples.append( (sample["x_px"], sample["y_px"]) )


    # Returns the distance between two points
    def distance(self, startx, starty, endx, endy):
        # Pythagoras
        return math.sqrt( (startx - endx) * (startx - endx) + (starty - endy) * (starty - endy) )

    # Returns the angle of the line between two points
    # Angle is measured as the counterclockwise deviation from the X-axis
    def angle(self, startx, starty, endx, endy):
        return math.atan2(endy - starty, endx - startx)

    # Returns the rotational difference between two angles in degrees
    def d_deg(self, a1, a2):
        d = abs(a1 - a2) % 360
        return min(d, 360 - d)

    # Returns the rotational difference between two angles in degrees
    def d_rad(self, a1, a2):
        d = abs(a1 - a2) % (2 * math.pi)
        return min(d, (2 * math.pi) - d)

    # For some special attributes we already know what type they should be
    def attrtypewrapper(self, attr):
        if attr == "EXP_TRIALID":
            return "varchar(10)"
        if attr == "FILE":
            return "varchar(20)"
        if attr in ["sacc_curvature", "ref_sacc_curvature"]:
            return "double"

        if type(attr) == str:
            if "_GROUP" in attr:
                return "varchar(100)"
            # Medians have the same type as the attribute they are the median of
            if "_MEDIAN" in attr:
                return self.attrtype(attr[:-7])

            # Percentile scores
            if "_PERC" in attr:
                return "double"
        return self.attrtype(attr)

    def sort_by_key(self, t1, t2):

        # If possible treat the values as floating point numbers, so we don't get something like: "3" > "20"
        try:
            if float(t1[self.sort_key]) < float(t2[self.sort_key]):
                return -1
            elif float(t1[self.sort_key]) == float(t2[self.sort_key]):
                return 0
            else:
                return 1
        except:
            if t1[self.sort_key] < t2[self.sort_key]:
                return -1
            elif t1[self.sort_key] == t2[self.sort_key]:
                return 0
            else:
                return 1

    # Parse the contents of a single file and write them to a mysql table
    def parse(self, file):
        print "Parsing %s ..." % file,

        trial = {}
        trialid = -1
        parsing = False
        blink = False
        self.nroftrials = 0


        # Process the file line-by-line
        for line in open(os.path.join("/Users/bronagh/Documents/LePelley/LePelley_2/data/asc/asc_LP24", file)):

            # Split the line into separate words and convert these into floats and integers if possible
            str_word = line.split()
            word = []
            for term in str_word:
                try:
                    if "." in term:
                        word.append(float(term))
                    else:
                        word.append(int(term))
                except:
                    word.append(term)


            # Catch the start of a trial
            if self.start_of_trial(word):                
                self.nroftrials = self.nroftrials + 1
                print "Trial {0}".format(self.nroftrials)
                self.determine_curvature = False
                parsing = True
                trialid = self.trial_id(word)
                trial[trialid] = {"TRIALID" : trialid, "FILE" : file, "OUTLIER" : 0, "TRIAL_START" : int(word[1])}
                # Process all the attributes that are given on the trialid line
                nextisname = True
                for attr in word[5:]:
                    if nextisname:
                        nextisname = False
                        attrname = attr
                    else:
                        nextisname = True
                        trial[trialid][attrname] = attr

                # Initialize the trial
                self.initialize(trial[trialid])

            # Register blinks
            if len(word) > 0 and word[0] == "SBLINK":
                blink = True

            if len(word) > 0 and word[0] == "EBLINK":
                blink = False

            if parsing:
                # Parse all variables
                self.parsevariables(trial[trialid], word)


            # Pass the rest of the parsing on to a experiment specific subroutine
            # Eye movement information during blinks should be ignored (based on parseblinks()), but messages not.
            if parsing and ((len(word) > 0 and word[0] == "MSG") or not blink or self.parseblinks()):          
                self.parseline(trial[trialid], word)
                
                if self.determine_curvature:
                    self.collect_samples(trial[trialid], word)
                    
            
            
            if self.end_of_trial(word):
                parsing = False
                trial[trialid]["TRIAL_END"] = int(word[1])
                self.finalize(trial[trialid])

                # Remove trial there are to be hidden (such as practice trials)
                if self.hidetrial(trial[trialid]):
                    del trial[trialid]


        print "%s trials" % self.nroftrials
        return trial

     # Constructor
    def __init__(self):

        start_time = time.time()

        print
        print "*** Data parser"
        print "*** Sebastiaan Mathot (2008-2010)"
        print ' Wouter changed it so that it could run without the MySQL bits '.center(35,'.')
        print
        print time.strftime("%c")
        print
        print "Description:", self.description()
        print

        skip_parsing = False
        only_parsing = False
        self.print_trial_ids = False

        # Skip parsing if there is already a datafile in the folder
        #if os.path.exists("data.csv"):
        #    skip_parsing = True
            
        # Process the command line arguments
        for arg in sys.argv:
            if arg == "-s":
                skip_parsing = True
            if arg == "--no-mysql":
                only_parsing = True
            if arg == "--trial-ids":
                self.print_trial_ids = True

        if skip_parsing:
            print "Skipping parsing, reading data from table %s" % self.table()
            print
            print "Proceeding directly to data summary"

            #import _mysql
        else:

            # Walk through all files in the current directory, but analyze them only if they are .asc files
            session = {}
            nroffiles = 0
            for file in os.listdir("/Users/bronagh/Documents/LePelley/LePelley_2/data/asc/asc_LP24"):
                (self.root, ext) = os.path.splitext(file)
                if ext == ".asc":
                    nroffiles = nroffiles + 1
                    session[file] = self.parse(file)

            # A quick hack to stop before we do the sql stuff, so we can run this on machines without mysql # TODO WK: here is where my stuff should go
            if only_parsing:
                print "Requested to stop after parsing. Stopping now...\n"
                return

           # import _mysql

            # Some trial may contain a list of dictionaries as attributes, register those
            # Also, some trials may be hidden, register those too
            expand = []
            hide = []
            for file, s in session.iteritems():
                for trialid, trial in s.iteritems():
                    for attrname, attrval in trial.iteritems():
                        if isinstance(attrval, list):
                            expand.append((attrname, file, trialid))
                        if self.hideattr(attrname):
                            hide.append((attrname, file, trialid))

            # Remove all trials that should be hidden
            for attrname, file, trialid in hide:
                del session[file][trialid][attrname]
            '''
            # Now expand all attributes that have been found to be a list
            for attrname, file, trialid in expand:

                # Create a new trial for each dictionary in the list
                c = 0

                for dic in session[file][trialid][attrname]:
                    newtrialid = "%s_%s" % (trialid, c)
                    newtrial = session[file][trialid].copy()
                    newtrial["EXP_TRIALID"] = newtrialid
                    del newtrial[attrname]

                    # Add all items from the dictionary as separate attributes
                    for attr in dic:
                        newtrial[attr] = dic[attr]
                    session[file][newtrialid] = newtrial
                    c = c + 1

                del session[file][trialid]
            '''

            # Some trials may have more attributes than others, so we need to get an exhaustive set of all attributes.
            # Just walk through all attributes of all trials and add them to the attrs list if they aren't already in there.
            attrs = []
            for file, s in session.iteritems():
                for trialid, trial in s.iteritems():
                    for attrname, attrval in trial.iteritems():
                        if not attrname in attrs:
                            attrs.append(attrname)


            # For some attributes, percentile scores may be requested
            # Stats will be calculated later, but register fields now so they are created in the database
            addedattrs = []
            first = True
            for attr in attrs:
                if not self.attrstats(attr) == False:
                    attrs.append("%s_PERC" % attr)
                    attrs.append("%s_MEDIAN" % attr)
                    attrs.append("%s_GROUP" % attr)

            # Connect to the mysql database
           # db = _mysql.connect(self.host(), self.user(), self.password(), self.database())

            # We're going to write all sql to data.sql and all data to data.csv
           # sql = open("data.sql", "w")
            dat = open("data.csv", "w")
           # sql.truncate(0)
            dat.truncate(0)

            # (Re)create the table
           # query = "drop table if exists %s" % self.table()
           # db.query(query)
           # sql.write("%s\n" % query)

            attrlist = ""

            # List all attributes
            comma = ""
            for attr in attrs:
                attrlist = "%s%s%s %s" % (attrlist, comma, attr, self.attrtypewrapper(attr))
                dat.write("%s%s" % (comma, attr))
                comma = ","

           # query = "create table %s (%s)" % (self.table(), attrlist)         
           # db.query(query)            
           # sql.write("%s\n" % query)
            dat.write("\n")

           # print query

            # If we have expanded the data, also create a table for a collapsed version of the data
           # if len(expand) > 0:
           #     query = "drop table if exists %s_collapsed" % self.table()
           #     db.query(query)
           #     sql.write("%s\n" % query)

               # query = "create table %s_collapsed (%s)" % (self.table(), attrlist)
               # db.query(query)
               # sql.write("%s\n" % query)
    
            
            # Now store all data to a mysql database
            for file, s in session.iteritems():
                for trialid, trial in s.iteritems():
                   # query = "insert into %s (" % self.table()

                   # # List all attributes
                   # comma = ""
                   # for attr in attrs:
                   #         query = "%s%s%s" % (query, comma, attr)
                   #         comma = ", "


                   # query = "%s) values (" % query

                    # List all attribute values, writing 0 if none is found
                    #print "HEYUEEUHU!!! " 
                    #comma = ""
                    
                    line = ''
                    for attr in attrs:
                        if attr in trial:
                           # query = "%s%s\"%s\"" % (query, comma, trial[attr])
                           line = line + "," + str(trial[attr])
                            
                        else:
                           # query = "%s%s\"0\"" % (query, comma)
                           line = line + ", "
                        comma = ","
                    #print line

                    
                    comma = ","
                    for attr in attrs:
                        if attr in trial:
                           # query = "%s%s\"%s\"" % (query, comma, trial[attr])
                            dat.write("%s%s" % (comma, trial[attr]))
                            
                        else:
                           # query = "%s%s\"0\"" % (query, comma)
                            dat.write("%s0" % comma)
                        comma = ","

                   # query = "%s)" % query

                   # db.query(query)
                   # sql.write("%s\n" % query)
                    dat.write("\n")

            # We may implement a different way to determine outliers
           # self.mark_outliers(db, sql)

            # How many trials were marked as outliers during the parsing phase?
       #     query = "select count(*) as n from %s" % self.table()
       #     db.query(query)
       #     sql.write("%s\n" % query)
       #     n_all = int(db.store_result().fetch_row()[0][0])

       #     query = "select count(*) as n from %s where Outlier = 1" % self.table()
       #     db.query(query)
       #     sql.write("%s\n" % query)
       #     n_aff = int(db.store_result().fetch_row()[0][0])

       #     print "Filtered %s of %s (%.2f%%) trials during parsing" % (n_aff, n_all, (float(n_aff) / n_all) * 100)

       #      Now we're going to filter the data according to specified criteria
       #     if type(self.criteria()) != list:
       #         raise Exception("Criteria should be a list of strings.")

       #     for criterium in self.criteria():
       #         query = "select count(*) as n from %s where Outlier = 0" % self.table()
       #         db.query(query)
       #         sql.write("%s\n" % query)
       #         n_tot = int(db.store_result().fetch_row()[0][0])

       #         query = "select FILE, count(*) as n, group_concat(TRIALID) as trials from %s where Outlier = 0 and %s group by FILE" % (self.table(), criterium)
       #         db.query(query)
       #         sql.write("%s\n" % query)
       #         result = db.store_result()
       #         print "Filtering using %s" % criterium

       #         for row in result.fetch_row(maxrows=0):
       #             if self.print_trial_ids:
       #                 print "%s\t%s\t%s" % (row[0], row[1], row[2])
       #             else:
       #                 print "%s\t%s" % (row[0], row[1])

       #         query = "update %s set Outlier = 1 where %s" % (self.table(), criterium)
       #         db.query(query)
       #         sql.write("%s\n" % query)
       #         n_aff = int(db.affected_rows())
       #         if n_tot > 0:
       #             print "Total: %s of %s (%.2f%%)" % (n_aff, n_tot, (float(n_aff) / n_tot) * 100)
       #         else:
       #             print "Total: %s of %s (0%%)" % (n_aff, n_tot)
       #         print

       #     # Determine median and percentile scores
       #     for attr in attrs:
       #         if not self.attrstats(attr) == False:
       #             print "Calculating median and percentile scores for %s ... " % attr

       #             group_attrs = self.attrstats(attr)
       #             attrlist = ", ".join(group_attrs)
       #             query = "select %s, %s from data where Outlier = 0" % (attr, attrlist)
       #             db.query(query)
       #             sql.write("%s\n" % query)

       #             # Create a dictionary of list of trials
       #             result = db.store_result()
       #             data_by_group = {}
       #             for row in result.fetch_row(maxrows=0, how=1):
       #                 group_key = ""
       #                 for group_attr in group_attrs:
       #                     group_key = "%s_%s" % (group_key, row[group_attr])
       #                 if group_key not in data_by_group:
       #                     data_by_group[group_key] = []
       #                 data_by_group[group_key].append(row)

       #             query_file = open("tmp.sql", "w")
       #             for group_key in data_by_group:
       #                 self.sort_key = attr
       #                 data_by_group[group_key].sort(self.sort_by_key)
       #                 median = None

       #                 for trial in data_by_group[group_key]:
       #                     group_where = ""
       #                     for group_attr in group_attrs:
       #                         group_where = "%s and %s = \"%s\"" % (group_where, group_attr, trial[group_attr])
       #                     pos = data_by_group[group_key].index(trial)
       #                     perc = float(pos) / len(data_by_group[group_key])
       #                     if median == None and perc >= .5:
       #                         median = float(trial[attr])
       #                         query = "update %s set %s_MEDIAN = %f where 1 %s" % (self.table(), attr, median, group_where)
       #                         db.query(query)
       #                         sql.write("%s\n" % query)
       #                     query = "update %s set %s_PERC = %f, %s_GROUP = \"%s\" where %s = %s%s;\n" % (self.table(), attr, perc, attr, group_key, attr, trial[attr], group_where)
       #                     query_file.write(query)
       #                     db.query(query)
       #                     sql.write("%s\n" % query)
       #             query_file.close()
       #             print "Committing changes ... "
       #             #os.system("mysql -ueyelink -peyelink EyelinkData < tmp.sql")
       #             print "Done"
       #             print

       #             """
       #             f = open("test.csv", "w")
       #             query = "select * from data"
       #             db.query(query)
       #             result = db.store_result()
       #             for row in result.fetch_row(maxrows=0, how=1):
       #                 for key in row:
       #                     f.write(key + ", ")
       #             f.close()
       #             """


       #     # Display the valid trials per file
       #     print
       #     query = "select FILE, sum(Outlier = 0) as Outlier, count(*) as N from %s group by FILE" % self.table()
       #     db.query(query)
       #     sql.write("%s\n" % query)
       #     for file, outlier, n in db.store_result().fetch_row(maxrows=0):
       #         print "%s of %s (%.2f%%) valid trials in %s" % (outlier, n, (float(outlier) / float(n)) * 100, file)


       #     query = "select count(*) as n from %s where Outlier = 0" % self.table()
       #     db.query(query)
       #     sql.write("%s\n" % query)
       #     n_tot = int(db.store_result().fetch_row()[0][0])

       #     print
       #     print "%s of %s (%.2f%%) valid trials" % (n_tot, n_all, (float(n_tot) / n_all) * 100)

       #     # Now, if needed, store a collapsed version of the data to [tablename]_collapsed
       #     if len(expand) > 0:
       #         query = "insert into %s_collapsed select * from %s group by FILE, TRIALID" % (self.table(), self.table())
       #         db.query(query)
       #         sql.write("%s\n" % query)

       # # If parsing has been skipped we still have to establish a connection and open the log files
       # if skip_parsing:
       #     # Connect to the mysql database
       #     db = _mysql.connect(self.host(), self.user(), self.password(), self.database())
       #     # Open the sql log
       #     sql = open("data.sql", "w")

       # # Start the data summary
       # print
       # print "Data summary:"

       # # Get all the queries, execute them and the display them in an orderly manner
       # for query in self.summary():
       #     query = query.strip(" \t\n")

       #     # Print comments
       #     if len(query) > 0 and query[0] == "#":
       #         print query[1:].strip()

       #     # Send everything else to mysql
       #     elif not query == "":
       #         db.query(query)
       #         sql.write("%s\n" % query)
       #         r = db.store_result()
       #         firstrow = True
       #         if r != None:
       #             for row in r.fetch_row(how=1, maxrows=0):
       #                 if firstrow:
       #                     firstrow = False
       #                     print
       #                     firstcolumn = True
       #                     for name in row:
       #                         if firstcolumn:
       #                             firstcolumn = False
       #                         else:
       #                             print "\t",
       #                         print name,
       #                     print
       #                 firstcolumn = True
       #                 for name in row:
       #                     if firstcolumn:
       #                         firstcolumn = False
       #                     else:
       #                         print "\t",

       #                     # Round float numbers
       #                     try:
       #                         if "." in row[name]:
       #                             print "%.3f" % float(row[name]),
       #                         else:
       #                             print row[name],
       #                     except:
       #                         print row[name],
       #                 print

       # # Close the connection to the mysql database and close data.sql and data.csv
       # db.close()
        # sql.close()

        # if not skip_parsing:
        #     dat.close()

        # If data and code is defined, outsource the stats to R
       # if os.path.exists("rinput.sql") and os.path.exists("rscript"):
       #     os.system("mysql -ueyelink -peyelink EyelinkData -B -e \"`cat rinput.sql`\" | sed 's/\\t/,/g;s/^//;s/$//;s/\\n//g' > rinput.txt")
       #     os.system("Rscript rscript > statistics.txt")
       #     print
        #    print "R-statistics saved in statistics.txt"

        # if skip_parsing:
        #     print
        #     print "SQL-statements have been written to data.sql"
        # else:
        #     print
        #     print "Data has been written to data.csv, SQL-statements have been written to data.sql"
        #     print
        #     print "Parsed and stored %s files, with %s attributes per trial" % (nroffiles, len(attrs))

        # stop_time = time.time() - start_time
        # print
        # print "Script finished in %.2f seconds" % stop_time