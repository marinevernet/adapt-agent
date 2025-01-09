import os
import pandas as pd
import numpy as np
from datetime import datetime


_mainDirectory = "sessions"
# ID = "147589736_147589736" # first participant (for debugging)

global _last10_VCT, _last10_withoutClick, _last10_TBC
global _num_total_trial, _num_trials_with_warning, _num_unsuccessful_trials , _num_nan_trials, _num_successful_trials, _average_timeBetweekClick
global _durationGame, _durationExperimentalBlocks

_last10_VCT = list()
_last10_withoutClick = list()
_last10_TBC = list()

_num_total_trial = list()
_num_trials_with_warning = list()
_num_perceived_click_first = list()
_num_unsuccessful_trials = list()
_num_nan_trials = list()
_num_successful_trials = list()
_average_timeBetweekClick = list()
_average_virtualClickTime = list()

_durationGame = list()
_durationExperimentalBlocks = list()

# Function to analyze an individual's results
# -------------------------------------------
def analyseSubject(ID):

    # name of the file
    print("# " + ID + " #")
    filepath = _mainDirectory + '/' + ID + '_experiment_results.txt'


    # Last 10 trials of the training block
    # -------------------------------------------        
    # loading the data of the first (training) block
    df = pd.read_csv(filepath, sep=';')
    df = df[df['block'] == 1] # keep only the training block
    df = df[df['popupType'].apply(lambda x: isinstance(x, float))] # remove the trials with warnings

    # in the last 10 trials
    _last10_VCT.append(df[-10:]['virtualClickTime'].mean()) # average virtualClickTime
    _last10_withoutClick.append(df[-10:]['humanClickTime'].isna().sum()/10) # ratio of NaN in humanClickTime
    _last10_TBC.append(df[-10:]['timeBetweenClicks'].mean()) # average timeBewteenClick

    # Number of trials in the experimental blocks
    # -------------------------------------------  
    # loading the data of the experimental blocks
    df=pd.read_csv(filepath, sep=';')
    df.drop(df[df['block'] <2].index, inplace = True) # remove the training trials and training block
    df.drop(df[df['block'] >7].index, inplace = True) # remove the extra-blocks, if any
    
    # calculating number of relevent trials
    num_total_trial = len(df)
    df = df[df['popupType'].apply(lambda x: isinstance(x, float))] # remove the trials with warnings
    num_trials_witout_warning = len(df)
    num_trials_with_warning = num_total_trial - num_trials_witout_warning
    num_perceived_click_first = len(df[df['perceivedFirstClickType']==1])
    df.drop(df[df['timeBetweenClicks'] >0].index, inplace = True) # keep only successful algorithm
    num_successful_trials1 = len(df)
    num_unsuccessful_trials = num_trials_witout_warning - num_successful_trials1
    columns = ['block', 'searchTime', 'features_peakSpeed', 'features_accelerationTime', 'features_distance', 'features_waitingTime', 'humanClickTime', 'virtualClickTime', 'distanceToTargetWhenHumanClick', 'distanceToTargetWhenVirtualClick', 'timeBetweenClicks', 'perceivedFirstClickType']
    df = df[columns]
    df = df.rename(columns={'features_peakSpeed': 'peakSpeed', 'features_accelerationTime': 'accelerationTime', 'features_distance': 'distance', 'features_waitingTime': 'waitingTime'})
    df = df.dropna()
    num_successful_trials = len(df)
    num_nan_trials = num_successful_trials1 - num_successful_trials
    _num_total_trial.append(num_total_trial)
    _num_trials_with_warning.append(num_trials_with_warning)
    _num_perceived_click_first.append(num_perceived_click_first)
    _num_unsuccessful_trials.append(num_unsuccessful_trials)
    _num_nan_trials.append(num_nan_trials)
    _num_successful_trials.append(num_successful_trials)
    _average_timeBetweekClick.append(df['timeBetweenClicks'].mean())
    _average_virtualClickTime.append(df['virtualClickTime'].mean())

    # Duration of the game
    # -------------------------------------------  
    # loading the file
    df = pd.read_csv(filepath, sep=';')
    # duration of the entire file
    startTime = datetime.strptime(df.iloc[0]['startTime'], '%d/%m/%Y, %H:%M:%S:%f') 
    endTime = datetime.strptime(df.iloc[-1]['endTime'], '%d/%m/%Y, %H:%M:%S:%f') 
    _durationGame.append((endTime-startTime).seconds/60)
    # duration of the experimental blocks
    df.drop(df[df['block'] <2].index, inplace = True) # remove the training trials and training block
    df.drop(df[df['block'] >7].index, inplace = True) # remove the extra-blocks, if any
    startTime = datetime.strptime(df.iloc[0]['startTime'], '%d/%m/%Y, %H:%M:%S:%f') 
    endTime = datetime.strptime(df.iloc[-1]['endTime'], '%d/%m/%Y, %H:%M:%S:%f') 
    _durationExperimentalBlocks.append((endTime-startTime).seconds/60)

# MAIN
# -------------------------------------------

directory = os.fsencode(_mainDirectory)
doneIDs = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if (filename[-4:] != ".txt"): continue
    if (filename.split('_')[2] != 'experiment'): continue
    fileSessionID = filename.split('_')[1]
    fileSubjectID = filename.split('_')[0]
    ID = fileSubjectID + '_' + fileSessionID
    if (ID=='203792202_203792202'): continue # remove participant with ADHD diagnosis
    if (ID=='99653929_99653929'): continue # remove participant who always answered she won
    if (ID=='45996241_45996241'): continue # remove participant who answered only twice that it was not hear
    if ID not in doneIDs:
        doneIDs.append(ID)
        analyseSubject(ID)
    else:
        continue

_last10_VCT = np.array(_last10_VCT)
_last10_withoutClick = np.array(_last10_withoutClick)
_last10_TBC = np.array(_last10_TBC)

print('\nLast ten trials of the training block')
print('average virtual click time = ' + str(_last10_VCT.mean()) + ' +/- ' + str(_last10_VCT.std()))
print('ratio of trials without human click = ' + str(_last10_withoutClick.mean()) + ' +/- ' + str(_last10_withoutClick.std()))
print('when human click, average time between clicks = ' + str(np.nanmean(_last10_TBC)) + ' +/- ' + str(np.nanstd(_last10_TBC)))


_num_total_trial = np.array(_num_total_trial)
_num_trials_with_warning = np.array(_num_trials_with_warning)
_num_perceived_click_first = np.array(_num_perceived_click_first)
_num_unsuccessful_trials = np.array(_num_unsuccessful_trials)
_num_nan_trials = np.array(_num_nan_trials)
_num_successful_trials = np.array(_num_successful_trials)
_average_timeBetweekClick = np.array(_average_timeBetweekClick)
_average_virtualClickTime = np.array(_average_virtualClickTime)

ratio_successful_trials = _num_successful_trials/480

print('\nDuring the experimental blocks')
print('number of total trial = ' + str(_num_total_trial.mean()) + ' +/- ' + str(_num_total_trial.std()))
print('number of trials with warnings = ' + str(_num_trials_with_warning.mean()) + ' +/- ' + str(_num_trials_with_warning.std()))
print('number of perceived click first (over 480) = ' + str(_num_perceived_click_first.mean())+ ' +/- ' + str(_num_perceived_click_first.std()))
print('minimum and maximum number of perceived click first (over 480) = ' + str(_num_perceived_click_first.min())+ ' - ' + str(_num_perceived_click_first.max()))
print('number of unsuccessful trials (no virtucal click before human click) = ' + str(_num_unsuccessful_trials.mean()) + ' +/- ' + str(_num_unsuccessful_trials.std()))
print('number of nan trials (participant not clicking) = ' + str(_num_nan_trials.mean()) + ' +/- ' + str(_num_nan_trials.std()))
print('number of successful trials (virtual click before human click) = ' + str(_num_successful_trials.mean()) + ' +/- ' + str(_num_successful_trials.std()))
print('ratio successful trials (virtual click before human click) = ' + str(ratio_successful_trials.mean()) + ' +/- ' + str(ratio_successful_trials.std()))
print('average time between click = ' + str(_average_timeBetweekClick.mean()) + ' +/- ' + str(_average_timeBetweekClick.std()))
print('average virtual click time = ' + str(_average_virtualClickTime.mean()) + ' +/- ' + str(_average_virtualClickTime.std()))


_durationGame = np.array(_durationGame)
_durationExperimentalBlocks = np.array(_durationExperimentalBlocks)

print('\nDuration of the game')
print('Entire game = ' + str(_durationGame.mean()) + ' +/- ' + str(_durationGame.std()))
print('Experimental blocks = ' + str(_durationExperimentalBlocks.mean()) + ' +/- ' + str(_durationExperimentalBlocks.std()))


