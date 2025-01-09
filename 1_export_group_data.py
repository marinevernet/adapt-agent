
import os
import pandas as pd
import numpy as np
import json

# ID = "147589736_147589736"
# ID = "101286468_101286468"
_mainDirectory = "sessions"


def reading_data_all_exp(filepath):

    df=pd.read_csv(filepath, sep=';')

    df.drop(df[df['block'] <2].index, inplace = True) # remove the training trials and training blocks
    df.drop(df[df['block'] >7].index, inplace = True) # remove the extra-blocks, if any

    return df


def reading_data_int_exp(filepath):

    df=pd.read_csv(filepath, sep=';')

    df.drop(df[df['block'] <2].index, inplace = True) # remove the training trials and training blocks
    df.drop(df[df['block'] >7].index, inplace = True) # remove the extra-blocks, if any
    df = df[df['popupType'].apply(lambda x: isinstance(x, float))] # remove the trials with warnings
    df.drop(df[df['timeBetweenClicks'] >= 0].index, inplace = True) # keep only successful algorithm
    columns = ['block', 'searchTime', 'features_waitingTime', 'features_peakSpeed', 'features_accelerationTime', 'features_distance', 'humanClickTime', 'virtualClickTime', 'distanceToTargetWhenHumanClick', 'distanceToTargetWhenVirtualClick', 'timeBetweenClicks', 'perceivedFirstClickType']
    df = df[columns]
    df = df.rename(columns={'features_waitingTime': 'waitingTime', 'features_peakSpeed': 'peakSpeed', 'features_accelerationTime': 'accelerationTime', 'features_distance': 'distance'})
    df = df.dropna() # remove trials for which the participant didn't click

    return df


def extract_one_line_participant_exp_data(ID):

    # reading the file
    filepath = _mainDirectory + '/' + ID + '_experiment_results.txt'
    df=pd.read_csv(filepath, sep=';')

    # kept trials: trials without warning (80 trials per block)
    df_kept = df[df['popupType'].apply(lambda x: isinstance(x, float))]

    # good trials: successful trials, i.e. virtual click before human click
    df_good = df_kept.copy()
    df_good.drop(df_kept[df_kept['timeBetweenClicks'] >= 0].index, inplace = True) # keep only successful algorithm
    columns_for_dropna = ['block', 'searchTime', 'features_waitingTime', 'features_peakSpeed', 'features_accelerationTime', 'features_distance', 'humanClickTime', 'virtualClickTime', 'distanceToTargetWhenHumanClick', 'distanceToTargetWhenVirtualClick', 'timeBetweenClicks', 'perceivedFirstClickType']
    df_good = df_good.dropna(axis=0, subset=columns_for_dropna)

    # initializing variables
    startTime = list()
    endTime = list()
    searchTime = list()
    waitingTime = list()
    peakSpeed = list()
    accelerationTime = list()
    distance = list()
    humanClickTime = list()
    virtualClickTime = list()
    distanceToTargetWhenHumanClick = list()
    distanceToTargetWhenVirtualClick = list()
    predictionError = list()

    totalTrials = list()
    keptTrials = list()
    goodTrials = list()

    MH_H = list()
    MH_M = list()
    HM_H = list()
    HM_M = list()
    H__H = list()
    H__M = list()
    M__H = list()
    M__M = list()

    mean_time = list()
    mean_SoA = list()
    mean_time_H = list()
    mean_time_M = list()

    for block in np.arange(0, 7):
        
        # one of the seven block
        df_all_block = df[df['block']==block+1]
        df_kept_block = df_kept[df_kept['block']==block+1] 
        df_good_block = df_good[df_good['block']==block+1]
        
        # ******   variables that are measured block 0-6  *******
        
        # From all trials
        startTime.append(df_all_block['startTime'].iloc[0])
        endTime.append(df_all_block['endTime'].iloc[-1])
        totalTrials.append(len(df_all_block))
        keptTrials.append(len(df_kept_block))

        # From succesful trials
        searchTime.append(df_good_block['searchTime'].mean())
        waitingTime.append(df_good_block['features_waitingTime'].mean())
        peakSpeed.append(df_good_block['features_peakSpeed'].mean())
        accelerationTime.append(df_good_block['features_accelerationTime'].mean())
        distance.append(df_good_block['features_distance'].mean())
        humanClickTime.append(df_good_block['humanClickTime'].mean())
        distanceToTargetWhenHumanClick.append(df_good_block['distanceToTargetWhenHumanClick'].mean())

        # ******   variables that are measured block 1-6  *******
        
        if block>0:
            
            # From kept trials (number of trials in each category)
            MH_H.append(len(df_kept_block[(df_kept_block['firstClickType'] == 0) & (df_kept_block['secondClickType'] == 1) & (df_kept_block['perceivedFirstClickType'] == 1)]))
            MH_M.append(len(df_kept_block[(df_kept_block['firstClickType'] == 0) & (df_kept_block['secondClickType'] == 1) & (df_kept_block['perceivedFirstClickType'] == 0)]))
            HM_H.append(len(df_kept_block[(df_kept_block['firstClickType'] == 1) & (df_kept_block['secondClickType'] == 0) & (df_kept_block['perceivedFirstClickType'] == 1)]))
            HM_M.append(len(df_kept_block[(df_kept_block['firstClickType'] == 1) & (df_kept_block['secondClickType'] == 0) & (df_kept_block['perceivedFirstClickType'] == 0)]))
            H__H.append(len(df_kept_block[(df_kept_block['firstClickType'] == 1) & (df_kept_block['secondClickType'].isna()) & (df_kept_block['perceivedFirstClickType'] == 1)]))
            H__M.append(len(df_kept_block[(df_kept_block['firstClickType'] == 1) & (df_kept_block['secondClickType'].isna()) & (df_kept_block['perceivedFirstClickType'] == 0)]))
            M__H.append(len(df_kept_block[(df_kept_block['firstClickType'] == 0) & (df_kept_block['secondClickType'].isna()) & (df_kept_block['perceivedFirstClickType'] == 1)]))
            M__M.append(len(df_kept_block[(df_kept_block['firstClickType'] == 0) & (df_kept_block['secondClickType'].isna()) & (df_kept_block['perceivedFirstClickType'] == 0)]))

            # From succesful trials
            goodTrials.append(len(df_good_block))

            virtualClickTime.append(df_good_block['virtualClickTime'].mean())
            distanceToTargetWhenVirtualClick.append(df_good_block['distanceToTargetWhenVirtualClick'].mean())
            # timeBetweenClicks.append(df_good_block['timeBetweenClicks'].mean())
            # perceivedFirstClickType.append(df_good_block['perceivedFirstClickType'].mean())

            mean_time.append(df_good_block['timeBetweenClicks'].mean())
            mean_SoA.append(df_good_block['perceivedFirstClickType'].mean())
            mean_time_H.append(df_good_block[df_good_block['perceivedFirstClickType'] == 1]['timeBetweenClicks'].mean())
            mean_time_M.append(df_good_block[df_good_block['perceivedFirstClickType'] == 0]['timeBetweenClicks'].mean())

            predictionError.append(df_good_block['prediction_error'].mean())

    variables = {
        'startTime': startTime, 'endTime': endTime, 'searchTime': searchTime, 'waitingTime': waitingTime, 'peakSpeed': peakSpeed, 'accelerationTime': accelerationTime, 'distance': distance, \
        'humanClickTime': humanClickTime, 'virtualClickTime': virtualClickTime, 'distanceToTargetWhenHumanClick': distanceToTargetWhenHumanClick, 'distanceToTargetWhenVirtualClick': distanceToTargetWhenVirtualClick, \
        'mean_time': mean_time, 'mean_SoA': mean_SoA, 'mean_time_H': mean_time_H, 'mean_time_M': mean_time_M,     
        'predictionError': predictionError, \
        'totalTrials': totalTrials, 'keptTrials': keptTrials, 'goodTrials': goodTrials, 'MH_H': MH_H, 'MH_M': MH_M, 'HM_H': HM_H, 'HM_M': HM_M, 'H__H': H__H, 'H__M': H__M, 'M__H': M__H, 'M__M': M__M\
        }        

    return variables


def extract_one_line_participant_survey_data(ID):

    filepath = _mainDirectory + '/' + ID + '_survey_results.txt'  
    f = open(filepath, newline='', encoding="utf-8-sig")
    surveyData = json.load(f)

    # hand
    age = surveyData['initial']['Age']
    sex = surveyData['initial']['Sexe']
    try: hand = surveyData['initial']['Latéralité']
    except: hand = surveyData['initial']['Hand-dominance']

    # IPC Questionnaire
    ipcDimensions = {
        "internality": {"subset": [1, 4, 5, 9, 18, 19, 21, 23], "score": 0},
        "powerfulOthers": {"subset": [3, 8, 11, 13, 15, 17, 20, 22], "score": 0},
        "chance": {"subset": [2, 6, 7, 10, 12, 14, 16, 24], "score": 0}}
    for key in ipcDimensions.keys():
        for item in ipcDimensions[key]['subset']:
            answer =  surveyData['final']['Q' + str(item)]
            translatedAnswer = answer
            if (answer <= 3): translatedAnswer = answer - 4
            else: translatedAnswer = answer - 3
            ipcDimensions[key]['score'] += translatedAnswer
        ipcDimensions[key]['score'] += 24
    IPC = [ipcDimensions['internality']['score'], ipcDimensions['powerfulOthers']['score'], ipcDimensions['chance']['score']]

    variables = {
        'subject': [ID, age, sex, hand],
        'IPC': IPC,
        'globalSoA': [i['Q1'] for i in surveyData['breaks']], 
        'globalConfidence': [i['Q2'] for i in surveyData['breaks']], 
        }
    comments = {
        'freeComments': [surveyData['postTraining'], surveyData['freeComments'][0], surveyData['freeComments'][1]]
        }

    return variables, comments


# ************** MAIN CODE ***************

directory = os.fsencode(_mainDirectory)
doneIDs = []

columns = ['ID', 'age', 'sex', 'hand', 'I', 'PO', 'C', 
'GlobalSoA1', 'GSoA2', 'GSoA3', 'GSoA4', 'GSoA5', 'GSoA6',
'GlobalConfidence1', 'GC2', 'GC3', 'GC4', 'GC5', 'GC6',
'BlockStart0', 'BS1', 'BS2', 'BS3', 'BS4', 'BS5', 'BS6',
'BlockEnd0', 'BE1', 'BE2', 'BE3', 'BE4', 'BE5', 'BE6',
'searchTime0', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6',
'waitingTime0', 'WT1', 'WT2', 'WT3', 'WT4', 'WT5', 'WT6',
'peakSpeed0', 'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6',
'accelerationTime0', 'AT1', 'AT2', 'AT3', 'AT4', 'AT5', 'AT6',
'distance0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
'humanClickTime0', 'HCT1', 'HCT2', 'HCT3', 'HCT4', 'HCT5', 'HCT6',
'virtualClickTime1', 'VCT2', 'VCT3', 'VCT4', 'VCT5', 'VCT6',
'distanceToTargetWhenHumanClick0', 'DTTHC1', 'DTTHC2', 'DTTHC3', 'DTTHC4', 'DTTHC5', 'DTTHC6',
'distanceToTargetWhenVirtualClick1', 'DTTVC2', 'DTTVC3', 'DTTVC4', 'DTTVC5', 'DTTVC6',
'meanTime1', 'MT2', 'MT3', 'MT4', 'MT5', 'MT6',
'meanSoA1', 'SoA2', 'SoA3', 'SoA4', 'SoA5', 'SoA6',
'meanTimeH1', 'MTH2', 'MTH3', 'MTH4', 'MTH5', 'MTH6',
'meanTimeM1', 'MTM2', 'MTM3', 'MTM4', 'MTM5', 'MTM6',
'predictionError1', 'PE2', 'PE3', 'PE4', 'PE5', 'PE6', 
'TotalTrials0', 'TT1', 'TT2', 'TT3', 'TT4', 'TT5', 'TT6',
'KeptTrials0', 'KT1', 'KT2', 'KT3', 'KT4', 'KT5', 'KT6',
'GoodTrials1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6',
'MH_H1', 'MH_H2', 'MH_H3', 'MH_H4', 'MH_H5', 'MH_H6',
'MH_M1', 'MH_M2', 'MH_M3', 'MH_M4', 'MH_M5', 'MH_M6',
'HM_H1', 'HM_H2', 'HM_H3', 'HM_H4', 'HM_H5', 'HM_H6',
'HM_M1', 'HM_M2', 'HM_M3', 'HM_M4', 'HM_M5', 'HM_M6',
'H__H1', 'H__H2', 'H__H3', 'H__H4', 'H__H5', 'H__H6',
'H__M1', 'H__M2', 'H__M3', 'H__M4', 'H__M5', 'H__M6',
'M__H1', 'M__H2', 'M__H3', 'M__H4', 'M__H5', 'M__H6',
'M__M1', 'M__M2', 'M__M3', 'M__M4', 'M__M5', 'M__H6'
]

columnsSurvey = ['ID', 'postTraining', 'postExp1', 'postExp2']

df_all = pd.DataFrame(columns=columns)
df_all_survey = pd.DataFrame(columns=columnsSurvey)

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
    # if (ID=='25001438_25001438'): continue # outlier
    if ID not in doneIDs:

        # where am I
        print(ID)

        # reading the data
        doneIDs.append(ID)
        
        exp_variables = extract_one_line_participant_exp_data(ID)
        survey_variables, survey_comments = extract_one_line_participant_survey_data(ID)

        # exp data
        all_variables = {**survey_variables, **exp_variables}
        all_values = [value for values in all_variables.values() for value in values]
        series = pd.Series(all_values)
        df = pd.DataFrame(series.values.reshape(1, -1), columns=columns)     
        df_all = pd.concat([df_all, df])
        
        # free comments
        df_survey1 = pd.DataFrame({'ID': [ID]})
        all_values_survey = [value for values in survey_comments.values() for value in values]
        series_survey = pd.Series(all_values_survey)
        df_survey2 = pd.DataFrame(series_survey.values.reshape(1, -1), columns=['postTraining', 'postExp1', 'postExp2'])
        df_survey = pd.concat([df_survey1, df_survey2], axis=1)     
        df_all_survey = pd.concat([df_all_survey, df_survey])
        df_all_survey.set_index(np.arange(len(df_all_survey)))



df_all.set_index(np.arange(len(df_all)), inplace=True)
df_all.to_csv('all_participants_exp_data.csv')
df_all_survey.set_index(np.arange(len(df_all_survey)), inplace=True)
df_all_survey.to_csv('all_participants_survey_comments.csv')

print('Number of participants = ' + str(len(df_all)))
print('Age = ' + str(df_all['age'].mean()) + '+/-' + str(df_all['age'].std()))
print('Number of females = ', str(len(df_all[df_all['sex']=='F'])))
print('Number of left-handed = ', str(len(df_all[df_all['hand']=='G'])))
print('Number of ambidextrous = ', str(len(df_all[df_all['hand']=='A'])))

