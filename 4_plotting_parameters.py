import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ***************  CHOICE OF PARAMETERS AND INITIALIZE VARIABLES ***************

_mainDirectory = "sessions"
_ID = ""

# choice_to_plot = 1 # plot for a single participant
# choice_to_plot = 2 # plot for all participants (be careful, very long!)
choice_to_plot = 3 # single plot including data from all participants

# _pred_colums2 = ['block', 'searchTime', 'peakSpeed', 'accelerationTime', 'distance', 'waitingTime', 'humanClickTime', 'virtualClickTime', 'distanceToTargetWhenHumanClick', 'distanceToTargetWhenVirtualClick', 'timeBetweenClicks']
_pred_colums2 = ['Block', 'Search Time', 'Peak Speed', 'Distance \nat Start', 'Distance \nat Human Click', 'Distance \nat Virtual Click', 'Time Between Clicks']
_choice_of_labels = 'local SoA'

# Set a global default font weight for labels
plt.rcParams['axes.labelweight'] = 'bold'
# Set a global default font weight and size for all text elements
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 14

# ******************* FUNCTIONS ***************

# Function to load the data of one participant
# --------------------------------------------
def reading_data(filepath):

    df=pd.read_csv(filepath, sep=';')

    df.drop(df[df['block'] <2].index, inplace = True) # remove the training trials and training block
    df.drop(df[df['block'] >7].index, inplace = True) # remove the extra-blocks, if any
    df = df[df['popupType'].apply(lambda x: isinstance(x, float))] # remove the trials with warnings
    df.drop(df[df['timeBetweenClicks'] >0].index, inplace = True) # keep only successful algorithm
    # columns = ['block', 'searchTime', 'features_waitingTime', 'features_peakSpeed', 'features_accelerationTime', 'features_distance', 'humanClickTime', 'virtualClickTime', 'distanceToTargetWhenHumanClick', 'distanceToTargetWhenVirtualClick', 'timeBetweenClicks', 'prediction_error', 'perceivedFirstClickType']
    columns = ['block', 'searchTime', 'features_peakSpeed', 'features_accelerationTime', 'features_distance', 'features_waitingTime', 'humanClickTime', 'virtualClickTime', 'distanceToTargetWhenHumanClick', 'distanceToTargetWhenVirtualClick', 'timeBetweenClicks', 'perceivedFirstClickType']
    df = df[columns]
    df = df.rename(columns={'features_peakSpeed': 'Peak Speed', 'features_accelerationTime': 'Acceleration Time', 'features_distance': 'Distance \nat Start', 'features_waitingTime': 'Waiting Time'})
    df = df.rename(columns={'block': 'Block', 'searchTime': 'Search Time', 'distanceToTargetWhenHumanClick': 'Distance \nat Human Click', 'distanceToTargetWhenVirtualClick': 'Distance \nat Virtual Click', 'timeBetweenClicks': 'Time Between Clicks', 'perceivedFirstClickType': 'local SoA'})
    df = df.dropna()

    return df


# ************** MAIN CODE ***************

# plot and save the sns.pairplot for a single participant
if choice_to_plot == 1:

    # reading the data
    if (len(_ID) == 0): ID = "147589736_147589736"
    else: ID = _ID
    filepath = _mainDirectory + '/' + ID + '_experiment_results.txt'
    
    df=reading_data(filepath)
    df = df[_pred_colums2 + [_choice_of_labels]]

    sns.pairplot(df, hue="local SoA", corner=True)
    plt.savefig('parameters_' + ID +'.png')
    # plt.show()


# plot and save all sns.pairplot
if choice_to_plot == 2:

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
            filepath = _mainDirectory + '/' + ID + '_experiment_results.txt'
            
            df=reading_data(filepath)
            df = df[_pred_colums2 + [_choice_of_labels]]
            sns.pairplot(df, hue="local SoA", corner=True)
            plt.savefig('parameters_' + ID +'.png')
        else:
            continue

# plot and save a single sns.pairplot for all participant
if choice_to_plot == 3:

    df_all = pd.DataFrame(columns=_pred_colums2 + [_choice_of_labels])

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
            filepath = _mainDirectory + '/' + ID + '_experiment_results.txt'
            
            df=reading_data(filepath)
            df = df[_pred_colums2 + [_choice_of_labels]]
            df_all = pd.concat([df_all, df])

        else:
            continue

    df_all['local SoA'] = df_all['local SoA'].astype(int)
    g = sns.pairplot(df_all, hue="local SoA", corner=True)

    # Move the legend to a specific position and size
    legend = g._legend
    legend.set_bbox_to_anchor((0.8, 0.6))
    legend.set_title(title="local SoA", prop={"size": 20})  # Change the font size (adjust as needed)
    for label in legend.get_texts():
        label.set_fontsize(20)

    plt.savefig('parameters_only_significant_all_players.png')
    plt.show()

