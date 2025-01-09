
import os
import pandas as pd
import numpy as np
import json

_mainDirectory = "sessions"


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

    ipcAllQuestions = {f'Q{i}': None for i in range(1, 24+1)}
    for question in np.arange(1, 25):
        answer = surveyData['final']['Q' + str(question)]
        translatedAnswer = answer
        if (answer <= 3): translatedAnswer = answer - 4
        else: translatedAnswer = answer - 3
        ipcAllQuestions[f'Q{question}'] = translatedAnswer
    ipcAllQuestionsValues = [ipcAllQuestions[f'Q{i}'] for i in range(1, 25)]

    variables = {
        'subject': [ID, age, sex, hand],
        'IPC': IPC,
        'ipcAllQuestions': ipcAllQuestionsValues
        }

    return variables


# ************** MAIN CODE ***************

directory = os.fsencode(_mainDirectory)
doneIDs = []

columns = ['ID', 'age', 'sex', 'hand', 'I', 'PO', 'C', 
           'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
           'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20',
           'Q21', 'Q22', 'Q23', 'Q24']

df_all = pd.DataFrame(columns=columns)

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
        
        survey_variables = extract_one_line_participant_survey_data(ID)

        # exp data
        all_variables = {**survey_variables}
        all_values = [value for values in all_variables.values() for value in values]
        series = pd.Series(all_values)
        df = pd.DataFrame(series.values.reshape(1, -1), columns=columns)     
        df_all = pd.concat([df_all, df])
        


df_all.set_index(np.arange(len(df_all)), inplace=True)
df_all.to_csv('all_participants_exp_data_details_IPC.csv')


