import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
from scipy.stats import ttest_ind
from scipy import stats

with open('1_export_group_data.py') as f:
    exec(f.read())


# Set a global default font weight for labels
plt.rcParams['axes.labelweight'] = 'bold'
# Set a global default font weight for all text elements
plt.rcParams['font.weight'] = 'bold'

# ********** CHECKING THE POST TRAINING COMMENTS **************
# Did the participants noticed that the game clicked before they did?

group_something_weird = [
    '227985121_227985121',
    '31394413_31394413',
    '134321459_134321459',
    '25171274_25171274',
    '21574206_21574206',
    '206810302_206810302',
    '76618630_76618630',
    '241085945_241085945']

group_nothing = [
    '23767256_23767256',
    '136189181_136189181',
    '4689213_4689213',
    '170791133_170791133',
    '48303334_48303334',
    '136295119_225465923',
    '48303334_48303334',
    '136295119_225465923',
    '242297519_242297519',
    '179968006_179968006',
    '177624241_177624241']

group_both = group_nothing + group_something_weird

print('\n')
print('Post Training Comments')
print('\n')
print('Participants who did not notice anything:' + str(len(group_nothing)))
print(df_all_survey[df_all_survey['ID'].isin(group_nothing)]['postTraining'])
print('\n')
print('Participants who notices something weird:' + str(len(group_something_weird)))
print(df_all_survey[df_all_survey['ID'].isin(group_something_weird)]['postTraining'])
print('\n')
print('Participants noticed the odd one was selected before they clicked:' + str(len(df_all_survey)-len(group_nothing)-len(group_something_weird)))
print(df_all_survey[~df_all_survey['ID'].isin(group_both)]['postTraining'])


# ********** CHECKING ALL COMMENTS **************
# Did the participants referring to warnings or having a bad experice with the game have a different SoA?

words_to_search = [
    'annoying',
    'stress',
    'corrections',
    # 'challenging',
    'difficile',
    'difficult',
    'difficulties',
    'erreurs',
    'error',
    # 'extensive',
    'frustrating',
    'frustration',
    # 'attention',
    'hard',
    'laggy',
    'précision',
    'missed',
    'mistakes',
    'painful',
    'pas toujours sûr',
    'penalties',
    'punishing',
    'really tricky',
    'reminder messages',
    'stressfull',
    'not easy',
    'tired',
    'tiring',
    'chiant',
    'warning',
    'warnings']


# Check if any of the words are present in any of the three columns
indices = []
for index, row in df_all_survey.iterrows():
    # if any(word in str(row[column]) for word in words_to_search for column in ['postTraining', 'postExp1', 'postExp2']):
    if any(word in value.lower() for word in words_to_search for value in row[['postTraining', 'postExp1', 'postExp2']]):
        indices.append(index)

participants_warnings_negative = df_all_survey.loc[indices].copy()['ID']
participants_rest_of = df_all_survey.drop(indices).copy()['ID']

# you can re-run 1_group_analysis or just open the csv file created by that script
df_all_calculation = pd.read_csv('all_participants_results_calculation.csv')
    
# add a column with the group classification
df_all_calculation['groups'] = np.where(np.isin(df_all_calculation['ID'], np.array(participants_warnings_negative)), 'group 1', 'group 2')
df_warnings_negative = df_all_calculation[df_all_calculation['groups']=='group 1']
df_rest_of = df_all_calculation[df_all_calculation['groups']=='group 2']

# How many participant per group
print('\n# of participants')
print('mentionning warnings or a negative experience: %d' % len(df_warnings_negative))
print('other: %d' % len(df_rest_of))

# See exemples of sentences
print('\nExamples')
print('\nmentionning warnings or a negative experience:')
print(df_all_survey.loc[indices].copy())
print('\nother:')
print(df_all_survey.drop(indices).copy())

# check if the group are different
print('\n Significant difference between the two groups?')

def check_group_different(title, var1, var2): # similar result with kruskal instead of mannwhitneyu
    if stats.shapiro(var1[title])[1]>0.05 and stats.shapiro(var2[title])[1]>0.05:
        a = stats.ttest_ind(var1[title], var2[title], equal_var=False) #, alternative='greater')
        cohen_s = ((len(var1[title])-1)*np.std(var1[title])**2+(len(var2[title])-1)*np.std(var2[title])**2)/(len(var1[title])+len(var2[title])-2)
        cohen_d = np.abs(np.mean(var1[title])-np.mean(var2[title]))/cohen_s
        print('%s (t-test): t=%.2f; p=%.4f; effect size d=%.2f' %(title, a.statistic, a.pvalue, cohen_d))

    else: 
        # The manual calculation of the effect size for Mann-Whitney U is correct. Same results with scipy.stats and pingouin.
        # a = stats.mannwhitneyu(var1[title], var2[title])
        # print('%s (scipy.stats Mann-Whitney): U=%.2f; p=%.4f; effect size r(rb)=%.2f' %(title, a.statistic, a.pvalue, 1-(2*a.statistic)/(len(var1[title])*len(var2[title]))))
        a2 = pg.mwu(var1[title], var2[title])
        print('%s (pingouin Mann-Whitney): U=%.2f; p=%.4f; effect size r(rb)=%.2f' %(title, a2['U-val'].MWU, a2['p-val'].MWU, a2['RBC'].MWU))

def check_group_different_bis(title, var1, var2): # similar result with kruskal instead of mannwhitneyu
    if stats.shapiro(var1)[1]>0.05 and stats.shapiro(var2)[1]>0.05:
        a = stats.ttest_ind(var1, var2, equal_var=False) #, alternative='greater')
        cohen_s = ((len(var1)-1)*np.std(var1)**2+(len(var2)-1)*np.std(var2)**2)/(len(var1)+len(var2)-2)
        cohen_d = np.abs(np.mean(var1)-np.mean(var2))/cohen_s
        print('%s (t-test): t(%d)=%.2f; p=%.4f; effect size d=%.2f' %(title, a.df, a.statistic, a.pvalue, cohen_d))
    else: 
        # The manual calculation of the effect size for Mann-Whitney U is correct. Same results with scipy.stats and pingouin.
        # a = stats.mannwhitneyu(var1, var2)
        # print('%s (scipy.stats Mann-Whitney): U=%.2f; p=%.4f; effect size r(rb)=%.2f' %(title, a.statistic, a.pvalue, 1-(2*a.statistic)/(len(var1)*len(var2))))
        a2 = pg.mwu(var1, var2)
        print('%s (pingouin Mann-Whitney): U=%.2f; p=%.4f; effect size r(rb)=%.2f' %(title, a2['U-val'].MWU, a2['p-val'].MWU, a2['RBC'].MWU))


check_group_different('time_mean', df_warnings_negative, df_rest_of)
check_group_different('time_fit', df_warnings_negative, df_rest_of)
check_group_different_bis(' # of warnings', df_warnings_negative[['TT1', 'TT2', 'TT3', 'TT4', 'TT5', 'TT6']].sum(axis=1)-6*80, df_rest_of[['TT1', 'TT2', 'TT3', 'TT4', 'TT5', 'TT6']].sum(axis=1)-6*80)
check_group_different('SoA_mean', df_warnings_negative, df_rest_of)
check_group_different('SoA_fit_corrected', df_warnings_negative, df_rest_of)
check_group_different('global_SoA_mean', df_warnings_negative, df_rest_of)
check_group_different('global_SoA_fit_corrected', df_warnings_negative, df_rest_of)

print('\nmeans')

def print_mean(title, var1, var2):
    print('%s: warning/nevative group: %.2f +/- %.2f; Other group: %.2f +/- %.2f' % (title, var1[title].mean(),  var1[title].std(), var2[title].mean(),  var2[title].std()))

print_mean('time_mean', df_warnings_negative, df_rest_of)
print_mean('global_SoA_mean', df_warnings_negative, df_rest_of)
print_mean('global_SoA_fit_corrected', df_warnings_negative, df_rest_of)

print('\nSignificant decrease (warning/negative group) or increase (other) of global SoA?')

def testing_shift(title, group, var1):
    if stats.shapiro(var1[title])[1]>0.05:
        a = stats.ttest_1samp(var1[title], 0) #, alternative='greater')
        print('%s, %s, t-test: t(%d)=%.2f; p=%.4f; Cohen d=%.2f' % (title, group, a.df, a.pvalue, a.pvalue, np.abs(np.array(var1[title]).mean())/np.array(var1[title]).std()))
    else: 
        # The manual calculation of the effect size for Wilcoxon was not correct (because it doesn't take into account the directionality: use pingouin instead)
        # a = stats.wilcoxon(var1[title])
        # print('%s, %s, scipy.stats Wilcoxon: t=%.2f; p=%.4f; effect size r(rb)=%.2f' % (title, group, a.statistic, a.pvalue, (1-(2*a.statistic/(len(var1[title])*(len(var1[title])+1))))))
        a2 = pg.wilcoxon((var1[title]))
        print('%s, %s, pingouin    Wilcoxon: t=%.2f; p=%.4f; effect size r(rb)=%.2f' % (title, group, a2['W-val'].Wilcoxon, a2['p-val'].Wilcoxon, a2['RBC'].Wilcoxon))

testing_shift('global_SoA_fit_corrected', 'warning/negative', df_warnings_negative)
testing_shift('global_SoA_fit_corrected', 'other', df_rest_of)



# ************ FOCUS ON GLOBAL SOA FOR PLOTTING ****************

# prepare the new dataframes
df_all_reduced = df_all_calculation[['ID', 'groups', 'GlobalSoA1', 'GSoA2', 'GSoA3', 'GSoA4', 'GSoA5', 'GSoA6']]
df_all_reduced = df_all_reduced.rename(columns={"GlobalSoA1": "block1", "GSoA2": "block2", "GSoA3": "block3", "GSoA4": "block4", "GSoA5": "block5", "GSoA6": "block6"})

id_vars = ['ID', 'groups']
value_vars = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6']
var_name = 'Block number'
value_name = 'global SoA'

melted_df = pd.melt(df_all_reduced, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
melted_df = melted_df.sort_values(by=['ID', 'Block number'])

melted_df_warnings_negative = melted_df[melted_df['groups']=='group 1']
melted_df_rest_of = melted_df[melted_df['groups']=='group 2']

# Draw a nested violinplot and split the violins for easier comparison
ax = sns.violinplot(data=melted_df, x="Block number", y="global SoA", hue="groups", hue_order=['group 1', 'group 2'],
                    split=True, linewidth=1, cut=0)

plt.plot([0, 1, 2, 3, 4, 5], [melted_df_warnings_negative[melted_df_warnings_negative['Block number']=='block1']['global SoA'].mean(), 
                              melted_df_warnings_negative[melted_df_warnings_negative['Block number']=='block2']['global SoA'].mean(), 
                              melted_df_warnings_negative[melted_df_warnings_negative['Block number']=='block3']['global SoA'].mean(), 
                              melted_df_warnings_negative[melted_df_warnings_negative['Block number']=='block4']['global SoA'].mean(), 
                              melted_df_warnings_negative[melted_df_warnings_negative['Block number']=='block5']['global SoA'].mean(), 
                              melted_df_warnings_negative[melted_df_warnings_negative['Block number']=='block6']['global SoA'].mean()],
          linewidth=2)

plt.plot([0, 1, 2, 3, 4, 5], [melted_df_rest_of[melted_df_rest_of['Block number']=='block1']['global SoA'].mean(), 
                              melted_df_rest_of[melted_df_rest_of['Block number']=='block2']['global SoA'].mean(), 
                              melted_df_rest_of[melted_df_rest_of['Block number']=='block3']['global SoA'].mean(), 
                              melted_df_rest_of[melted_df_rest_of['Block number']=='block4']['global SoA'].mean(), 
                              melted_df_rest_of[melted_df_rest_of['Block number']=='block5']['global SoA'].mean(), 
                              melted_df_rest_of[melted_df_rest_of['Block number']=='block6']['global SoA'].mean()],
          linewidth=2)


# Adjust the figure size to accommodate the label
fig = plt.gcf()
fig.set_size_inches(8, 6)  # Adjust the size as per your requirements

# Move the legend outside the figure
ax.legend(bbox_to_anchor=(1, 1), loc='upper center')

# Show the plot
plt.show()



