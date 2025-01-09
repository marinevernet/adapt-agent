from re import S
import sre_compile
import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns
from scipy.stats import ttest_1samp
from scipy.stats import pearsonr
from scipy.stats import t
from scipy import stats
import math
# from Rosner_ESD import *


with open('1_export_group_data.py') as f:
    exec(f.read())

# ************* Definition of functions **************************************


def plot_the_violins(array_to_plot1, array_to_plot2, ylabel, subplot_line, subplot_column, subplot_nb, deletexlabel):

    # Prepare a Panda data frame
    dataViolin = pd.DataFrame({'block 1': array_to_plot1[:,0], 'block 2': array_to_plot1[:,1], 'block 3': array_to_plot1[:,2], 'block 4': array_to_plot1[:,3], 'block 5': array_to_plot1[:,4], 'block 6': array_to_plot1[:,5]})

    # Add patient ID to data
    dataViolin['subject_id'] = range(1, len(array_to_plot1)+1)

    # Reshape data for plotting
    data_melt = pd.melt(dataViolin, id_vars=['subject_id'], value_vars=['block 1', 'block 2', 'block 3', 'block 4', 'block 5', 'block 6'],
                        var_name='distribution', value_name='value')
    prettycolor = {'block 1': (0.945, 0.855, 0.835), 'block 2': (0.886, 0.706, 0.728), 'block 3': (0.792, 0.561, 0.659), 'block 4': (0.651, 0.427, 0.592), 'block 5':(0.459, 0.302, 0.486), 'block 6': (0.235, 0.165, 0.310)}
    
    # open the figure
    plt.subplot(subplot_line, subplot_column, subplot_nb)

    # Create violin plot
    sns.violinplot(x='distribution', y='value', data=data_melt, inner=None, cut=0, palette=prettycolor, ax=plt.gca())

    # Create strip plot
    sns.stripplot(x='distribution', y='value', data=data_melt, jitter=True, color='black', size=2, ax=plt.gca())

    plt.plot([0, 1, 2, 3, 4, 5], [np.mean(array_to_plot1[:,0]), np.mean(array_to_plot1[:,1]), np.mean(array_to_plot1[:,2]), np.mean(array_to_plot1[:,3]), np.mean(array_to_plot1[:,4]), np.mean(array_to_plot1[:,5])], color='black', alpha=1, linewidth=2)

    # Connect dots for repeated measures
    for pnb in range(1, len(array_to_plot1)+1): #range(1, n_patients+1):
        patient_data = data_melt[data_melt['subject_id'] == pnb]
        plt.plot([0, 1, 2, 3, 4, 5], patient_data['value'], color='gray', alpha=0.2, linewidth=0.5)

    # Add axis labels and title
    plt.ylabel(ylabel)
    plt.gca().set_xlabel("")  # This line removes the x-axis label
    if deletexlabel:
        plt.xticks([])

    # Add identification of the pannel
    if subplot_line>1 or subplot_column>1:
        # plt.text(-0.2, 1.1, chr(64 + subplot_nb), transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', ha='center')
        plt.text(-0.2, 1.1, chr(64 + subplot_nb), transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', ha='center')

    # Avoid overlap
    plt.tight_layout()


def testing_against_0(var1, title):

    if stats.shapiro(var1)[1]>0.05:
        a = stats.ttest_1samp(var1, 0) 
        print(title +  ': mean = %.2f +/- %.2f; (t-test) t(%d) = %.2f; p = %.4f; Cohen d=%.2f' % (np.array(var1).mean(), np.array(var1).std(), a.df, a.statistic, a.pvalue, np.array(var1).mean()/np.array(var1).std()))
    else: 
        a2 = pg.wilcoxon(var1)
        print(title +  ': mean = %.2f +/- %.2f; (pingouin Wilcoxon) t = %.2f; p = %.4f; effect size r(rb)=%.2f' % (np.array(var1).mean(), np.array(var1).std(), a2['W-val'].Wilcoxon, a2['p-val'].Wilcoxon, a2['RBC'].Wilcoxon))

def fitting_function(var1, var2, list1, list2):

    z1 = np.polyfit(var1, var2, 1)
    p1 = np.poly1d(z1)
    list1.append(z1[0])
    list2.append(np.mean(var2))

    return list1, list2


# ************* Main script **************************************

blocks = np.array([1, 2, 3, 4, 5, 6])

# ****** extract data of interest accross the 6 blocks ******
time = np.array(df_all[['meanTime1', 'MT2', 'MT3', 'MT4', 'MT5', 'MT6']]).astype(float)
SoA = np.array(df_all[['meanSoA1', 'SoA2', 'SoA3', 'SoA4', 'SoA5', 'SoA6']]).astype(float)
time_H = np.array(df_all[['meanTimeH1', 'MTH2', 'MTH3', 'MTH4', 'MTH5', 'MTH6']]).astype(float)
time_M = np.array(df_all[['meanTimeM1', 'MTM2', 'MTM3', 'MTM4', 'MTM5', 'MTM6']]).astype(float)

# ****** extract LOC values ******
I = np.array(df_all['I']).astype(float)
PO = np.array(df_all['PO']).astype(float)
C = np.array(df_all['C']).astype(float)

# ****** extract global SoA and confidence ******
global_SoA = np.array(df_all[['GlobalSoA1', 'GSoA2', 'GSoA3', 'GSoA4', 'GSoA5', 'GSoA6']]).astype(float)
global_confidence = np.array(df_all[['GlobalConfidence1', 'GC2', 'GC3', 'GC4', 'GC5', 'GC6']]).astype(float)

# extract participants' performance and feedback
searchTime = np.array(df_all[[ 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6']]).astype(float) # signficant decrease
waitingTime = np.array(df_all[[ 'WT1', 'WT2', 'WT3', 'WT4', 'WT5', 'WT6']]).astype(float) # no change
peakSpeed = np.array(df_all[[ 'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6']]).astype(float) # signficant increase
accelerationTime = np.array(df_all[['AT1', 'AT2', 'AT3', 'AT4', 'AT5', 'AT6']]).astype(float) # no change
humanClickTime = np.array(df_all[[ 'HCT1', 'HCT2', 'HCT3', 'HCT4', 'HCT5', 'HCT6']]).astype(float) # significant decrease
virtualClickTime = np.array(df_all[[ 'virtualClickTime1', 'VCT2', 'VCT3', 'VCT4', 'VCT5', 'VCT6']]).astype(float) # significant decrease
distanceAtStart = np.array(df_all[[ 'D1', 'D2', 'D3', 'D4', 'D5', 'D6']]).astype(float) # no change
distanceAtHumanClick = np.array(df_all[[ 'DTTHC1', 'DTTHC2', 'DTTHC3', 'DTTHC4', 'DTTHC5', 'DTTHC6']]).astype(float) # maybe (does not resist correct for MC)
distanceAtVirtualClick = np.array(df_all[[ 'distanceToTargetWhenVirtualClick1', 'DTTVC2', 'DTTVC3', 'DTTVC4', 'DTTVC5', 'DTTVC6']]).astype(float) # maybe (does not resist correct for MC)

# ****** initiate lists ******

time_fit=list()
time_mean=list()

SoA_mean = list()
SoA_fit = list()
SoA_fit_corrected = list()
global_SoA_mean = list()
global_SoA_fit = list()
global_SoA_fit_corrected = list()

searchTime_mean = list()
searchTime_fit = list()
waitingTime_mean = list()
waitingTime_fit = list()
peakSpeed_mean = list()
peakSpeed_fit = list()
accelerationTime_mean = list()
accelerationTime_fit = list()
humanClickTime_mean = list()
humanClickTime_fit = list()
virtualClickTime_mean = list()
virtualClickTime_fit = list()
distanceAtStart_mean = list()
distanceAtStart_fit = list()
distanceAtHumanClick_mean = list()
distanceAtHumanClick_fit = list()
distanceAtVirtualClick_mean = list()
distanceAtVirtualClick_fit = list()

global_confidence_mean = list()

corr_coeff_uncorrected = list()
corr_coeff_corrected1= list()
corr_coeff_corrected2 = list()
corr_coeff_global_uncorrected = list()
corr_coeff_global_corrected2 = list()
corr_coeff_searchTime_uncorrected = list()
corr_coeff_waitingTime_uncorrected = list()
corr_coeff_peakSpeed_uncorrected = list()
corr_coeff_accelerationTime_uncorrected = list()
corr_coeff_humanClickTime_uncorrected = list()
corr_coeff_virtualClickTime_uncorrected = list()
corr_coeff_distanceAtStart_uncorrected = list()
corr_coeff_distanceAtHumanClick_uncorrected = list()
corr_coeff_distanceAtVirtualClick_uncorrected = list()

residuals_time_SoA_list = list()
residuals_time_global_SoA_list = list()
residuals_block_time_list = list()

blocks_list = list()
subjects_list = list()

i = list()
po = list()
c = list()

corr_time_SoA = list()


# Set a global default font weight for labels
plt.rcParams['axes.labelweight'] = 'bold'
# Set a global default font weight for all text elements
plt.rcParams['font.weight'] = 'bold'

# ******* testing if correlation coefficents increase with times *****
# ****** fit trendline accross blocks ******


for subject in range(len(time)):

    blocks_list.append(blocks)
    subjects_list.append([subject, subject, subject, subject, subject, subject])

    # fit variable = f(block)
    time_fit, time_mean = fitting_function(blocks[:], time[subject, :],  time_fit, time_mean)
    SoA_fit, SoA_mean = fitting_function(blocks[:], SoA[subject, :], SoA_fit, SoA_mean)
    global_SoA_fit, global_SoA_mean = fitting_function(blocks[:], global_SoA[subject, :], global_SoA_fit, global_SoA_mean)
    searchTime_fit, searchTime_mean = fitting_function(blocks[:], searchTime[subject, :], searchTime_fit, searchTime_mean)
    waitingTime_fit, waitingTime_mean = fitting_function(blocks[:], waitingTime[subject, :], waitingTime_fit, waitingTime_mean)
    peakSpeed_fit, peakSpeed_mean = fitting_function(blocks[:], peakSpeed[subject, :], peakSpeed_fit, peakSpeed_mean)
    accelerationTime_fit, accelerationTime_mean = fitting_function(blocks[:], accelerationTime[subject, :], accelerationTime_fit, accelerationTime_mean)
    humanClickTime_fit, humanClickTime_mean = fitting_function(blocks[:], humanClickTime[subject, :], humanClickTime_fit, humanClickTime_mean)
    virtualClickTime_fit, virtualClickTime_mean = fitting_function(blocks[:], virtualClickTime[subject, :], virtualClickTime_fit, virtualClickTime_mean)
    distanceAtStart_fit, distanceAtStart_mean = fitting_function(blocks[:], distanceAtStart[subject, :], distanceAtStart_fit, distanceAtStart_mean)
    distanceAtHumanClick_fit, distanceAtHumanClick_mean = fitting_function(blocks[:], distanceAtHumanClick[subject, :], distanceAtHumanClick_fit, distanceAtHumanClick_mean)
    distanceAtVirtualClick_fit, distanceAtVirtualClick_mean = fitting_function(blocks[:], distanceAtVirtualClick[subject, :], distanceAtVirtualClick_fit, distanceAtVirtualClick_mean)

    # correlations with blocks
    corr_coeff_uncorrected.append(pearsonr(blocks[:], SoA[subject, :])[0])
    corr_coeff_global_uncorrected.append(pearsonr(blocks[:], global_SoA[subject, :])[0])
    corr_coeff_searchTime_uncorrected.append(pearsonr(blocks[:], searchTime[subject, :])[0])
    corr_coeff_waitingTime_uncorrected.append(pearsonr(blocks[:], waitingTime[subject, :])[0])
    corr_coeff_peakSpeed_uncorrected.append(pearsonr(blocks[:], peakSpeed[subject, :])[0])
    corr_coeff_accelerationTime_uncorrected.append(pearsonr(blocks[:], accelerationTime[subject, :])[0])
    corr_coeff_humanClickTime_uncorrected.append(pearsonr(blocks[:], humanClickTime[subject, :])[0])
    corr_coeff_virtualClickTime_uncorrected.append(pearsonr(blocks[:], virtualClickTime[subject, :])[0])
    corr_coeff_distanceAtStart_uncorrected.append(pearsonr(blocks[:], distanceAtStart[subject, :])[0])
    corr_coeff_distanceAtHumanClick_uncorrected.append(pearsonr(blocks[:], distanceAtHumanClick[subject, :])[0])
    corr_coeff_distanceAtVirtualClick_uncorrected.append(pearsonr(blocks[:], distanceAtVirtualClick[subject, :])[0])

    # global confidence
    global_confidence_mean.append(np.mean(global_confidence[subject, :]))


    # ************* partial correlation *******************
    # ******** partial correlation (SoA = f(Block, controlling for time algo)**************

    # 'partial correlation (with pinguin)'
    df = pd.DataFrame(np.transpose([blocks[:], SoA[subject, :], time[subject, :]]))
    statsres = pg.partial_corr(data=df, x=0, y=1, covar=2)
    print(str(subject)+ ' ' + str(statsres['r'].iloc[0]))
    corr_coeff_corrected1.append(statsres['r'].iloc[0])

    # 'partial correlation (my calculation with residuals)'
    z1 = np.polyfit(time[subject, :], SoA[subject, :], 1)
    residuals_time_SoA = SoA[subject, :]-(z1[0]*time[subject, :] + z1[1])
    residuals_time_SoA_list.append(residuals_time_SoA)

    z1 = np.polyfit(time[subject, :], global_SoA[subject, :], 1)
    residuals_time_global_SoA = global_SoA[subject, :]-(z1[0]*time[subject, :] + z1[1])
    residuals_time_global_SoA_list.append(residuals_time_global_SoA)

    z1 = np.polyfit(time[subject, :], blocks[:], 1)
    residuals_block_time = blocks[:]-(z1[0]*time[subject, :] + z1[1])
    residuals_block_time_list.append(residuals_block_time)

    a = pearsonr(residuals_block_time, residuals_time_SoA)
    corr_coeff_corrected2.append(a[0])
    a = pearsonr(residuals_block_time, residuals_time_global_SoA)
    corr_coeff_global_corrected2.append(a[0])

    # 'fit residual SoA = f(residual block), same with global_SoA'
    z1 = np.polyfit(residuals_block_time, residuals_time_SoA, 1)
    SoA_fit_corrected.append(z1[0])
    z1 = np.polyfit(residuals_block_time, residuals_time_global_SoA, 1)
    global_SoA_fit_corrected.append(z1[0])

    # correlation between time and SoA
    a = pearsonr(time[subject, :], SoA[subject, :])
    corr_time_SoA.append(a[0])
   
    # ************* IPC *******************
    i.append(I[subject])
    po.append(PO[subject])
    c.append(C[subject])



# ****** Figure 1: Local SoA increases  ******

fig1 = plt.figure()
plot_the_violins(SoA, None, 'local SoA', 1, 1, 1, False)
plt.show()


# ****** Figure 2 both SoA and tested time increased accross blocks (and are correlated) ******

blocks_array = np.concatenate(blocks_list, axis=0)
subjects_array = np.concatenate(subjects_list, axis=0)
SoA_array = np.concatenate(SoA, axis=0)
global_SoA_array = np.concatenate(global_SoA, axis=0)
time_array = np.concatenate(time, axis=0)

d = {'block': blocks_array, 'local SoA': SoA_array, 'Time Between Clicks': time_array}
df = pd.DataFrame(data=d)
sns.set_theme(style="ticks")
g = sns.pairplot(df, hue="block", corner=True)
# Rotate the last diagonal KDE plot by 90 degrees
g.axes[-1, -1].set_xlim(g.axes[-1, -1].get_xlim()[::-1])
g.axes[1, 0].set_ylim(g.axes[-1, -1].get_xlim()[::-1])

# Add titles to each panel
x_margin = 0.05  # Adjust the horizontal margin
y_margin = 0.05  # Adjust the vertical margin
g.axes[0, 0].text(x_margin, 1 - y_margin, 'A', transform=g.axes[0, 0].transAxes,
                  fontsize=12, va='top', ha='left', fontweight='bold')
g.axes[1, 0].text(x_margin, 1 - y_margin, 'B', transform=g.axes[1, 0].transAxes,
                  fontsize=12, va='top', ha='left', fontweight='bold')
g.axes[1, 1].text(x_margin, 1 - y_margin, 'C', transform=g.axes[1, 1].transAxes,
                  fontsize=12, va='top', ha='left', fontweight='bold')

sns.regplot(x=df['local SoA'], y=df['Time Between Clicks'], color='grey', scatter=False, ax=g.axes[1][0])

plt.show()

corr_time_SoA = np.array(corr_time_SoA)
print('\nTesting difference against 0:')
testing_against_0(corr_time_SoA, 'Correlation Time Between Clicks/SoA (R)')

# ****** testing main hypothesis  ******

testing_against_0(SoA_fit, 'Mean SoA Fit')
testing_against_0(SoA_fit_corrected, 'Mean SoA Fit corrected')
testing_against_0(corr_coeff_uncorrected, 'corr coeff')
testing_against_0(corr_coeff_corrected2, 'corr coeff corrected')
testing_against_0(global_SoA_fit, 'Global SoA Fit')
testing_against_0(global_SoA_fit_corrected, 'Global SoA Fit corrected')


# ****** another way to look at the corrected correlation  ******

residuals_block_time_array = np.concatenate(residuals_block_time_list, axis=0)
residuals_time_SoA_array = np.concatenate(residuals_time_SoA_list, axis=0)
a = pearsonr(residuals_block_time_array, residuals_time_SoA_array)
print('\nCorrelation of the residuals (one big correlation all participants together):') 
print('R = %.2f; p = %.4f; CI = [%.2f, %.2f]' % (a[0], a.pvalue, a.confidence_interval(0.95)[0], a.confidence_interval(0.95)[1]))


fig7 = plt.figure()
for subject in range(len(time)):
    # plt.plot(df[df['Participants']==subject]['residuals_blocks'], df[df['Participants']==subject]['residuals_SoA'], 'o')
    plt.plot(residuals_block_time_list[subject], residuals_time_SoA_list[subject], 'o')
    # sns.regplot(x=residuals_block_time_list[subject], y=residuals_time_SoA_list[subject], color='grey', scatter=False, ci=None)
# z1 = np.polyfit(residuals_block_time_array, residuals_time_SoA_array, 1)
# plt.plot(residuals_block_time_array, z1[0]*residuals_block_time_array+z1[1], 'k')
sns.regplot(x=residuals_block_time_array, y=residuals_time_SoA_array, color='k', scatter=False)
plt.xlabel('corrected Block')
plt.ylabel('corrected local SoA')
plt.show()


# ******** correlating partial r with LOC questionnaire ******

# ****** coherence of LOC questionnaire  ******

print('\nTesting correlation LOC:')
print('no correlation i and po : R= %.2f; p = %.4f; CI = [%.2f, %.2f]' % (pearsonr(i, po).statistic, pearsonr(i, po).pvalue, pearsonr(i, po).confidence_interval(0.95)[0], pearsonr(i, po).confidence_interval(0.95)[1]))
print('no correlation i and c : R= %.2f; p = %.4f; CI = [%.2f, %.2f]' % (pearsonr(i, c).statistic, pearsonr(i, c).pvalue, pearsonr(i, c).confidence_interval(0.95)[0], pearsonr(i, c).confidence_interval(0.95)[1]))
print('correlation po and c : R= %.2f; p = %.4f; CI = [%.2f, %.2f]' % (pearsonr(po, c).statistic, pearsonr(po, c).pvalue, pearsonr(po, c).confidence_interval(0.95)[0], pearsonr(po, c).confidence_interval(0.95)[1]))

d = {'I': i, 'PO': po, 'C': c}
df = pd.DataFrame(data=d)
fig10 = plt.figure()
ax10 = fig10.add_subplot(projection='3d')
ax10.scatter(i, po, c)
ax10.set_xlabel('I')
ax10.set_ylabel('PO')
ax10.set_zlabel('C')
ax10.plot(I, C, 'r.', zdir='y', zs=50)
ax10.plot(PO, C, 'g.', zdir='x', zs=-5)
ax10.plot(I, PO, 'k.', zdir='z', zs=-10)
z = np.polyfit(PO, C, 1)
p = np.poly1d(z)
ax10.plot(PO, p(PO), 'g--', zdir='x', zs=-5)

plt.show()

# ****** Agency - LOC  ******

print('\nTesting correlation Agency-LOC:')

def correlation_agency_loc(varname, var1, var2):
    rescor = pearsonr(var1, var2)
    if rescor.pvalue <= 0.05: print('%s : R=%.2f; p = %.4f; CI=[%.2f, %.2f]' % (varname, rescor.statistic, rescor.pvalue, rescor.confidence_interval(0.95)[0], rescor.confidence_interval(0.95)[1]))

print('\ncorrelation with mean SOA:')
correlation_agency_loc('I', i, SoA_mean)
correlation_agency_loc('PO', po, SoA_mean)
correlation_agency_loc('C', c, SoA_mean)
correlation_agency_loc('I/total', np.divide(np.array(i), (np.array(i)+np.array(po)+np.array(c))), SoA_mean)
correlation_agency_loc('PO/total', np.divide(np.array(i), (np.array(po)+np.array(po)+np.array(c))), SoA_mean)
correlation_agency_loc('C/total', np.divide(np.array(i), (np.array(c)+np.array(po)+np.array(c))), SoA_mean)

print('\ncorrelation with mean global SOA:')
correlation_agency_loc('I', i, global_SoA_mean)
correlation_agency_loc('PO', po, global_SoA_mean)
correlation_agency_loc('C', c, global_SoA_mean)
correlation_agency_loc('I/total', np.divide(np.array(i), (np.array(i)+np.array(po)+np.array(c))), global_SoA_mean)
correlation_agency_loc('PO/total', np.divide(np.array(i), (np.array(po)+np.array(po)+np.array(c))), global_SoA_mean)
correlation_agency_loc('C/total', np.divide(np.array(i), (np.array(c)+np.array(po)+np.array(c))), global_SoA_mean)

print('\ncorrelation with mean SoA_fit_corrected:')
correlation_agency_loc('I', i, SoA_fit_corrected)
correlation_agency_loc('PO', po, SoA_fit_corrected)
correlation_agency_loc('C', c, SoA_fit_corrected)
correlation_agency_loc('I/total', np.divide(np.array(i), (np.array(i)+np.array(po)+np.array(c))), SoA_fit_corrected)
correlation_agency_loc('PO/total', np.divide(np.array(i), (np.array(po)+np.array(po)+np.array(c))), SoA_fit_corrected)
correlation_agency_loc('C/total', np.divide(np.array(i), (np.array(c)+np.array(po)+np.array(c))), SoA_fit_corrected)

print('\ncorrelation with global_SoA_fit_corrected:')
correlation_agency_loc('I', i, global_SoA_fit_corrected)
correlation_agency_loc('PO', po, global_SoA_fit_corrected)
correlation_agency_loc('C', c, global_SoA_fit_corrected)
correlation_agency_loc('I/total', np.divide(np.array(i), (np.array(i)+np.array(po)+np.array(c))), global_SoA_fit_corrected)
correlation_agency_loc('PO/total', np.divide(np.array(i), (np.array(po)+np.array(po)+np.array(c))), global_SoA_fit_corrected)
correlation_agency_loc('C/total', np.divide(np.array(i), (np.array(c)+np.array(po)+np.array(c))), global_SoA_fit_corrected)

fig, ax = plt.subplots(2, 2)

sns.regplot(x=SoA_mean, y=np.divide(np.array(i), (np.array(i)+np.array(po)+np.array(c))), ax=ax[0,0])
ax[0,0].set_xlabel('SoA mean')
ax[0,0].set_ylabel('I/(I+PO+C)')

sns.regplot(x=global_SoA_mean, y=po, ax=ax[0,1])
ax[0,1].set_xlabel('global SoA mean')
ax[0,1].set_ylabel('PO')

sns.regplot(x=global_SoA_fit_corrected, y=po, ax=ax[1,0])
ax[1,0].set_xlabel('global SoA mean')
ax[1,0].set_ylabel('PO')

sns.regplot(x=global_SoA_fit_corrected, y=np.divide(np.array(po), (np.array(i)+np.array(po)+np.array(c))), ax=ax[1,1])
ax[1,1].set_xlabel('global SoA mean')
ax[1,1].set_ylabel('PO/(I+PO+C)')

plt.show()


# ****** SoA (local and global , mean and fit)  ******

print('\nTesting correlations Local/Global SoA:')

d = {
     'mean local SoA': SoA_mean, 'mean global SoA': global_SoA_mean, \
     'corrected local SoA shift': SoA_fit_corrected, 'corrected global SoA shift': global_SoA_fit_corrected,
     }
df = pd.DataFrame(data=d)
g = sns.pairplot(df, diag_kind='kde', corner=True)

# Add trendlines to the scatter panels
for ax in g.axes.flat:
    try:
        a = pearsonr(df[ax.get_xlabel()], df[ax.get_ylabel()])
        if a[1]<=0.05:
            sns.regplot(data=df, x=ax.get_xlabel(), y=ax.get_ylabel(), ax=ax, scatter=False)
            print('correlation between %s and %s: R = %.2f; p = %.4f; CI = [%.2f, %.2f]' % (ax.get_xlabel(), ax.get_ylabel(), a[0], a[1], a.confidence_interval(0.95)[0], a.confidence_interval(0.95)[1]))
    except:
        pass

# Add 0 to diagonal distribution of fit
for axnb, ax in enumerate(g.axes):
    if axnb>1:
        ax[axnb].axvline(0, linestyle='--')

# Add titles to each panel
panel_titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
nb = 0
for ii in np.arange(4):
    for j in np.arange(4):
        if ii>=j:
            ax = g.axes[ii][j]
            x_margin = 0.05  # Adjust the horizontal margin
            y_margin = 0.05  # Adjust the vertical margin
            ax.text(x_margin, 1 - y_margin, panel_titles[nb], transform=ax.transAxes,
                    fontsize=12, va='top', ha='left', fontweight='bold')
            nb=nb+1
plt.show()



# ****** explort results  ******


df_all['time_mean'] = time_mean
df_all['time_fit'] = time_fit

df_all['SoA_mean'] = SoA_mean
df_all['SoA_fit'] = SoA_fit
df_all['SoA_fit_corrected'] = SoA_fit_corrected
df_all['corr_coeff_uncorrected'] = corr_coeff_uncorrected
df_all['corr_coeff_corrected'] = corr_coeff_corrected2
df_all['corr_coeff_corrected1'] = corr_coeff_corrected1

df_all['global_SoA_mean'] = global_SoA_mean
df_all['global_SoA_fit'] = global_SoA_fit
df_all['global_SoA_fit_corrected'] = global_SoA_fit_corrected
df_all['corr_coeff_global_uncorrected'] = corr_coeff_global_uncorrected
df_all['corr_coeff_global_corrected'] = corr_coeff_global_corrected2

df_all['global_confidence_mean'] = global_confidence_mean

df_all['residuals_time_SoA_list'] = residuals_time_SoA_list

df_all['residuals_time_global_SoA_list'] = residuals_time_global_SoA_list
df_all['residuals_block_time_list'] = residuals_block_time_list
df_all['residuals_block_time_list'] = residuals_block_time_list
df_all['i'] = i
df_all['po'] = po
df_all['c'] = c

df_all.to_csv('all_participants_results_calculation.csv')


# ***** checking the visual searchPerformance *****

fig100 = plt.figure(figsize=(9, 6))
plot_the_violins(searchTime, None, 'Search Time', 2, 2, 1, True)
plot_the_violins(peakSpeed, None, 'Peak Speed', 2, 2, 2, True)
plot_the_violins(humanClickTime, None, 'Human Click Time', 2, 2, 3, True)
plot_the_violins(virtualClickTime, None, 'Virtual Click Time', 2, 2, 4, True)

plt.show()


print('\nTesting difference against 0:')
testing_against_0(searchTime_fit, 'SearchTime fit')
testing_against_0(corr_coeff_searchTime_uncorrected, 'SearchTime corr corref')
testing_against_0(waitingTime_fit,'WaitingTime fit')
testing_against_0(corr_coeff_waitingTime_uncorrected, 'WaitingTime corr corref')
testing_against_0(peakSpeed_fit, 'PeakSpeed fit')
testing_against_0(corr_coeff_peakSpeed_uncorrected, 'PeakSpeed corr corref')
testing_against_0(accelerationTime_fit, 'AccelerationTime fit')
testing_against_0(corr_coeff_accelerationTime_uncorrected, 'AccelerationTime corr corref')
testing_against_0(humanClickTime_fit, 'HumanClickTime fit')
testing_against_0(corr_coeff_humanClickTime_uncorrected, 'HumanClickTime corr corref')
testing_against_0(virtualClickTime_fit, 'virtualClickTime fit')
testing_against_0(corr_coeff_virtualClickTime_uncorrected, 'virtualClickTime corr corref')
testing_against_0(distanceAtStart_fit, 'DistanceAtStart fit')
testing_against_0(corr_coeff_distanceAtStart_uncorrected, 'DistanceAtStart corr corref')
testing_against_0(distanceAtHumanClick_fit, 'distanceAtHumanClick fit')
testing_against_0(corr_coeff_distanceAtHumanClick_uncorrected, 'distanceAtHumanClick corr corref')
testing_against_0(distanceAtVirtualClick_fit, 'distanceAtVirtualClick fit')
testing_against_0(corr_coeff_distanceAtVirtualClick_uncorrected, 'distanceAtVirtualClick corr corref')
