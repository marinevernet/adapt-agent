import os
import numpy as np
import pandas as pd
from itertools import combinations
import statsmodels.api as sm
from tqdm import tqdm


# *********  CHOICE OF PARAMETERS ***************

_mainDirectory = "sessions"
_pred_colums2 = ['block', 'searchTime', 'peakSpeed', 'accelerationTime', 'distance', 'waitingTime', 'humanClickTime', 'virtualClickTime', 'distanceToTargetWhenHumanClick', 'distanceToTargetWhenVirtualClick', 'timeBetweenClicks']
_choice_of_labels = 'perceivedFirstClickType'


# *************** FUNCTIONS *********************

# Reading the data and selecting/renaming columns
# -----------------------------------------------
def reading_data(filepath):

    df=pd.read_csv(filepath, sep=';')

    df.drop(df[df['block'] <2].index, inplace = True) # remove the training trials and training block
    df.drop(df[df['block'] >7].index, inplace = True) # remove the extra-blocks, if any
    df = df[df['popupType'].apply(lambda x: isinstance(x, float))] # remove the trials with warnings
    df.drop(df[df['timeBetweenClicks'] >0].index, inplace = True) # keep only successful algorithm
    # columns = ['block', 'searchTime', 'features_waitingTime', 'features_peakSpeed', 'features_accelerationTime', 'features_distance', 'humanClickTime', 'virtualClickTime', 'distanceToTargetWhenHumanClick', 'distanceToTargetWhenVirtualClick', 'timeBetweenClicks', 'prediction_error', 'perceivedFirstClickType']
    columns = ['block', 'searchTime', 'features_peakSpeed', 'features_accelerationTime', 'features_distance', 'features_waitingTime', 'humanClickTime', 'virtualClickTime', 'distanceToTargetWhenHumanClick', 'distanceToTargetWhenVirtualClick', 'timeBetweenClicks', 'perceivedFirstClickType']
    df = df[columns]
    df = df.rename(columns={'features_peakSpeed': 'peakSpeed', 'features_accelerationTime': 'accelerationTime', 'features_distance': 'distance', 'features_waitingTime': 'waitingTime'})
    df = df.dropna()

    return df


# Fitting model (after normalizing the predictors) to predict the SoA
# -------------------------------------------------------------------
def fitting_one_model(df, predictors):
    
    y = df[_choice_of_labels]
    x = df[predictors]
    x = (x - x.mean())/(x.std())
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    
    return model


# Fitting all possible combinations of models, adding the participants as random variables
# ----------------------------------------------------------------------------------------
def fitting_combination_models2(df, preds, preds_participants):

    model_ind_aic = list()
    list_of_combo = list()

    
    for r in tqdm(np.arange(1, len(preds)+1)):
        for combo in combinations(preds, r):
            
            # for all possible combinations of predictors
            if len(combo) == 1: 
                combo= [combo[0]]
            else:
                combo = np.array(combo)
            
            # fit the model once
            model = fitting_one_model(df, combo)
            model_ind_aic.append(model.aic)
            list_of_combo.append(combo)

            # add all participants as random predictors
            combo=list(combo)
            for part in preds_participants:
                combo.append(part)
  
            # fit the model again
            model = fitting_one_model(df, combo)
            model_ind_aic.append(model.aic)
            list_of_combo.append(combo)

    # find the best model based on the lowest AIC
    best_model_idx = np.argmin(np.array(model_ind_aic))

    # list of predictors for the best model, best model, weights
    best_model_predictors = list_of_combo[best_model_idx]
    best_model = fitting_one_model(df, best_model_predictors)
    best_model_weights = best_model.params

    return model_ind_aic, best_model, best_model_predictors, best_model_weights, list_of_combo


# ************** MAIN *****************************
# find the best model for the group of participants

params_group = list()

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
    if (ID=='45996241_45996241'): continue # remove participant who answered only twice that it was not har
    
    if ID not in doneIDs:

        # where am I
        print(ID)
        
        filepath = _mainDirectory + '/' + ID + '_experiment_results.txt'  
        df=reading_data(filepath)

        df['participant']=ID

        if any(doneIDs): df_all = pd.concat([df_all, df])
        else: df_all = df

        doneIDs.append(ID)

    else:
        continue

# Convert the participants' label column into as many columns as there are participants
df_all = pd.get_dummies(df_all, drop_first=True, columns=['participant'], prefix=['category'])

# Prepare the pd dataframe with the participants' columns
y = df_all.drop(columns=_choice_of_labels)
yy = y.drop(columns=_pred_colums2)

# Find the best model
model_ind_aic, best_model, best_model_predictors, best_model_weights, list_of_combo = fitting_combination_models2(df_all, _pred_colums2, yy.columns)

# best_model.params[0:8] =
# const                               0.746191
# block                               0.008635
# searchTime                          0.006164
# peakSpeed                           0.017118
# distance                           -0.014030
# distanceToTargetWhenHumanClick     -0.011393
# distanceToTargetWhenVirtualClick   -0.030139
# timeBetweenClicks                   0.196092

# best_model.pvalues[0:8]
# const                               0.000000e+00
# block                               5.322713e-07
# searchTime                          2.098147e-03
# peakSpeed                           2.511515e-18
# distance                            3.479137e-10
# distanceToTargetWhenHumanClick      1.814511e-07
# distanceToTargetWhenVirtualClick    1.047848e-33
# timeBetweenClicks                   0.000000e+00

# best_model_predictors =
# ['block',
#  'searchTime',
#  'peakSpeed',
#  'distance',
#  'distanceToTargetWhenHumanClick',
#  'distanceToTargetWhenVirtualClick',
#  'timeBetweenClicks',
#  'category_101286468_101286468',
#  'category_102330358_102330358',
#  'category_10502553_10502553',
#  'category_106282497_106282497',
#  'category_114166405_114166405',
#  'category_116269512_116269512',
#  'category_121162388_121162388',
#  'category_122052773_122052773',
#  'category_12303458_12303458',
#  'category_127366482_127366482',
#  'category_129771717_129771717',
#  'category_134321459_134321459',
#  'category_135326258_135326258',
#  'category_136189181_136189181',
#  'category_136295119_225465923',
#  'category_138057247_138057247',
#  'category_147119413_147119413',
#  'category_147589736_147589736',
#  'category_149412238_149412238',
#  'category_150404285_150404285',
#  'category_15634040_15634040',
#  'category_156391236_156391236',
#  'category_161563474_161563474',
#  'category_163537234_163537234',
#  'category_163618449_163618449',
#  'category_164532367_164532367',
#  'category_166465762_166465762',
#  'category_169128813_169128813',
#  'category_169318478_169318478',
#  'category_169472058_169472058',
#  'category_170791133_170791133',
#  'category_171525381_171525381',
#  'category_17361736_17361736',
#  'category_173753237_173753237',
#  'category_174770575_174770575',
#  'category_177624241_177624241',
#  'category_178637920_178637920',
#  'category_178699978_178699978',
#  'category_179968006_179968006',
#  'category_181534843_25665390',
#  'category_182327542_182327542',
#  'category_186122125_186122125',
#  'category_187175550_187175550',
#  'category_187562196_187562196',
#  'category_197577548_197577548',
#  'category_199249500_199249500',
#  'category_19970485_103116652',
#  'category_201482909_201482909',
#  'category_202178857_202178857',
#  'category_20496846_20496846',
#  'category_205336752_205336752',
#  'category_206810302_206810302',
#  'category_207567088_207567088',
#  'category_208421181_208421181',
#  'category_212221613_212221613',
#  'category_21574206_21574206',
#  'category_21961422_21961422',
#  'category_222324918_222324918',
#  'category_22676373_22676373',
#  'category_227985121_227985121',
#  'category_228307330_228307330',
#  'category_233054812_233054812',
#  'category_233361278_233361278',
#  'category_234195761_234195761',
#  'category_23729824_23729824',
#  'category_23767256_23767256',
#  'category_238092845_238092845',
#  'category_241085945_241085945',
#  'category_242297519_242297519',
#  'category_243329838_243329838',
#  'category_245112698_245112698',
#  'category_245138573_245138573',
#  'category_247632331_247632331',
#  'category_248339442_248339442',
#  'category_249688569_249688569',
#  'category_25001438_25001438',
#  'category_250814597_187832916',
#  'category_25171274_25171274',
#  'category_25509906_25509906',
#  'category_260866741_260866741',
#  'category_265669543_265669543',
#  'category_267993085_267993085',
#  'category_28618912_28618912',
#  'category_31394413_31394413',
#  'category_32559076_32559076',
#  'category_40319355_40319355',
#  'category_43150525_43150525',
#  'category_46879303_46879303',
#  'category_4689213_4689213',
#  'category_48303334_48303334',
#  'category_5801872_5801872',
#  'category_58563529_58563529',
#  'category_65435647_65435647',
#  'category_66022916_66022916',
#  'category_66206712_66206712',
#  'category_68643580_68643580',
#  'category_69785882_69785882',
#  'category_7188230_7188230',
#  'category_72010381_72010381',
#  'category_76618630_76618630',
#  'category_77101454_77101454',
#  'category_79197636_79197636',
#  'category_8387634_8387634',
#  'category_85281525_85281525',
#  'category_85476979_85476979',
#  'category_91960078_91960078',
#  'category_92189776_92189776',
#  'category_98299546_98299546']




# In [11]: best_model.summary()
# Out[11]: 
# <class 'statsmodels.iolib.summary.Summary'>
# """
#                                OLS Regression Results                              
# ===================================================================================
# Dep. Variable:     perceivedFirstClickType   R-squared:                       0.334
# Model:                                 OLS   Adj. R-squared:                  0.332
# Method:                      Least Squares   F-statistic:                     195.5
# Date:                     Mon, 20 Nov 2023   Prob (F-statistic):               0.00
# Time:                             13:43:13   Log-Likelihood:                -17290.
# No. Observations:                    45026   AIC:                         3.481e+04
# Df Residuals:                        44910   BIC:                         3.582e+04
# Df Model:                              115                                         
# Covariance Type:                 nonrobust                                         
# ====================================================================================================
#                                        coef    std err          t      P>|t|      [0.025      0.975]
# ----------------------------------------------------------------------------------------------------
# const                                0.7462      0.002    445.133      0.000       0.743       0.749
# block                                0.0086      0.002      5.015      0.000       0.005       0.012
# searchTime                           0.0062      0.002      3.076      0.002       0.002       0.010
# peakSpeed                            0.0171      0.002      8.735      0.000       0.013       0.021
# distance                            -0.0140      0.002     -6.277      0.000      -0.018      -0.010
# distanceToTargetWhenHumanClick      -0.0114      0.002     -5.218      0.000      -0.016      -0.007
# distanceToTargetWhenVirtualClick    -0.0301      0.002    -12.111      0.000      -0.035      -0.025
# timeBetweenClicks                    0.1961      0.002     93.390      0.000       0.192       0.200
# category_101286468_101286468        -0.0124      0.002     -5.229      0.000      -0.017      -0.008
# category_102330358_102330358        -0.0227      0.002     -9.574      0.000      -0.027      -0.018
# category_10502553_10502553          -0.0463      0.002    -21.433      0.000      -0.051      -0.042
# category_106282497_106282497        -0.0124      0.002     -5.302      0.000      -0.017      -0.008
# category_114166405_114166405         0.0016      0.002      0.644      0.520      -0.003       0.006
# category_116269512_116269512        -0.0381      0.002    -16.419      0.000      -0.043      -0.034
# category_121162388_121162388         0.0042      0.002      1.797      0.072      -0.000       0.009
# category_122052773_122052773        -0.0149      0.002     -6.234      0.000      -0.020      -0.010
# category_12303458_12303458          -0.0024      0.002     -1.020      0.308      -0.007       0.002
# category_127366482_127366482         0.0074      0.002      3.149      0.002       0.003       0.012
# category_129771717_129771717        -0.0082      0.002     -3.410      0.001      -0.013      -0.003
# category_134321459_134321459        -0.0099      0.002     -4.167      0.000      -0.015      -0.005
# category_135326258_135326258        -0.0506      0.002    -21.584      0.000      -0.055      -0.046
# category_136189181_136189181        -0.0008      0.002     -0.378      0.705      -0.005       0.003
# category_136295119_225465923         0.0040      0.002      1.822      0.068      -0.000       0.008
# category_138057247_138057247         0.0024      0.002      1.052      0.293      -0.002       0.007
# category_147119413_147119413        -0.0038      0.002     -1.585      0.113      -0.009       0.001
# category_147589736_147589736        -0.0165      0.002     -6.944      0.000      -0.021      -0.012
# category_149412238_149412238        -0.0159      0.002     -6.832      0.000      -0.020      -0.011
# category_150404285_150404285         0.0086      0.002      3.798      0.000       0.004       0.013
# category_15634040_15634040           0.0062      0.002      2.648      0.008       0.002       0.011
# category_156391236_156391236        -0.0029      0.002     -1.355      0.175      -0.007       0.001
# category_161563474_161563474         0.0065      0.002      2.963      0.003       0.002       0.011
# category_163537234_163537234        -0.0074      0.002     -3.148      0.002      -0.012      -0.003
# category_163618449_163618449         0.0002      0.002      0.076      0.939      -0.004       0.005
# category_164532367_164532367        -0.0102      0.002     -4.295      0.000      -0.015      -0.006
# category_166465762_166465762         0.0006      0.002      0.246      0.805      -0.004       0.005
# category_169128813_169128813        -0.0054      0.002     -2.264      0.024      -0.010      -0.001
# category_169318478_169318478        -0.0200      0.002     -8.457      0.000      -0.025      -0.015
# category_169472058_169472058        -0.0068      0.002     -2.903      0.004      -0.011      -0.002
# category_170791133_170791133        -0.0044      0.002     -1.844      0.065      -0.009       0.000
# category_171525381_171525381        -0.0147      0.002     -6.338      0.000      -0.019      -0.010
# category_17361736_17361736          -0.0145      0.002     -6.197      0.000      -0.019      -0.010
# category_173753237_173753237        -0.0181      0.002     -7.673      0.000      -0.023      -0.013
# category_174770575_174770575        -0.0036      0.002     -1.576      0.115      -0.008       0.001
# category_177624241_177624241         0.0006      0.002      0.283      0.777      -0.004       0.005
# category_178637920_178637920         0.0053      0.002      2.304      0.021       0.001       0.010
# category_178699978_178699978        -0.0115      0.002     -4.904      0.000      -0.016      -0.007
# category_179968006_179968006         0.0081      0.002      3.848      0.000       0.004       0.012
# category_181534843_25665390         -0.0026      0.002     -1.153      0.249      -0.007       0.002
# category_182327542_182327542        -0.0117      0.002     -4.927      0.000      -0.016      -0.007
# category_186122125_186122125        -0.0173      0.002     -7.600      0.000      -0.022      -0.013
# category_187175550_187175550         0.0059      0.002      2.602      0.009       0.001       0.010
# category_187562196_187562196         0.0042      0.002      1.795      0.073      -0.000       0.009
# category_197577548_197577548        -0.0165      0.002     -6.937      0.000      -0.021      -0.012
# category_199249500_199249500        -0.0115      0.002     -5.712      0.000      -0.015      -0.008
# category_19970485_103116652         -0.0006      0.002     -0.245      0.807      -0.005       0.004
# category_201482909_201482909         0.0020      0.002      0.872      0.383      -0.003       0.007
# category_202178857_202178857        -0.0104      0.002     -4.455      0.000      -0.015      -0.006
# category_20496846_20496846          -0.0165      0.002     -6.983      0.000      -0.021      -0.012
# category_205336752_205336752        -0.0079      0.002     -3.384      0.001      -0.012      -0.003
# category_206810302_206810302        -0.0053      0.002     -2.433      0.015      -0.010      -0.001
# category_207567088_207567088        -0.0050      0.002     -2.142      0.032      -0.010      -0.000
# category_208421181_208421181         0.0042      0.002      1.768      0.077      -0.000       0.009
# category_212221613_212221613         0.0071      0.002      3.085      0.002       0.003       0.012
# category_21574206_21574206          -0.0195      0.002     -8.349      0.000      -0.024      -0.015
# category_21961422_21961422          -0.0397      0.002    -17.887      0.000      -0.044      -0.035
# category_222324918_222324918         0.0033      0.002      1.443      0.149      -0.001       0.008
# category_22676373_22676373          -0.0450      0.002    -19.201      0.000      -0.050      -0.040
# category_227985121_227985121        -0.0171      0.002     -7.710      0.000      -0.021      -0.013
# category_228307330_228307330        -0.0320      0.002    -13.756      0.000      -0.037      -0.027
# category_233054812_233054812         0.0069      0.002      2.919      0.004       0.002       0.012
# category_233361278_233361278        -0.0102      0.002     -4.579      0.000      -0.015      -0.006
# category_234195761_234195761         0.0056      0.002      2.431      0.015       0.001       0.010
# category_23729824_23729824          -0.0139      0.002     -5.924      0.000      -0.018      -0.009
# category_23767256_23767256          -0.0039      0.002     -1.746      0.081      -0.008       0.000
# category_238092845_238092845        -0.0166      0.002     -7.267      0.000      -0.021      -0.012
# category_241085945_241085945        -0.0048      0.002     -2.035      0.042      -0.010      -0.000
# category_242297519_242297519         0.0009      0.002      0.391      0.696      -0.004       0.005
# category_243329838_243329838        -0.0113      0.002     -4.905      0.000      -0.016      -0.007
# category_245112698_245112698        -0.0275      0.002    -11.719      0.000      -0.032      -0.023
# category_245138573_245138573        -0.0027      0.002     -1.127      0.260      -0.007       0.002
# category_247632331_247632331        -0.0099      0.002     -4.222      0.000      -0.015      -0.005
# category_248339442_248339442        -0.0128      0.002     -5.426      0.000      -0.017      -0.008
# category_249688569_249688569        -0.0004      0.002     -0.160      0.873      -0.005       0.004
# category_25001438_25001438          -0.0083      0.002     -3.601      0.000      -0.013      -0.004
# category_250814597_187832916        -0.0064      0.002     -2.702      0.007      -0.011      -0.002
# category_25171274_25171274           0.0035      0.002      1.429      0.153      -0.001       0.008
# category_25509906_25509906           0.0009      0.002      0.380      0.704      -0.004       0.005
# category_260866741_260866741         0.0118      0.002      5.001      0.000       0.007       0.016
# category_265669543_265669543        -0.0263      0.002    -11.592      0.000      -0.031      -0.022
# category_267993085_267993085        -0.0070      0.002     -3.021      0.003      -0.011      -0.002
# category_28618912_28618912          -0.0165      0.002     -6.798      0.000      -0.021      -0.012
# category_31394413_31394413          -0.0059      0.002     -2.472      0.013      -0.010      -0.001
# category_32559076_32559076           0.0109      0.002      4.565      0.000       0.006       0.016
# category_40319355_40319355          -0.0090      0.002     -3.784      0.000      -0.014      -0.004
# category_43150525_43150525           0.0081      0.002      3.401      0.001       0.003       0.013
# category_46879303_46879303          -0.0338      0.002    -14.212      0.000      -0.038      -0.029
# category_4689213_4689213            -0.0035      0.002     -1.501      0.133      -0.008       0.001
# category_48303334_48303334          -0.0316      0.002    -13.249      0.000      -0.036      -0.027
# category_5801872_5801872            -0.0283      0.002    -11.876      0.000      -0.033      -0.024
# category_58563529_58563529          -0.0099      0.002     -4.238      0.000      -0.015      -0.005
# category_65435647_65435647           0.0023      0.002      0.985      0.325      -0.002       0.007
# category_66022916_66022916          -0.0052      0.002     -2.286      0.022      -0.010      -0.001
# category_66206712_66206712          -0.0579      0.002    -24.572      0.000      -0.062      -0.053
# category_68643580_68643580           0.0137      0.002      5.884      0.000       0.009       0.018
# category_69785882_69785882          -0.0205      0.002     -8.964      0.000      -0.025      -0.016
# category_7188230_7188230            -0.0166      0.002     -7.029      0.000      -0.021      -0.012
# category_72010381_72010381           0.0115      0.002      5.053      0.000       0.007       0.016
# category_76618630_76618630          -0.0128      0.002     -5.496      0.000      -0.017      -0.008
# category_77101454_77101454           0.0051      0.002      2.158      0.031       0.000       0.010
# category_79197636_79197636           0.0125      0.002      5.284      0.000       0.008       0.017
# category_8387634_8387634             0.0084      0.002      3.558      0.000       0.004       0.013
# category_85281525_85281525          -0.0148      0.002     -6.372      0.000      -0.019      -0.010
# category_85476979_85476979           0.0042      0.002      1.792      0.073      -0.000       0.009
# category_91960078_91960078           0.0118      0.002      5.007      0.000       0.007       0.016
# category_92189776_92189776          -0.0216      0.002     -9.531      0.000      -0.026      -0.017
# category_98299546_98299546           0.0086      0.002      3.721      0.000       0.004       0.013
# ==============================================================================
# Omnibus:                     3242.845   Durbin-Watson:                   1.826
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3984.945
# Skew:                          -0.720   Prob(JB):                         0.00
# Kurtosis:                       3.227   Cond. No.                         15.3
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# """

