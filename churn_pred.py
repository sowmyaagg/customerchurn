import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb
import pprint
import numpy as np
from collections import defaultdict
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance


def read_data_from_dir(filepath,filename):
    currWorkDir =  os.getcwd()
    print(os.listdir(filepath))
    # change directory
    os.chdir(filepath)
    # Read file
    dataDF = pd.read_csv(filename)
    # change to original working directory
    os.chdir(currWorkDir)
    return dataDF

def split_onefeature_frmDf(df, outcome_colname):
    outcomeVarDF = df.loc[:, outcome_colname]
    df           = df.drop(outcome_colname, axis=1)
    return outcomeVarDF, df

def unique_vals_dataframe(DF):
    # to see what's in each column
    for variable in list(DF):
        if not isinstance(DF[variable],(int,float)):
            print(variable)
            print(DF[variable].unique())


def convert_col_numeric(df_mixed,list_col):
    # df_mixed['tenure'] = pd.to_numeric(df_mixed['tenure'],errors='coerce')
    df_mixed[list_col] \
        = df_mixed[list_col].apply(pd.to_numeric, errors='coerce')
    # #check output
    # print('Check columns, '+ repr(list_col) + ', conversion to numeric datatype:\n')
    # print(df_mixed.info())
    return df_mixed


def get_categorical_vars(featuresDF):
    num_categoricalvars = 0
    categoricalDict     = defaultdict(list)
    for variable in list(featuresDF):
        if (featuresDF[variable].dtype != 'int64') and (featuresDF[variable].dtype != 'float64'):
            num_categoricalvars += 1
            distinct_vals     = list(featuresDF[variable].unique())
            distinct_vals_num = eval('[distinct_vals, len(distinct_vals)]')
            categoricalDict.update({variable: distinct_vals_num})

    categoricalDF = pd.DataFrame(categoricalDict)
    pprint.pprint(categoricalDict)
    # pprint.pprint(categoricalDF)
    return categoricalDict, categoricalDF


def encode_categorical_dataframe(featDF,categoricalDF):
    # featDF - dataframe that have one or more columns with categorical variables
    # categorical dict has key-name corresponding to column-name in dataDF
    # categorical dict value has distinct values that need encoding

    #0 we should also check if categoricalDF is a pandas Dataframe

    #1 check if we have only two values in categorical column
    val_criteria_2 = categoricalDF.iloc[1] == 2
    list_vars_2    = list(val_criteria_2.index[val_criteria_2==1])
    print('list_vars_2 = ', repr(list_vars_2))
    categ_lookup_encoding = defaultdict(lambda:defaultdict(int))
    for var in list_vars_2:
        categ_lookup_encoding.update({var : {categoricalDF.loc[0,var][0]:0,categoricalDF.loc[0,var][1]:1}})
        #pd.to_numeric(featDF[var].replace(categoricalDF.loc[0,var],[0,1],inplace=False),errors='coerce')
        #featDF[var] = pd.to_numeric(featDF[var], errors='coerce')
    featDF.replace(categ_lookup_encoding,inplace=True)
    print('categ_lookup_encoding:\n')
    pprint.pprint(categ_lookup_encoding)

    #2 for others use get dummies, if no variables have this then skip
    val_criteria_mult = categoricalDF.iloc[1] > 2
    list_multiple_vals = list(val_criteria_mult.index[val_criteria_mult==1])
    dummies = pd.get_dummies(featDF,columns = list_multiple_vals,prefix = list_multiple_vals)
    featDF  = featDF.drop(list_multiple_vals,axis=1) # drop multiple columns
    featDF  = pd.concat([featDF,dummies],axis=1)

    featDF  = featDF.select_dtypes(include='number')
    featDF = featDF.loc[:, ~featDF.columns.duplicated()]
    return featDF, categ_lookup_encoding



# script starts here
filepath    = '/Users/Sowmya/Documents/Data_for_Practice/Logistic_Regression/Telcom_Customer_Churn/'
filename    = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
dataDF      = read_data_from_dir(filepath,filename)

# browse data
# print(dataDF.info())
# https://stats.stackexchange.com/questions/201962/classification-algorithm-for-categorical-data

# split primary key from features and outcome set in dataframe
prime_key            = 'customerID'
prime_keyDF, dataDF  = split_onefeature_frmDf(dataDF, prime_key)

# # browse datatypes in dataDF to catch fix inaccurate data-types
# unique_vals_dataframe(dataDF)

# fix datatype of the dataDF's columns from string-object to numeric datatype
list_col  = ['tenure','TotalCharges']
dataDF    = convert_col_numeric(dataDF,list_col)
dataDF    = dataDF.replace('',np.nan)             # also convert empty strings to NaN
dataDF    = dataDF.dropna(axis=0,how='any')       #drop rows with any NaN values

# Create a look-up table for categorical variables
categoricalDict,categoricalDF = get_categorical_vars(dataDF)

# encode categorical variables in features and outcome columns of dataDF
originalDF= dataDF.copy()
dataDF, categ_encode_lookup = encode_categorical_dataframe(dataDF,categoricalDF)
# pprint.pprint(dataDF.iloc[0,:])

# Split data dataframe to outcome variable's dataframe and original data dataframe with only features columns
outcome_colname            = 'Churn'
outcomeVarDF, fdataDF       = split_onefeature_frmDf(dataDF, outcome_colname)
outcomeCatVarDF, catDataDF  = split_onefeature_frmDf(originalDF, outcome_colname)




'''totalChurn      = outcomeVarDF.sum(axis=1)
totalNoChurn    = outcomeVarDF.shape[0] - outcomeVarDF.sum(axis=1)
totalChurnArray = np.array([totalChurn, totalNoChurn ])'''

# summaries https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

print('sum of Tenure per churn' + repr(originalDF.groupby('Churn')['tenure'].sum()))
print('sum of churn per gender ' + repr(dataDF.groupby('SeniorCitizen')['Churn'].sum()))

# why am I not able to work with dataDF

# plot relationships


# Box plot
#sb.catplot(x='gender',y='TotalCharges',hue = 'Churn', data=originalDF) #equal churns in gender



#sb.catplot(x='SeniorCitizen',y='tenure',hue='Churn',kind= 'box',data=originalDF) #the strip, default shows that seniors are more churned

#sb.catplot(x='Churn',y='tenure',kind='box',data = originalDF) # we can see tenure is a significant predictor

'''sb.catplot(x='gender',y='tenure',hue='Churn',kind = 'box',data=originalDF) #there is no difference
sb.catplot(x='SeniorCitizen',y='tenure',hue='Churn',kind = 'box',data=originalDF) #there is some difference
sb.catplot(x='StreamingTV',y='tenure',hue='Churn',kind = 'box',data=originalDF) #StreamingTV customers have higher tenure, some have churn some not
sb.catplot(x='DeviceProtection',y='tenure',hue='Churn',kind = 'box',data=originalDF) #StreamingTV customers have higher tenure. Only StreamingTV customer with lower tenures have churn
sb.catplot(x='DeviceProtection',y='MonthlyCharges',hue='Churn',kind = 'box',data=originalDF) #StreamingTV customers have higher tenure. Only StreamingTV customer with lower tenures have churn
sb.catplot(x='Churn',y='MonthlyCharges',kind = 'box',data=originalDF) #Monthly Charges have some influence on churn, higher monthly charge higher churn
sb.catplot(x='Churn',y='TotalCharges',kind = 'box',data=originalDF) #Total Charges have some influence on churn, Lower Total Charges have positive affect on churn'''
#sb.scatterplot(x='tenure',y='MonthlyCharges',hue = 'Churn',data=originalDF ) # the distribution seems classifiable by linear method

'''
f, ax = plt.subplots(figsize=(10, 8))
corr = dataDF.corr()
sb.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sb.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

plt.show()
'''

#https://github.com/SSaishruthi/LogisticRegression_Vectorized_Implementation/blob/master/Logistic_Regression.ipynb

'''scaler       = StandardScaler()
outcomeVarDF = scaler.fit_transform(outcomeVarDF)'''

# Train Logistic Regression model with all features

X_train, X_test, y_train, y_test = train_test_split(fdataDF,outcomeVarDF,test_size=0.2, random_state=42)


# train with RandomForestClassifier
clf = RandomForestClassifier(random_state=0).fit(X_train,y_train)
print('RF Model fitted with'+ repr(40) + 'features\n RF feature importances\n')
important_features = pd.Series(data=clf.feature_importances_,index=fdataDF.columns)
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(fdataDF.columns, clf.feature_importances_):
    feats[feature] = importance #add the name/value pair

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)
plt.rcParams.update({'font.size': 5})
plt.show()
print('RF decision path')
pprint.pprint(clf.decision_path(X_train))
y_pred = clf.predict(X_test)
print('RF Score')
print(clf.score(X_test,y_test))
print('RMSE\n')
print(np.sqrt(metrics.mean_squared_error(y_pred,y_test)))
# computing permutation importance of each feature in dataset
perm  = PermutationImportance(clf, random_state=1).fit(X_train, y_train)

# # create a structured array
# dtype = [('feature', str), ('permutation_weights', float)]
features = np.array(X_train.columns.to_list())
permF = np.array([X_train.columns.to_list, perm.feature_importances_])
rankedFeatureIds   = perm.feature_importances_.argsort()[::-1] #[::-1] reverses the ascending result of argsort, indices of arrays sorted
rankedFeatures     = features[rankedFeatureIds]
numRanks           = 15 # RF: At 15 (score:0.786) and 30 max scores. Score declines with decreasing features
                        # LR: at 10 scores were less than full-set but at 15 features, scores halved.
                        # With PermutationInporatance RF performs better than LR
featuresTopNRanks  = list(rankedFeatures[0:numRanks])
featuresToDrop     = list(rankedFeatures[numRanks:-1])
print('Selected features \n', featuresTopNRanks)
print('features to drop:\n', repr(featuresToDrop))
# print('Feature importance in accordance to weights\n')
# print(features[permFeatureRanks])

# only for printing in readable format
permExpWghts = eli5.explain_weights(perm,feature_names = X_train.columns.to_list())
permFeatureRanksText =eli5.format_as_text(permExpWghts) # only for printing
print(permFeatureRanksText)

# based on importance, select only top 10 columns for building model

dataDFp = dataDF.copy()
fdataDFp = dataDFp[featuresTopNRanks]
X_train, X_test, y_train, y_test = train_test_split(fdataDFp,outcomeVarDF,test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=0).fit(X_train,y_train)

print('RF Model fitted with'+ repr(numRanks) + 'features\n RF feature importances\n')
important_features = pd.Series(data=clf.feature_importances_[0:numRanks],index=featuresTopNRanks)
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(featuresTopNRanks, clf.feature_importances_[0:numRanks]):
    feats[feature] = importance #add the name/value pair

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)
plt.rcParams.update({'font.size': 5})
plt.show()
print('RF Score')
print(clf.score(X_test,y_test))
print('RF decision path')
pprint.pprint(clf.decision_path(X_train))
y_pred = clf.predict(X_test)
print('RMSE\n')
print(np.sqrt(metrics.mean_squared_error(y_pred,y_test)))

