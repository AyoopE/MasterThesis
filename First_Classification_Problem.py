
from all_functions import *


# The following dataset has 71 samples (40 patients and 31 normal) and 364 features
# Note: Status variable contains the status of tests for persons if he is patient (1) or healthy (0)
# All variables that starts with "Var-" represents the 364 features

data = pd.read_csv('First_Classification_Problem_data_part1.csv', delimiter = ";")

data.head()

data.shape

#data.dtypes
data.describe()


# # Data perprocessing

#Check total missing values in dataset
np.sum( data.isnull().sum() )


#Check missing values in columns
mask = data.isnull().any(axis=0)
data.columns[mask]


#Check missing values in rows
mask = data.isnull().any(axis=1)
data[mask]


#Check missing vlaues individually row by row
for i in np.arange(data[mask].index.size):
    print('For person with index ' + str(data[mask].index[i]) + ' total missing values are : ' 
          + str( data.iloc[ data[mask].index[i] ].isnull().sum() ) )


#Sample with index 70 (it is the last sample) contains most of the missing values and it will be removed
data = data.drop(index=([70]))



#Check missing vlaues in sample with index 38
mask = data.iloc[38].isnull()
data.columns[mask]


#Since the sample with index 38 is for patient, I replace the missing value by mean value of other patient samples

data.loc[38, 'Var-20'] = data['Var-20'].iloc[0:40].mean()



#Check missing vlaues in sample with index 69
mask = data.iloc[69].isnull()
data.columns[mask]


#Since the sample with index 69 is for normal, I replace the missing value by mean value of other normal samples

data.loc[69, 'Var-193'] = data['Var-193'].iloc[40:70].mean()
data.loc[69, 'Var-363'] = data['Var-363'].iloc[40:70].mean()



#Check if there is any missing value left
np.sum( data.isnull().sum() )



# Identify the features (input_data) and target (output_data) in the dataset
input_data = data.drop(['Person', 'Status'], axis=1)
output_data = data[['Status']]



# So we have in total 70 samples (40 patients and 30 normal) and 364 features
input_data.shape



# Plot features in 2D space
plot_var_2d_rev01('Var-15', 'Var-25', input_data, 'n')



# Plot features in 2D space
plot_var_2d_rev01('Var-15', 'Var-25', input_data, 'p')



# New aspect of features and target variables
X = np.power(input_data, (1/5))
y = output_data



# # Dataset 2

data2 = pd.read_csv('First_Classification_Problem_data_part2.csv', delimiter = ";")
data2.head()


#Check total missing values in dataset
np.sum( data2.isnull().sum() )


#Check missing values in rows
mask = data2.isnull().any(axis=1)
data2[mask]


#Check missing vlaues individually row by row

for i in np.arange(data2[mask].index.size):
    print('For person with index ' + str(data2[mask].index[i]) + ' total missing values are : ' 
          + str( data2.iloc[ data2[mask].index[i] ].isnull().sum() ) )


#Check missing values in columns
mask = data2.isnull().any(axis=0)
data2.columns[mask]


#Check missing vlaues individually column by column

L = data2.columns[mask].to_list()

for i in np.arange(len(L)):
    print('For feature ' + str(L[i]) + ' total missing values are: ' 
          + str( data2[L[i]].isnull().sum() ) )


#Since the missing values for each row or column are not significant, and all samples in the dataset are for patients, 
#so the missing values are going to be replaced by the mean value of each feature
for item in L:
    data2[item] = data2[item].fillna(data2[item].mean())


#Check if there is any missing value left
np.sum( data2.isnull().sum() )



# Identify the features in the dataset (input_data2)
input_data2 = data2.drop(['Person', 'Progression', 'Period_rounded'], axis=1)

# Columns in the new dataset:
# Period_rounded: is the period in months from B1-test to the current test
# Progression: 1 if there is progression, or 0 if there is no progression in the test


# New aspect of features
X2 = np.power(input_data2, (1/5))


#test_classifiers(X_new_chosen_nested_cv, y, 'ROC')


# # Function for data frame for all patients and their tests

def main_func(X_new_chosen, clf):
    
    #This function is created here locally since the procedures will used many times
    #with new selected features or new classifier
    
    #Function arguments:-
    #X_new_chosen: data frame of chosen features
    #clf: classifier

    #####################################################################################################
    
    # dataset 1

    X3 = X[X_new_chosen.columns]

    #Create new DataFrame for just patients from the baseline dataset (dataset 1)
    Test_data_1 = pd.DataFrame(data['Person'][0:40]) 

    #Add "Progression" column since the baseline dataset does not have one, and values are 0 (no progression)
    Test_data_1['Progression'] = 0

    #Add "Period_rounded" column since the baseline dataset does not have one, and the time to B1-test is 0
    Test_data_1['Period_rounded'] = 0

    #Add new column for prediction for 40 patients in baseline dataset
    Test_data_1['Prediction'] = clf.predict_proba(X3[0:40])[:,1]

    #Make some changes to column "Person"
    Test_data_1['Person'] = Test_data_1['Person'].str.replace('Met_Fraction_', '')
    Test_data_1['Person'] = Test_data_1['Person'] + '_B1'

    #Print DataFrame Test_data_1 to csv file
    #Test_data_1.to_csv('Test_data_1', index=False)

    #####################################################################################################

    # dataset 2

    X4 = X2[X_new_chosen.columns]

    #Create new DataFrame for all patients in dataset 2
    Test_data_2 = pd.DataFrame(data2[['Person', 'Progression', 'Period_rounded']])

    #Add new column for prediction for all patients in dataset 2
    Test_data_2['Prediction'] = clf.predict_proba(X4)[:,1]

    #Make some changes to column "Person"
    Test_data_2['Person'] = Test_data_2['Person'].str.replace('Met_Fraction_', '')

    #Print DataFrame Test_data_2 to csv file
    #Test_data_2.to_csv('Test_data_2', index=False)

    #####################################################################################################

    # DataFrame for all patients and tests from both datasets (dataset 1 & dataset 2)

    #Make new DataFrame for all patients and tests from both Dataframes Test_data_1 & Test_data_2
    df_all = pd.concat([Test_data_1, Test_data_2],axis=0, ignore_index=True)

    #Split Persons and Tests in "Person" column into two seperated columns
    df_all[['Person','Test']] = df_all['Person'].str.split('_',expand=True)

    #Rearrange the columns in the DataFrame
    df_all = df_all[['Person','Test', 'Period_rounded', 'Progression', 'Prediction']]

    #Add 0 to persons and tests with one digit to be sorted correctly. (To make all persons and tests contain two digits)
    df_all['Person'][df_all['Person'].str.len() == 2]  = df_all['Person'].str.replace('P', 'P0')
    df_all['Test'][df_all['Test'].str.len() == 2]  = df_all['Test'].str.replace('B', 'B0')

    #Sort the whole DataFrame by persons then by tests
    df_all = df_all.sort_values(by=['Person', 'Test'], ascending=True).reset_index(drop=True)
    
    #Print DataFrame df_all to csv file
    #df_all.to_csv('df_all', index=False)

    return df_all


#  

# # Part 1 (Predict patient vs normal)

#  

#  

# # Feature Selection


features_lasso, top_param_lasso = select_features_clf(X, y, 'Lasso', 'Yes')


features_tree, top_param_tree = select_features_clf(X, y, 'Tree-based')


# In[36]:


features_forward, top_param_forward = select_features_clf(X, y, 'Forward Selection')


# In[37]:


features_backward, top_param_backward = select_features_clf(X, y, 'Backward Selection')


# In[38]:


features_lasso.head(50)


# # Evaluation of classifiers using selected features

# In[39]:


# Choose the top features and top hyperparameters:
# k: is the method (1 for Lasso , 2 for Tree-based , 3 for Forward Selection, 4 for Backward Selection)
# i: is the number of features

def test_ncv(k, i):
    
    if k == 1:
        X_new_chosen = X[(features_lasso.iloc[0:i,:]['Feature'].to_list())]
        top_param = top_param_lasso
    
    if k == 2:
        X_new_chosen = X[(features_tree.iloc[0:i,:]['Feature'].to_list())]
        top_param = top_param_tree

    if k == 3:
        X_new_chosen = X[(features_forward.iloc[0:i,:]['Feature'].to_list())]
        top_param = top_param_forward

    if k == 4:
        X_new_chosen = X[(features_backward.iloc[0:i,:]['Feature'].to_list())]
        top_param = top_param_backward
    
    return X_new_chosen, top_param


# In[40]:


# 1.Lasso=23 , 2.Tree-based=12 , 3.Forward=11 , 4.Backward=16
# Insert -1 as the number of features to choose all of them
X_new_chosen, top_param = test_ncv(1, 23)


# In[41]:


# Best hyperparameter for each classifier from nested cross validation:

clf_list = ['LR', 'SVC', 'KNN', 'RF', 'NB'] #'QDA', 
for k in np.arange(5):
    print('\n Best hyperparameter for ' + clf_list[k] + ':\n')
    print(top_param[k])


# In[42]:


# Test classifiers with selected features
test_classifiers(X_new_chosen, y, top_param, 'ROC')


# # Continue with Logestic Regression Classifier and features selected by Lasso

# In[43]:


# Find best parameters for logistic Regression Classifier with just the selected features by Lasso
best_param = clf_h_p_opt(X_new_chosen, y)


# In[44]:


# Create LogisticRegression object and pipeline
LR = LogisticRegression(random_state=0)
pipe_LR = Pipeline([('scaler', StandardScaler()), ('LR', LR)])

# Assign best parameters for LR
pipe_LR.set_params(**best_param[0])

# Fit with the whole dataset (with selected features)
pipe_LR.fit(X_new_chosen, y.values.ravel())

# The model is ready to be used


# # Predict status for patients in dataset 2

# In[45]:


# Test of dataset 2 (Patient vs Normal)

X4 = X2[X_new_chosen.columns]

#Create new DataFrame for all patients in dataset 2
dataset_2_results = pd.DataFrame(data2[['Person', 'Progression', 'Period_rounded']])

#Add new column for binary prediction for all patients in dataset 2
dataset_2_results['Prediction'] = pipe_LR.predict(X4)

#Make some changes to column "Person"
dataset_2_results['Person'] = dataset_2_results['Person'].str.replace('Met_Fraction_', '')

#Export DataFrame dataset_2_results to csv file
dataset_2_results.to_csv('./results/1st_classifier_dataset_2_results_LR', index=False)

dataset_2_results


# ### Accuracy of predition

# In[46]:


print( np.round(len(dataset_2_results[ dataset_2_results['Prediction'] == 1 ]) / 
                len(dataset_2_results['Prediction']) * 100, 2), '%')


#  

# # Part 2 (Monitoring of ctDNA level)

#  

#  

# # Test by new classifier Logistic Regression using 23 features selected by Lasso

# ## Create data frame for all patients in both datasets using selected features and classifier

# In[47]:


df_all = main_func(X_new_chosen, pipe_LR)


# # Make a plot for all tests for each patient

# In[48]:


#You have to create a folder "images" in current directory:

plot_each_patient(df_all)


# #### Plots
# 

# # Plot for tests with first progression as reference point

# In[49]:


months_1 = plot_1st_prog_ref(df_all)


# #### Plots

# In[50]:


months_1#.to_csv('./months_1.csv', mode = 'w')


# # Wilcoxon Signed-Rank Test for first progression as reference point

# In[52]:


wilcoxon_1st_prog(months_1)


# # Plot for patients tests with first test (B1-test) as reference point

# In[53]:


months_2 = plot_B1_test_ref(df_all)


# ### Plot

# In[54]:


months_2


# # Wilcoxon Signed-Rank Test for B1 test as reference point

# In[55]:


wilcoxon_B1_test(months_2)


# # (New trial) Test by Naive Bayes using 23 features selected by Lasso

# In[56]:


# Create GaussianNB object and pipeline
NB = GaussianNB()
pipe_NB = Pipeline([('scaler', StandardScaler()), ('NB', NB)])

# Assign best parameters for NB
pipe_NB.set_params(**best_param[4])

# Fit with the whole dataset (with selected features)
pipe_NB.fit(X_new_chosen, y.values.ravel())

# The model is ready to be used


# ## Create data frame for all patients in both datasets using selected features and classifier

# In[57]:


df_all = main_func(X_new_chosen, pipe_NB)


# # Make a plot for all tests for each patient

# In[58]:


#You have to create a folder "images" in current directory:

plot_each_patient(df_all)


# #### Plots
# 

# # Plot for tests with first progression as reference point

# In[59]:


months_1 = plot_1st_prog_ref(df_all)


# #### Plots

# In[60]:


months_1#.to_csv('./months_1.csv', mode = 'w')


# # Wilcoxon Signed-Rank Test for first progression as reference point

# In[61]:


wilcoxon_1st_prog(months_1)


# # Plot for patients tests with first test (B1 test) as reference point

# In[62]:


months_2 = plot_B1_test_ref(df_all)


# ### Plot

# In[63]:


months_2


# # Wilcoxon Signed-Rank Test for B1 test as reference point

# In[64]:


wilcoxon_B1_test(months_2)


# # Try new method; Lasso features with lowest p_value

# In[65]:


p_value = []

for i in np.arange(features_lasso.index.size):
    p_value.append(float(T_test_rev01(X[[features_lasso.iloc[i,0]]])))

p_value_features = features_lasso
p_value_features['p_value'] = p_value
p_value_features = p_value_features.sort_values(by='p_value')

p_value_features.head(20)


# In[66]:


X_new_p_value = X[(p_value_features.iloc[0:8,:]['Feature'].to_list())]

X_new_p_value


# In[67]:


chosen_features(X_new_p_value)


# # Leave-one-out Evaluation (Lasso features)

# In[68]:


Leave_one_out(X_new_chosen, y, pipe_LR)


# # Leave-one-out Evaluation (lowest p_value features)

# In[69]:


best_param_2 = clf_h_p_opt(X_new_p_value, y)


# In[70]:


# Create LogisticRegression object and pipeline
LR = LogisticRegression(random_state=0)
pipe_LR_2 = Pipeline([('scaler', StandardScaler()), ('LR', LR)])

# Assign best parameters for LR
pipe_LR_2.set_params(**best_param_2[0])

Leave_one_out(X_new_chosen, y, pipe_LR_2)


# # New evaluation of classifiers

# In[71]:


# Find best parameters for Classifiers with new selected features
best_param_2 = clf_h_p_opt(X_new_p_value, y)


# In[72]:


# Test classifiers with new selected features
test_classifiers(X_new_p_value, y, best_param_2, 'ROC')


# # Decided to continue with: Random Forest Classifier

# # Random Forest Classifier

# In[73]:


# Create RandomForestClassifier object and pipeline
RF = RandomForestClassifier(random_state=0)
pipe_RF = Pipeline([('scaler', StandardScaler()), ('RF', RF)])

# Assign best parameters for RF
pipe_RF.set_params(**best_param_2[3])

# Fit with the whole dataset (with selected features)
pipe_RF.fit(X_new_p_value, y.values.ravel())

# The model is ready to be used


# # Predict status for patients in dataset 2

# In[74]:


# Test of dataset 2 (Patient vs Normal)

X4 = X2[X_new_p_value.columns]

#Create new DataFrame for all patients in dataset 2
dataset_2_results = pd.DataFrame(data2[['Person', 'Progression', 'Period_rounded']])

#Add new column for binary prediction for all patients in dataset 2
dataset_2_results['Prediction'] = pipe_RF.predict(X4)

#Make some changes to column "Person"
dataset_2_results['Person'] = dataset_2_results['Person'].str.replace('Met_Fraction_', '')

#Export DataFrame dataset_2_results to csv file
dataset_2_results.to_csv('./results/1st_classifier_dataset_2_results_RF', index=False)

dataset_2_results


# ### Accuracy of predition

# In[75]:


print( np.round(len(dataset_2_results[ dataset_2_results['Prediction'] == 1 ]) / 
                len(dataset_2_results['Prediction']) * 100, 2), '%')


# ## Create data frame for all patients in both datasets using selected features and classifier

# In[76]:


df_all = main_func(X_new_p_value, pipe_RF)


# # Make a plot for all tests for each patient

# In[77]:


#You have to create a folder "images" in current directory:

plot_each_patient(df_all)


# #### Plots
# 

# # Plot for tests with first progression as reference point

# In[78]:


months_1 = plot_1st_prog_ref(df_all)


# #### Plots

# In[79]:


months_1#.to_csv('./months_1.csv', mode = 'w')


# # Wilcoxon Signed-Rank Test for first progression as reference point

# In[80]:


wilcoxon_1st_prog(months_1)


# # Plot for patients tests with first test (B1 test) as reference point

# In[81]:


months_2 = plot_B1_test_ref(df_all)


# ### Plot

# In[82]:


months_2


# # Wilcoxon Signed-Rank Test for B1 test as reference point

# In[ ]:


wilcoxon_B1_test(months_2)


# In[ ]:




