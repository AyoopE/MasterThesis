###### Note: You have to create two folders in the working dirctory with names: "images" and "results"

from all_functions import *


###################### Dataset 1 ######################

# The following dataset has 98 samples (67 samples for patients and 31 samples for normal) and 364 features
# Note:
# Mut_Freq variable contains mutation frquency values for all persons (patients & healthy people)
# Period_rounded: is the period in months from B1-test to the current test
# Progression: 1 if there is progression, or 0 if there is no progression in the test (stable)
# All variables that starts with "Var-" represents the 364 features

data = pd.read_csv('Regression_Problem_data_part1.csv', delimiter = ";")

data.tail()

data.shape

data.dtypes

#data.dtypes
data.describe()



########## Data perprocessing #########

#Check total missing values in dataset
np.sum( data.isnull().sum() )


#Missing values in colomns
mask = data.isnull().any(axis=1)
data[mask]


#Check missing values in rows
mask = data.isnull().any(axis=1)
data[mask]


#Check missing vlaues individually row by row
for i in np.arange(data[mask].index.size):
    print('For person with index ' + str(data[mask].index[i]) + ' total missing values are : ' 
          + str( data.iloc[ data[mask].index[i] ].isnull().sum() ) )

    
#Sample with index 30 (normal person) contains 37 missing values which is significant (about 10%) and it will be removed
data = data.drop(index=([30])).reset_index(drop=True)


#Check missing values in columns
mask = data.isnull().any(axis=0)
data.columns[mask]


#Check missing vlaues individually column by column
L = data.columns[mask].to_list()

for i in np.arange(len(L)):
    print('For feature ' + str(L[i]) + ' total missing values are: ' 
          + str( data[L[i]].isnull().sum() ) )

    
#Missing values in colomns again
mask = data.isnull().any(axis=1)
data[mask]


#Check missing vlaues in sample with index 29
mask = data.iloc[29].isnull()
data.columns[mask]


#Since the sample with index 29 is for normal, I replace the missing value by mean value of other normal samples
data.loc[29, 'Var-193'] = data['Var-193'].iloc[0:30].mean()
data.loc[29, 'Var-363'] = data['Var-363'].iloc[0:30].mean()


#Check missing vlaues in sample with index 61
mask = data.iloc[61].isnull()
data.columns[mask]


#Check missing vlaues in sample with index 62
mask = data.iloc[62].isnull()
data.columns[mask]


#Since the samples with indices 61 & 62 are for patients, I replace the missing value by mean value of other patient samples
data.loc[61, 'Var-285'] = data['Var-285'].iloc[30:97].mean()
data.loc[62, 'Var-315'] = data['Var-315'].iloc[30:97].mean()
data.loc[62, 'Var-331'] = data['Var-331'].iloc[30:97].mean()
data.loc[62, 'Var-363'] = data['Var-363'].iloc[30:97].mean()


#Check if there is any missing value left
np.sum( data.isnull().sum() )


# We need to shuffle the dataset since the used feature selection methods do not have built-in shuffle option like in 
# previous classification cases.
data_shuffled = data.sample(frac = 1, random_state=0).reset_index(drop=True)
data_shuffled.head(15)


# Identify the features (input_data) and target (output_data) in the dataset
input_data = data_shuffled.drop(['Person', 'Mut_Freq','Progression', 'Period_rounded'], axis=1)
output_data = data_shuffled[['Mut_Freq']]


# So we have in total 364 features and 97 samples (67 samples for patients and 30 samples for normal)
input_data.shape


# New aspect of features and target variables
X = input_data
y = output_data



###################### Dataset 2 ######################

data2 = pd.read_csv('Regression_Problem_data_part2.csv', delimiter = ";")
data2.head()


#Check total missing values in dataset
np.sum( data2.isnull().sum() )


#Check missing values in rows
mask = data2.isnull().any(axis=1)
data2[mask]


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
# Progression: 1 if there is progression, or 0 if there is no progression in the test (stable)


# New aspect of features
X2 = input_data2



# # Function for data frame for all patients and their tests
def main_func(X_new_chosen, Reg):
    
    #This function is created here locally since the procedures will used many times
    #with new selected features or new regressor
    
    #Function arguments:-
    #X_new_chosen: data frame of chosen features
    #Reg: Regressor
    
    #Note: X and X2 are features for the two datasets and has to be excuted before run this function

    #####################################################################################################
    
    # dataset 1

    X3 = X[X_new_chosen.columns]
    
    #Create new DataFrame for all patients in dataset 1
    Test_data_1 = pd.DataFrame(data_shuffled[['Person', 'Progression', 'Period_rounded']])

    #Add new column for prediction for all patients in dataset 1
    Test_data_1['Prediction'] = Reg.predict(X3)

    #Make some changes to column "Person"
    Test_data_1['Person'] = Test_data_1['Person'].str.replace('Met_Fraction_', '')
    
    #To delete control persons
    Test_data_1 = Test_data_1.sort_values(by='Person', ascending=False).reset_index(drop=True)
    Test_data_1 = Test_data_1.iloc[30:].reset_index(drop=True)

    #Print DataFrame Test_data_1 to csv file
    #Test_data_1.to_csv('Test_data_1', index=False)   

    #####################################################################################################

    # dataset 2
    
    X4 = X2[X_new_chosen.columns]

    #Create new DataFrame for all patients in dataset 2
    Test_data_2 = pd.DataFrame(data2[['Person', 'Progression', 'Period_rounded']])

    #Add new column for prediction for all patients in dataset 2
    Test_data_2['Prediction'] = Reg.predict(X4)

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

    return df_all


###################### Feature selection ######################

Best_features = feature_selection_reg(X, y, 'LassoCV')



#Note: My choices: Lasso 40  Ridge 90 ElasticNet 90
# Select number of features
X_new_chosen = X[Best_features.iloc[0:40]['Feature']]
X_new_chosen


###################### Evaluation of Regressors ######################

# ### Hyperparameters Optimization
# Note: Regressors (LassoCV, RidgeCV, ElasticNetCV) do not need optimization. The best model is selected by cross-validation.
# By default CV folds for LassoCV and ElasticNetCV is 5, while for RidgeCV, I set the CV folds value manually to 5

# Find best parameters for Random Forest Regressor by using selected features
param = RF_hyp_opt(X_new_chosen, y)


# ### Evaluation by 10-fold cross validation
Results_df = test_regressors(X_new_chosen, y, param)



###################### Forward with 40 Lasso features and RidgeCV regressor ######################

# # Test RidgeCV regressor with train_test_split


# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new_chosen, y, test_size=0.20, shuffle=True, random_state=0)


# Create linear regression object

R_CV = linear_model.RidgeCV(cv=5)

pipe_R_CV = Pipeline([('scaler', StandardScaler()), ('R_CV', R_CV)])

# Train the model using the training sets
pipe_R_CV.fit(X_train, y_train.values.ravel())

# Make predictions using the testing set
y_pred = pipe_R_CV.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


R = pd.DataFrame(y_test)

R['Prediction'] = y_pred

D = PredictionErrorDisplay.from_predictions(R['Mut_Freq'], R['Prediction'], kind="actual_vs_predicted", scatter_kwargs={"alpha": 0.5})

plt.grid()
plt.title('RidgeCV Regressor')
plt.show()


###################### Selected Model:  RidgeCV Regressor ######################

# # RidgeCV Regressor

# RidgeCV Regressor find the best parameters internally by cross validation

# Create RidgeCV Regressor object & pipeline
R_CV = linear_model.RidgeCV(cv=5) 
pipe_R_CV = Pipeline([('scaler', StandardScaler()), ('R_CV', R_CV)])

#### Fit with all data
pipe_R_CV.fit(X_new_chosen, y.values.ravel())

# Now RidgeCV Regressor is ready to be used



###################### Predict of mutation frequency for patients in dataset 2 ######################

# Prediction for dataset 2 (Mutation frequency)

X4 = X2[X_new_chosen.columns]

#Create new DataFrame for all patients in dataset 2
dataset_2_results = pd.DataFrame(data2[['Person', 'Progression', 'Period_rounded']])

#Add new column for binary prediction for all patients in dataset 2
dataset_2_results['Prediction'] = pipe_R_CV.predict(X4)

#Make some changes to column "Person"
dataset_2_results['Person'] = dataset_2_results['Person'].str.replace('Met_Fraction_', '')

dataset_2_results

##Export DataFrame dataset_2_results to csv file DataFrame dataset_2_results to csv file
dataset_2_results.to_csv('./results/RidgeCV_dataset_2_results', index=False)




###################### Monitoring of ctDNA level ######################

# ## Create data frame for all patients in both datasets using selected features and classifier
df_all = main_func(X_new_chosen, pipe_R_CV)


# # Make a plot for all tests for each patient
#You have to create a folder "images" in current directory
plot_each_patient(df_all)


# # Plot for tests with first progression as reference point
months_1 = plot_1st_prog_ref(df_all)


# # Wilcoxon Signed-Rank Test for first progression as reference point
wilcoxon_1st_prog(months_1)


# # Plot for tests  with first test (B1 test) as reference point
months_2 = plot_B1_test_ref(df_all)


# # Wilcoxon Signed-Rank Test for B1 test as reference point
wilcoxon_B1_test(months_2)

