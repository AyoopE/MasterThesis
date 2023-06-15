###### Note: You have to create two empty folders in the working dirctory with names: "images" and "results"

from all_functions import *


###################### Dataset 1 ######################


# The follwoing dataset has 82 samples for patients (39 tests with progression and 43 tests with stable status) and 364 features
# Note: 
# Status variable contains the status of tests for patients if it is progression (1) or stable (0)
# Period_rounded: is the period in months from B1-test to the current test
# Progression: 1 if there is progression, or 0 if there is no progression in the test (stable)
# Status variable is actually the same as Progression variable
# All variables that starts with "Var-" represents the 364 features

data = pd.read_csv('Second_Classification_Problem_data_part1.csv', delimiter = ";")
data.head()

data.shape

#data.dtypes
data.describe()


####### Data perprocessing for Dataset 1 #######


#Check total missing values in dataset
np.sum( data.isnull().sum() )


#Check missing values in rows
mask = data.isnull().any(axis=1)
data[mask]


#Check missing values in columns
mask = data.isnull().any(axis=0)
data.columns[mask]


#All persons in the dataset are patients, so missing values will be replaced by mean value in corresponding features
var_list = data.columns[mask].to_list()

for i in var_list:
    data[i] = data[i].fillna(data[i].mean())


#Check if there is any missing value left
np.sum( data.isnull().sum() )


# Identify the features (input_data) and target (output_data) in the dataset
input_data = data.drop(['Person', 'Status', 'Progression', 'Period_rounded'], axis=1)
output_data = data[['Status']]


# So we have in total 364 features and 82 samples (39 tests with progression and 43 tests with stable status)
input_data.shape



# Plot features in 2D space normally
plot_var_2d_rev02('Var-15', 'Var-25', input_data, 'n')


# Plot features in 2D space with applied power function
plot_var_2d_rev02('Var-15', 'Var-25', input_data, 'p')


# New aspect of features and target variables
X = np.power(input_data, (1/5))
y = output_data


###################### Dataset 2 ######################

data2 = pd.read_csv('Second_Classification_Problem_data_part2.csv', delimiter = ";")
data2.head()


########## Data perprocessing Dataset 2 #########

#Check total missing values in dataset
np.sum( data2.isnull().sum() )


#Check missing values in rows
mask = data2.isnull().any(axis=1)
data2[mask]


#Check missing values in columns
mask = data2.isnull().any(axis=0)
data2.columns[mask]


#Since the missing values for each row or column are not significant, and all samples in the dataset are for patients, 
#so the missing values are going to be replaced by the mean value of each feature
L = data2.columns[mask].to_list()

for item in L:
    data2[item] = data2[item].fillna(data2[item].mean())


#Check if there is any missing value left
np.sum( data2.isnull().sum() )


# Identify the features in the dataset
input_data2 = data2.drop(['Person', 'Progression', 'Period_rounded'], axis=1)

# Columns in the new dataset:
# Period_rounded: is the period in months from B1-test to the current test
# Progression: 1 if there is progression, or 0 if there is no progression in the test (stable)


# New aspect of features
X2 = np.power(input_data2, (1/5))


# # Function for data frame for all patients and their tests
def main_func(X_new_chosen, clf):
    
    #This function is created here locally since the procedures may be used many times
    #with new selected features or new classifier
    
    #Function arguments:-
    #X_new_chosen: data frame of chosen features
    #clf: classifier

    #####################################################################################################
    
    # dataset 1

    X3 = X[X_new_chosen.columns]
    
    #Create new DataFrame for all patients in dataset 1
    Test_data_1 = pd.DataFrame(data[['Person', 'Progression', 'Period_rounded']])

    #Add new column for prediction for all patients in dataset 1
    Test_data_1['Prediction'] = clf.predict_proba(X3)[:,1]

    #Make some changes to column "Person"
    Test_data_1['Person'] = Test_data_1['Person'].str.replace('Met_Fraction_', '')

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

    return df_all


###################### Feature selection ######################
#Note: The results for selected features, selected top hyperparameters and accuracy score
#could change each time depending on the selection of some features with low 
#importance in each cross-validation loop in the feature selection process.
#Results here are more stable with Lasso feature selection method than other methods
#The function "select_features_clf" returns the features as dataframe with precentage of appearnce.
#The function also returns the top selected hyperparameters

features_lasso, top_param_lasso = select_features_clf(X, y, 'Lasso')

features_tree, top_param_tree = select_features_clf(X, y, 'Tree-based')

features_forward, top_param_forward = select_features_clf(X, y, 'Forward Selection')

features_backward, top_param_backward = select_features_clf(X, y, 'Backward Selection')


features_lasso.head(50)


###################### Evaluation of classifiers using selected features ######################

#This function 'test_ncv' is for choosing any number of features and top hyperparameters from any method to be 
#tested with the function "test_classifiers" to check which gorup of features will give us the best performance
#I tried the most frequent features with about 50% appearance and above, and this gave me good results

def test_ncv(k, i):
    
# Choose the top features and top hyperparameters:
# k: is the method (1 for Lasso , 2 for Tree-based , 3 for Forward Selection, 4 for Backward Selection)
# i: is the number of features
    
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


# Note: My choices were: 1.Lasso=32 , 2.Tree-based=11 , 3.Forward=9 , 4.Backward=14
# Insert -1 as the number of features to choose all of them
X_new_chosen, top_param = test_ncv(1, 32)


# This shows best hyperparameter for each classifier from nested cross validation:
clf_list = ['LR', 'SVC', 'KNN', 'RF', 'NB']
for k in np.arange(5):
    print('\n Best hyperparameter for ' + clf_list[k] + ':\n')
    print(top_param[k])


# Test classifiers with selected features and top hyperparameters
test_classifiers(X_new_chosen, y, top_param, 'ROC')



###################### Continue with Logistic Regression Classifier and features selected by Lasso ######################

# Find best parameters for logistic Regression Classifier with just the selected features by Lasso
best_param = clf_h_p_opt(X_new_chosen, y)


# Create LogisticRegression object and pipeline
LR = LogisticRegression(random_state=0)
pipe_LR = Pipeline([('scaler', StandardScaler()), ('LR', LR)])

# Assign best parameters for LR
pipe_LR.set_params(**best_param[0])

# Fit with the whole dataset (with selected features)
pipe_LR.fit(X_new_chosen, y.values.ravel())

# The model is ready to be used



###################### Predict status for patients in dataset 2 ######################

# Prediction for dataset 2 (Stable vs Progression):

X4 = X2[X_new_chosen.columns]

#Create new DataFrame for all patients in dataset 2
dataset_2_results = pd.DataFrame(data2[['Person', 'Progression', 'Period_rounded']])

#Add new column for binary prediction for all patients in dataset 2
dataset_2_results['Prediction'] = pipe_LR.predict(X4)

#Make some changes to column "Person"
dataset_2_results['Person'] = dataset_2_results['Person'].str.replace('Met_Fraction_', '')

##Export DataFrame dataset_2_results to csv file DataFrame dataset_2_results to csv file
#dataset_2_results.to_csv('./results/2nd_classifier_dataset_2_results', index=False)

dataset_2_results


# ### Total disease cases classified as progression:
len (dataset_2_results[ dataset_2_results['Prediction'] == 1 ])


# ### Total disease cases classified as stable:
len (dataset_2_results[ dataset_2_results['Prediction'] == 0 ])



###################### Monitoring of ctDNA level ######################

# ## Predict the probability for all tests for patients in both datasets and create data frame for all patients
df_all = main_func(X_new_chosen, pipe_LR)

# Export Data Frame df_all to csv file
df_all.to_csv('./results/2nd_classification_df_all_LR', index=False)


# # Create a plot for all tests for each patient
plot_each_patient(df_all)


# # Create a plot for tests with first progression as reference point
#Note: months_1 is the database for the created plots
months_1 = plot_1st_prog_ref(df_all)


# # Wilcoxon Signed-Rank Test for first progression as reference point
print(wilcoxon_1st_prog(months_1))


# # Create a plot for tests with first test (B1 test) as reference point
#Note: months_2 is the database for the created plots
months_2 = plot_B1_test_ref(df_all)


# # Wilcoxon Signed-Rank Test for B1 test as reference point
print(wilcoxon_B1_test(months_2))


