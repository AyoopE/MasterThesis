import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.metrics import PredictionErrorDisplay, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, SelectKBest, mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor


#This function is for the first classification problem
#Function apply t-test between patients and normal persons values in each feature
def T_test_rev01(var):
    
    t1 = var.iloc[0:40, :]
    t2 = var.iloc[40:70, :]
    
    _ , p_value = stats.ttest_ind(t1, t2, equal_var=False, alternative='two-sided')

    return p_value


#This function is for the second classification problem
def T_test_rev02(var):
    
    t1 = var.iloc[0:39, :]
    t2 = var.iloc[39:82, :]
    
    _ , p_value = stats.ttest_ind(t1, t2, equal_var=False, alternative='two-sided')

    return p_value


#This function is for the first classification problem
#Function to test statistics between patients and normal persons values in a feature
def check_var_stat(S, input_data):
    
    ###Arguments:
    #S: is the feature
    #input_data: is a datafarme where the feature "S" is a part of
    
    Test1 = input_data.iloc[0:40, :]
    Test2 = input_data.iloc[40:70,:]
    B1 = Test1[S]
    B2 = Test2[S]
    C = pd.concat([B1.describe(), B2.describe()], axis=1, keys= [S +' patients values', S +' healthy values'])
    print(C)
    
    return


#This function is for the first classification problem
def plot_var_2d_rev01(v1, v2, input_data, Met):
    
    if Met == 'n':
        X_data = input_data
    elif Met == 'l':
        X_data = np.log10(input_data+1)
    elif Met == 'p':
        X_data = np.power(input_data, (1/5))
    else:
        return print('Wrong choice')
        
    X_data_1 = X_data.iloc[0:40, :]
    X_data_2 = X_data.iloc[40:70, :]

    p_df = pd.concat([X_data_1[v1], X_data_1[v2]], axis=1)
    c_df = pd.concat([X_data_2[v1], X_data_2[v2]], axis=1)

    #Plot
    ax = plt.gca()
    
    p_df.plot(kind = 'scatter', x = v1, y = v2, color = 'red', ax = ax, figsize=(18, 12))
    
    c_df.plot(kind = 'scatter', x = v1, y = v2, color = 'blue', ax = ax)
    
    ax.set_xlabel(v1, fontdict={'fontsize':20})
    ax.set_ylabel(v2, fontdict={'fontsize':20})
    ax.legend(["2D point for patient", "2D point for healthy"], loc='best', fontsize=16);
    
    return


#This function is for the second classification problem
def plot_var_2d_rev02(v1, v2, input_data, Met):
    
    if Met == 'n':
        X_data = input_data
    elif Met == 'l':
        X_data = np.log10(input_data+1)
    elif Met == 'p':
        X_data = np.power(input_data, (1/5))
    else:
        return print('Wrong choice')
        
    X_data_1 = X_data.iloc[0:39, :]
    X_data_2 = X_data.iloc[39:82, :]

    p_df = pd.concat([X_data_1[v1], X_data_1[v2]], axis=1)
    c_df = pd.concat([X_data_2[v1], X_data_2[v2]], axis=1)

    #Plot
    ax = plt.gca()
    
    p_df.plot(kind = 'scatter', x = v1, y = v2, color = 'red', ax = ax, figsize=(18, 12))
    
    c_df.plot(kind = 'scatter', x = v1, y = v2, color = 'blue', ax = ax)
    
    ax.set_xlabel(v1, fontdict={'fontsize':20})
    ax.set_ylabel(v2, fontdict={'fontsize':20})
    ax.legend(["2D point for patient with progression status", "2D point for patient with stable status"], loc='best', fontsize=16);
    
    return


#This function is for the first classification problem
def chosen_features(X):
    
    X_p = X.iloc[0:40, :]
    X_c = X.iloc[40:70, :]
    
    plt.figure(figsize=(16,8))
    L1 = X_p.columns.to_list()
    T=np.arange(1000,1000*(len(L1)+1),1000) #
    plt.boxplot(X_p,0,'',positions=T-100,widths=150)
    plt.boxplot(X_c,0,'',positions=T+100,widths=150)
    
    plt.ylim(-0.5,3)
    plt.xticks(T, L1)
    plt.grid()
    plt.xlabel("Features", fontdict={'fontsize':14})
    plt.ylabel("Values", fontdict={'fontsize':14})
    
    L = [Line2D([0], [0], color='k', marker='.', linestyle='None', 
        label='In each feature: Left are values from patients samples & Right are values from healthy persons samples')]
    
    plt.legend(handles=L, loc='best', fontsize=12);
    
    plt.title('Compare patinets and healthy persons values in each feature', fontdict={'fontsize':16})
    plt.show()
    
    return


#Make a plot for all tests for each patient
def plot_each_patient(input_df):
    
    test = []  #Empty list to store all tests belong to each person through while loop
    prob = []  #Empty list to store prediction values for alle tests belong to each person through while loop
    prog = []  #Empty list to store all progression values for each person through while loop
    
    i = 0   #index value in data frame "input_df"

    while i != len(input_df.index):

        test.append(input_df['Test'].iloc[i])
        prob.append(input_df['Prediction'].iloc[i])
        prog.append(input_df['Progression'].iloc[i])

        
        if i != len(input_df.index)-1 :  #This will work for all persons except the last one

            # if it is the same person, pass and continue storing new test and its prediction and progression values
            if input_df['Person'].iloc[i+1] == input_df['Person'].iloc[i] :
                pass

            else:
                
                #Plot the values in the lists: test, prob, prog
                fig, ax = plt.subplots(figsize=(16,6))
                pcolors = ['red' if value == 1 else 'blue' for value in prog]
                plt.plot(test, prob, color='g')
                plt.scatter(test, prob, marker='x', color=pcolors)
                    
                if input_df['Prediction'].max() > 1:
                    ax.set_yticks(np.arange(-5,100.5,5))
                else:
                    ax.set_yticks(np.arange(0,1.1,0.05))
    
                L1 = Line2D([], [], color='blue', marker='x', label='Test without progression', linestyle='None')
                L2 = Line2D([], [], color='red', marker='x', label='Test with progression', linestyle='None')
                ax.legend(handles=[L1, L2], loc='best', fontsize=12);
                
                ax.grid()
                ax.set_title('Patient: ' + str(input_df['Person'].iloc[i]), color='k', fontsize=20)
                plt.show()
                #plt.savefig('./images/' + str(input_df['Person'].iloc[i]) +'.jpg') #To save pircture

                #Empty values in the lists: test, prob, prog for next person
                test =[]
                prob = []
                prog = []


        else:

            #Plot the values in the lists: test, prob, prog for last person in the data frame "input_df"
            fig, ax = plt.subplots(figsize=(16,6))
            pcolors = ['red' if value == 1 else 'blue' for value in prog]
            plt.plot(test, prob, color='g')
            plt.scatter(test, prob, marker='x', color=pcolors)
                
            if input_df['Prediction'].max() > 1:
                ax.set_yticks(np.arange(-5,100.5,5))
            else:
                ax.set_yticks(np.arange(0,1.1,0.05))
            
            L1 = Line2D([], [], color='blue', marker='x', label='Test without progression', linestyle='None')
            L2 = Line2D([], [], color='red', marker='x', label='Test with progression', linestyle='None')
            ax.legend(handles=[L1, L2], loc='best', fontsize=12);
    
            ax.grid()
            ax.set_title('Patient: ' + str(input_df['Person'].iloc[i]), color='k', fontsize=20)
            plt.show()
            #plt.savefig('./images/' + str(input_df['Person'].iloc[i]) +'.jpg') #To save pircture


        i = i + 1   #Move to next index in data frame "input_df"
        
    return       



# Plot for tests with first progression as reference point
def plot_1st_prog_ref(input_df):

    # Calculate how many patients have proression
    n_persons = len( pd.DataFrame( input_df[input_df['Progression'] == 1]['Person'].unique() ) )

    # Create an empty list to resgister patients with progression
    prog_log = []
    
    # Create an empty DataFrame to register all values needed for tests
    # The 7 columns of the data frame will be as follows:
    # ['B01-test', '-3 months', '-2 months', '-1 month', 'Progression', 'Person', 'B01-test place']
    months = pd.DataFrame(index=range(n_persons),columns=range(7))

    i = 0  #First index in "input_df" and starts the counter for while loop
    p = 0  #Patient index in the data frame "months"

    fig, ax = plt.subplots(figsize=(16,12))

    while i != len(input_df.index):


        # The dataframe "input_df" is sorted by person and test number, så it will go ahead with just the first proression
        # prog_log is a list to ensure that the patients with more than one progression will not enter the if statement again  
        if input_df['Progression'].iloc[i] == 1 and input_df['Person'].iloc[i] not in prog_log:

            # Register the patient in the list prog_log
            prog_log.append(input_df['Person'].iloc[i])

            # Register the patient in column with index 5 in the dataframe "months"
            months.iloc[p][5] = input_df['Person'].iloc[i]
            

            # Store the first progression happened with the pasient
            prog_index = i
            
            # Identify B1-test index
            cond = (input_df['Person'] == input_df['Person'].iloc[i]) & (input_df['Test'] == 'B01')
            B01_index = input_df[cond].index

            # Plot and register the first progression in the column with index 4 in the data frame "months"
            plt.scatter(4, input_df['Prediction'].iloc[i], marker='+', color='r')
            months.iloc[p][4] = input_df['Prediction'].iloc[i]

            
            D = 0  #Represtents the total period from first progression until the test we are going to plot and register
            B1_reg = 0  #To confirm that B1-test is plotted and registered in the data frame "months"

            for k in np.arange(3):

                # To ensure to not exceed/move below B1-test index
                if (prog_index - k - 1) >=  B01_index:

                    # D represtents the total period from first progression until the test we are going to plot and register
                    D_new = input_df['Period_rounded'].iloc[prog_index-k] - input_df['Period_rounded'].iloc[prog_index-k-1]
                    D = D + D_new

                    # We are interseted in just the 3 months before the first progression
                    if D < 4:

                        #Note for the following: First progression has index 4 in plotting and also data frame "months", 
                        #so we have to subtract D from 4 to place the test in the correct place (month column)
                        
                        if (prog_index - k - 1) !=  B01_index:
                            
                            # Plot and register tests in the 3 months before first progression in the data frame "months"
                            plt.scatter(4-D, input_df['Prediction'].iloc[prog_index - k - 1], marker='+', color='b')
                            months.iloc[p][4-D] = input_df['Prediction'].iloc[prog_index - k - 1]
                            

                        else:
                            
                            # Plot and register B1-test in the data frame months
                            plt.scatter(4-D, input_df['Prediction'].iloc[prog_index - k - 1], marker='+', color='g')
                            months.iloc[p][4-D] = input_df['Prediction'].iloc[prog_index - k - 1]

                            months.iloc[p][6] = str('-') + str(int(D)) + str(' m') #Tells where B1-test is
                            
                            B1_reg = 1  #To confirm that B1-test is registered in the data frame "months"


            # To ensure that B1-test plotted and registered before moving 
            # to next person after finishing the for loop:
            if B1_reg == 0:
                
                # Plot and register B1-test in the column with index 0 in the data frame months
                plt.scatter(0, np.float64( input_df['Prediction'].iloc[B01_index].values ), marker='+', color='g')
                months.iloc[p][0] = np.float64( input_df['Prediction'].iloc[B01_index].values )

                months.iloc[p][6] = str('B1-test') #Tells where B1-test is

            p = p + 1    #Move to next index/person in data frame "months"

        i = i + 1   #Move to next index in data frame "input_df"

    L = ['B1-test', '-3 months', '-2 months', '-1 month', 'Progression']
    
    ax.set_xticks(np.arange(0,5,1))
    ax.set_xticklabels(L)
    
    #Condition if the dataframe is for regresion model, so change the y-axis values
    if input_df['Prediction'].max() > 1:
        ax.set_yticks(np.arange(-5,100.5,5))
    else:
        ax.set_yticks(np.arange(0,1.1,0.05))
    
    L1 = Line2D([], [], color='green', marker='x', label='B1-test', linestyle='None')
    L2 = Line2D([], [], color='blue', marker='x', label='Test without progression', linestyle='None')
    L3 = Line2D([], [], color='red', marker='x', label='Test with progression', linestyle='None')
    
    ax.legend(handles=[L1, L2, L3], loc='best', fontsize=12);
    ax.set_title('Plot for tests with first progression as reference point', color='k', fontsize=16)
    ax.grid()
    
    plt.show()
    

    ################################################################################################################
    # Make table for months

    months.columns = ['B1-test', '-3 m', '-2 m', '-1 m', 'Progression', 'Person', 'B1-test place']
    months = months[['Person', 'B1-test place', 'B1-test', '-3 m', '-2 m', '-1 m', 'Progression']]
    months


    ################################################################################################################
    # Print boxplot for months:

    plt.figure(figsize=(16,8))
    months_new = months[['B1-test', '-3 m', '-2 m', '-1 m', 'Progression']].apply(pd.to_numeric)
    months_new.columns = ['B1-test', '-3 months', '-2 months', '-1 month', 'Progression']

    months_new.boxplot()
    
    #Condition if the dataframe is for regresion model, so change the y-axis values
    if input_df['Prediction'].max() > 1:
        plt.yticks(np.arange(-5, input_df['Prediction'].max()+5, 5))
    else:
        plt.yticks(np.arange(0,1.1,0.05))
        
    plt.title('Boxplot for plot for tests with first progression as reference point', color='k', fontsize=16)
    plt.show()


    return months


# Wilcoxon Signed-Rank Test for first progression as reference point
def wilcoxon_1st_prog(months):

    list_of_Tests = ['B1-test', '-3 m', '-2 m', '-1 m', 'Progression']
    Wilcoxon_df = pd.DataFrame(index=range(5),columns=range(5))

    for i in np.arange(5):

        for j in np.arange(5):
            if i != j:
                g1 = months[ ( months[list_of_Tests[i]].notnull() & months[list_of_Tests[j]].notnull() ) ][list_of_Tests[i]]
                g2 = months[ ( months[list_of_Tests[i]].notnull() & months[list_of_Tests[j]].notnull() ) ][list_of_Tests[j]]
                Wilcoxon_df.iloc[i][j] = stats.wilcoxon(g1, g2)[1]

    Wilcoxon_df.columns = ['B1-test', '-3 months', '-2 months', '-1 month', 'Progression']
    Wilcoxon_df.index = ['B1-test', '-3 months', '-2 months', '-1 month', 'Progression']


    ################################################################################################################
    # Collect all B1-tests in a group to test againts progression
    B1tests_list = []
    
    for i in np.arange(months.index.size): #n_persons
        B1tests_list.append(months[ months['B1-test place'].iloc[i] ].iloc[i])

    B1tests_S = pd.Series(B1tests_list)


    # Wilcoxon Signed-Rank Test for total B1-test tests vs Progression group
    g1 = B1tests_S
    g2 = months['Progression']
    Wilcoxon_B1tests_Prog = stats.wilcoxon(g1, g2)

    print('Total B1-tests vs Progression: ', Wilcoxon_B1tests_Prog[1] )

    # Change value of B1-test/Progression in the previous table
    Wilcoxon_df.iloc[0][4] = Wilcoxon_df.iloc[4][0] = Wilcoxon_B1tests_Prog[1]
    
    Wilcoxon_df[:] = np.triu(Wilcoxon_df.values)
    Wilcoxon_df = Wilcoxon_df.replace(np.nan, '', regex=True)
    Wilcoxon_df = Wilcoxon_df.replace(0, '', regex=True)
    
    Wilcoxon_df = Wilcoxon_df.drop(['B1-test'], axis=1)
    Wilcoxon_df = Wilcoxon_df.drop(['Progression'], axis=0)
    
    return Wilcoxon_df



# Plot for tests with first test (B1 test) as reference point
def plot_B1_test_ref(input_df):
    
    # Calculate how many patients have proression
    n_persons = len( pd.DataFrame( input_df[input_df['Progression'] == 1]['Person'].unique() ) )

    # Create an empty list to resgister patients with progression
    prog_log = []
    
    # Create an empty DataFrame to register all values needed for tests
    # The 7 columns of the data frame will be as follows:
    # ['B01-test', '+1 month', '+2 months', '+3 months', 'Progression', 'Person', 'Progression place']
    months = pd.DataFrame(index=range(n_persons),columns=range(7))

    i = 0  #First index in "input_df" and starts the counter for while loop
    p = 0  #Patient index in the data frame "months"

    fig, ax = plt.subplots(figsize=(16,12))

    while i != len(input_df.index):


        # The dataframe "input_df" is sorted by person and test number, so it will go ahead with just the first proression
        # prog_log is a list to ensure that the patients with more than one progression will not enter the if statement again  
        if input_df['Progression'].iloc[i] == 1 and input_df['Person'].iloc[i] not in prog_log:
            
            # Register the patient in the list prog_log
            prog_log.append(input_df['Person'].iloc[i])

            # Register the patient in column with index 5 in the dataframe "months"
            months.iloc[p][5] = input_df['Person'].iloc[i]

            # Store the first progression happened with the pasient
            prog_index = i
            
            # Identify B1-test index
            cond = (input_df['Person'] == input_df['Person'].iloc[i]) & (input_df['Test'] == 'B01')
            B01_index = input_df[cond].index
            
            # Plot and register B01 test in the data frame "months"
            plt.scatter(0, np.float64(input_df['Prediction'].iloc[B01_index].values) , marker='x', color='g')
            months.iloc[p][0] = np.float64(input_df['Prediction'].iloc[B01_index].values)

            
            D = 0  #Represtents the total period from B1 test until the test we are going to plot and register
            prog_reg = 0  #To confirm that the first progression is plotted and registered in the data frame "months"

            for k in np.arange(3):

                # To ensure to not exceed the first prog index, and to not exceed the total length of the dataframe "input_df"
                if (B01_index + k + 1) <=  prog_index and (B01_index + k + 1) < len(input_df.index):

                    # D represtents the total period from B1 test until the test we are going to plot and register
                    D_new = input_df['Period_rounded'].iloc[B01_index+k+1].values - input_df['Period_rounded'].iloc[B01_index+k].values
                    D = D + D_new

                    # We are interseted in just first 3 months
                    if D < 4:

                        if (B01_index + k + 1) !=  prog_index:
                            
                            # Plot and register tests in the 3 months after B1-test in the data frame "months"
                            plt.scatter(D, input_df['Prediction'].iloc[B01_index + k + 1], marker='+', color='b')
                            months.iloc[p][D] = input_df['Prediction'].iloc[B01_index + k + 1]

                        else:
                            
                            # Plot and register the first progression in the data frame "months"
                            plt.scatter(D, input_df['Prediction'].iloc[B01_index + k + 1], marker='+', color='r')
                            months.iloc[p][D] = input_df['Prediction'].iloc[B01_index + k + 1]

                            months.iloc[p][6] = str('+') + str(int(D)) + str(' m') #Tells where progession is
                            
                            prog_reg = 1  #To confirm that the progression is registered in the data frame "months"


            # To ensure that the first progression plotted and registered before moving 
            # to next person after finishing the for loop:
            if prog_reg == 0:
                
                # Plot and register the first progression in the column with index 4 in the data frame "months"
                plt.scatter(4, input_df['Prediction'].iloc[prog_index], marker='+', color='r')
                months.iloc[p][4] = input_df['Prediction'].iloc[prog_index]

                months.iloc[p][6] = str('Progression') #Tells where progession is



            p = p + 1   #Move to next index/person in data frame "months"

        i = i + 1     #Move to next index in data frame "input_df"

    L = ['B1-test', '+1 month', '+2 months', '+3 months', 'Progression']
    
    ax.set_xticks(np.arange(0,5,1))
    ax.set_xticklabels(L)
    
    #Condition if the dataframe is for regresion model, so change the y-axis values
    if input_df['Prediction'].max() > 1:
        ax.set_yticks(np.arange(-5,100.5,5))
    else:
        ax.set_yticks(np.arange(0,1.1,0.05))
        
    L1 = Line2D([], [], color='green', marker='x', label='B1-test', linestyle='None')
    L2 = Line2D([], [], color='blue', marker='x', label='Test without progression', linestyle='None')
    L3 = Line2D([], [], color='red', marker='x', label='Test with progression', linestyle='None')
    
    ax.legend(handles=[L1, L2, L3], loc='best', fontsize=12);
    ax.set_title('Plot for tests with first test (B1-test) as reference point', color='k', fontsize=16)
    ax.grid()
    
    plt.show()

    ################################################################################################################
    # Assign names for columns in the data frame months
    months.columns = ['B1-test', '+1 m', '+2 m', '+3 m', 'Progression', 'Person', 'Progression place']
    
    # Rearrange the columns in the data frame months
    months = months[['Person', 'B1-test', '+1 m', '+2 m', '+3 m', 'Progression', 'Progression place']]

    ################################################################################################################
    # Plot boxplot for the data frame months:
    
    plt.figure(figsize=(16,8))
    months_new = months[['B1-test', '+1 m', '+2 m', '+3 m', 'Progression']].apply(pd.to_numeric)
    months_new.columns = ['B1-test', '+1 month', '+2 months', '+3 months', 'Progression']

    months_new.boxplot()
    
    #Condition if the dataframe is for regresion model, so change the y-axis values
    if input_df['Prediction'].max() > 1:
        plt.yticks(np.arange(-5, input_df['Prediction'].max()+5, 5))
    else:
        plt.yticks(np.arange(0,1.1,0.05))
    
    plt.title('Boxplot for plot for tests with first test (B1-test) as reference point', color='k', fontsize=16)
    plt.show()
    

    return months


# Wilcoxon Signed-Rank Test for B1 test as reference point
def wilcoxon_B1_test(months):
    
    list_of_Tests = ['B1-test', '+1 m', '+2 m', '+3 m', 'Progression']
    Wilcoxon_df = pd.DataFrame(index=range(5),columns=range(5))

    for i in np.arange(5):

        for j in np.arange(5):
            if i != j:
                g1 = months[ ( months[list_of_Tests[i]].notnull() & months[list_of_Tests[j]].notnull() ) ][list_of_Tests[i]]
                g2 = months[ ( months[list_of_Tests[i]].notnull() & months[list_of_Tests[j]].notnull() ) ][list_of_Tests[j]]
                Wilcoxon_df.iloc[i][j] = stats.wilcoxon(g1, g2)[1]

    Wilcoxon_df.columns = ['B1-test', '+1 month', '+2 months', '+3 months', 'Progression']
    Wilcoxon_df.index = ['B1-test', '+1 month', '+2 months', '+3 months', 'Progression']


    ################################################################################################################
    # Collect all progression in a group to test againts B01 tests
    Prog_list = []
    
    for i in np.arange(months.index.size): #n_persons
        Prog_list.append(months[ months['Progression place'].iloc[i] ].iloc[i])

    Prog_S = pd.Series(Prog_list)

    # Wilcoxon Signed-Rank Test for total B01 tests vs Progression group
    g1 = months['B1-test']
    g2 = Prog_S
    Wilcoxon_B1tests_Prog = stats.wilcoxon(g1, g2)

    print('B1-tests vs Total Progression: ', Wilcoxon_B1tests_Prog[1] )

    # Change value of B01/Progression in the previous table
    Wilcoxon_df.iloc[0][4] = Wilcoxon_df.iloc[4][0] = Wilcoxon_B1tests_Prog[1]
    
    Wilcoxon_df[:] = np.triu(Wilcoxon_df.values)
    Wilcoxon_df = Wilcoxon_df.replace(np.nan, '', regex=True)
    Wilcoxon_df = Wilcoxon_df.replace(0, '', regex=True)
    
    Wilcoxon_df = Wilcoxon_df.drop(['B1-test'], axis=1)
    Wilcoxon_df = Wilcoxon_df.drop(['Progression'], axis=0)
    
    return Wilcoxon_df



# This function is to implement leave one out technique
def Leave_one_out(X_data, y_data, clf):
    
    sample_idx  = []   #Empty list to contain indices for samples to be predicted through cross validation
    y_test      = []   #Empty list to contain true values for samples through cross validation
    y_pro       = []   #Empty list to contain the probability of predicted values for samples through cross validation

    test_scores = []  #Accumulate score through all cross validation loops
    
    
    # Since the number of KFold splits are equal to total number of samples, so it will perform a leave one out cross validation
    cv = KFold(n_splits=70, shuffle=True, random_state=0)

    
    for train_idx, test_idx in cv.split(X_data, y_data):

        clf.fit(X_data.iloc[train_idx], y_data.iloc[train_idx].values.ravel())

        # Score on test fold (test_idx)
        test_scores.append(clf.score(X_data.iloc[test_idx], y_data.iloc[test_idx].values.ravel()))

        print('Sample to be predicted is : ',  test_idx, 'and accuracy is: %.2f%%' % (test_scores[-1]*100) )

        sample_idx.extend( test_idx )
        y_test.extend( y_data.iloc[test_idx].values.ravel() )
        y_pro.extend( clf.predict_proba(X_data.iloc[test_idx])[:,1] )


    print('\nTotal accuracy %.2f%% +/- %.2f' % (np.mean(test_scores) * 100, np.std(test_scores) * 100))

    plot_df = pd.DataFrame(zip(sample_idx, y_pro), columns=['Person', 'Prediction'])
    
    
    ############################################################################################################
    #Plot patients and normal persons on two gorups
    
    plot_df = plot_df.sort_values(by='Person', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(28,12))

    for i in np.arange(30):
        ax.scatter(1, plot_df['Prediction'].iloc[i], marker='x', color='b')

    for i in np.arange(30,70,1):
        ax.scatter(2, plot_df['Prediction'].iloc[i], marker='x', color='r')

    ax.set_yticks(np.arange(0,1.1,0.05))
    ax.set_autoscalex_on(False)
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['','Healthy','Pasient',''], fontsize=12);
    ax.grid()
    ax.set_title('Patients and healthy persons', color='k', fontsize=20)
    ax.set_ylabel('Probability', fontsize=16);
    
    
    ############################################################################################################
    #Plot probability function for patients and normal persons
    
    plot_df = plot_df.sort_values(by='Prediction', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(28,8))

    ax.set_autoscalex_on(False)
    pcolors = ['blue' if value > 39 else 'red' for value in plot_df['Person']]
    ax.scatter(plot_df.index, plot_df['Prediction'].iloc[0:70], marker='x', color=pcolors)

    ax.set_yticks(np.arange(0,1.1,0.05))

    ax.set_xticks(plot_df.index)
    ax.set_xticklabels(plot_df['Person'].values, fontsize=12);
    ax.grid()

    ax.set_title('Probability function', color='k', fontsize=20)
    ax.set_xlabel('Baseline tests', fontsize=16);
    ax.set_ylabel('Probability', fontsize=16);
    ax.legend(['Blue is Healthy & Red is Patient'], fontsize=16)
    
    return



def select_features_clf(X_data, y_data, Met = 'Lasso', LF = 'No'):
    
#Features selection function

### Arguments for this function ###
# X_data: Input variables
# y_data: Target variable
# Met   : Method for feature selection
# LF    : Just for Lasso, if someone would like to get feature coefficients for each feature selection loop using Lasso

        
    clf_list = ['LR ', 'SVC', 'KNN', 'RF ', 'NB '] #'QDA', 
    
    pipe = {}
    param_grid = {}
    clf_gs = {}
    
    LR = LogisticRegression(random_state=0)
    pipe[0] = Pipeline([('std', StandardScaler()), ('LR', LR)])
    param_grid[0] = [{"LR__C":np.logspace(-2,2,num=5), "LR__penalty":["l2"], "LR__solver":['liblinear','newton-cg']}]
    
    SvC = SVC(random_state=0)
    pipe[1] = Pipeline([('std', StandardScaler()), ('SvC', SvC)])
    param_grid[1] = [{'SvC__kernel': ['linear'], 'SvC__C': np.logspace(-2,2,num=5)}]

    KNN = KNeighborsClassifier()
    pipe[2] = Pipeline([('std', StandardScaler()), ('KNN', KNN)])
    param_grid[2] = [{'KNN__n_neighbors': list(range(1, 10))}]
    
    RF = RandomForestClassifier(random_state=0)
    pipe[3] = Pipeline([('std', StandardScaler()), ('RF', RF)])
    param_grid[3] =  {'RF__max_depth': [10, 20, 30, 40, 50], 'RF__n_estimators': [10, 50, 100]}
    
    NB = GaussianNB()
    pipe[4] = Pipeline([('std', StandardScaler()), ('NB', NB)])
    param_grid[4] = [{'NB__var_smoothing': np.logspace(0,-9, num=10)}]

    
    ##############################################################################
    ### Mutual Information test before Forward and Backward feature selection ####
    selector = SelectKBest(mutual_info_classif, k=70)
    selector.fit(X_data, y_data.values.ravel())
    
    feature_names = X_data.columns
    new_features = feature_names[selector.get_support()]            

    X_data_ftest = X_data[new_features]
    ##############################################################################

    F_list = []    #To store the selected features in each outer cv loop
    ListOpt = [[] for k in range(6)]  #To store all hyperparameters chosen in the inner cv loops
    top_param = {}    #To store just the best top hyperparameters for each classifier


    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    
    for i in np.arange(5):
        clf_gs[i] = GridSearchCV(estimator=pipe[i],param_grid=param_grid[i], 
                    cv=inner_cv, scoring='accuracy', verbose=0, refit=True)
                    #cv=inner_cv,scoring='accuracy',n_jobs=-1,verbose=0,refit=True)


    outer_cv_number = 10
    outer_scores = [[] for k in range(5)]   #To store accuracy scores and give general evaluation for all classifiers in nested cv
    outer_cv = StratifiedKFold(n_splits=outer_cv_number, shuffle=True, random_state=0)

    Loop = 1    #Outer cv loop counter

    Lasso_features_df = pd.DataFrame() #Empty dataframe for just Lasso fearues

    for train_idx, test_idx in outer_cv.split(X_data, y_data):
        
        print('Cross validation loop number ' + str(Loop) + ' out of ' + str(outer_cv_number) + ' loops is started')

        if Met == 'Forward Selection':

            ### Forward Selection ###            
            clf_sfs = SequentialFeatureSelector(estimator=LogisticRegression(), n_features_to_select=15)
            clf_sfs.fit(X_data_ftest.iloc[train_idx], y_data.iloc[train_idx].values.ravel())

            feature_names = X_data_ftest.columns
            new_features = feature_names[clf_sfs.get_support()]            
            F_list.extend(new_features)

            X_new_sfs = X_data_ftest[new_features]
            X_data_new = X_new_sfs


        elif Met == 'Backward Selection':

            ## Backward Selection ###            
            clf_sbs = SequentialFeatureSelector(estimator=LogisticRegression(), direction='backward', n_features_to_select=15)
            clf_sbs.fit(X_data_ftest.iloc[train_idx], y_data.iloc[train_idx].values.ravel())

            feature_names = X_data_ftest.columns
            new_features = feature_names[clf_sbs.get_support()]            
            F_list.extend(new_features)

            X_new_sbs = X_data_ftest[new_features]
            X_data_new = X_new_sbs


        elif Met == 'Tree-based':

            ## Tree-based ###
            clf_tree = RandomForestClassifier()
            clf_tree.fit(X_data.iloc[train_idx], y_data.iloc[train_idx].values.ravel())
            model = SelectFromModel(clf_tree, max_features=15, prefit=True)

            feature_names = X_data.columns
            new_features = feature_names[model.get_support()]            
            F_list.extend(new_features)

            X_new_tree = X_data[new_features]
            X_data_new = X_new_tree


        elif Met == 'Lasso':

            ## Lasso ###
            clf_lasso = LogisticRegression(penalty='l1', solver='liblinear')
            clf_lasso.fit(X_data.iloc[train_idx], y_data.iloc[train_idx].values.ravel())
            model = SelectFromModel(clf_lasso, prefit=True)

            feature_names = X_data.columns
            new_features = feature_names[model.get_support()]            
            F_list.extend(new_features)

            X_new_lasso = X_data[new_features]
            X_data_new = X_new_lasso


            ##############################################################################################
            # Lasso_features_df: is a data frame created to have the chosen features and their importances 
            # (coeffecients) in each outer CV loop
            
            Lasso_features = pd.DataFrame({'Feature':X_data.columns,'Coeff':clf_lasso.coef_.ravel()})
            Lasso_features = Lasso_features.sort_values(by='Coeff', ascending=False).reset_index(drop=True)
            Lasso_features = Lasso_features[Lasso_features['Coeff'] != 0].reset_index(drop=True)
            Lasso_features_df = pd.concat([Lasso_features_df, Lasso_features])

            ###############################################################################################


        else:
            print('Please enter correct feature selection method')
            return



        ### Run inner loop to select the best hyperparameters for classifiers ###        
        for i in np.arange(5):
            clf_gs[i].fit(X_data_new.iloc[train_idx], y_data.iloc[train_idx].values.ravel())
            ListOpt[i].append(clf_gs[i].best_params_)


        # Score on test fold (test_idx)
        for i in np.arange(5):
            outer_scores[i].append(clf_gs[i].best_estimator_.score(X_data_new.iloc[test_idx], y_data.iloc[test_idx].values.ravel()))
            print('Accuracy on the outer test fold for ' + clf_list[i] + ' : %.2f%%' % (outer_scores[i][-1]*100))
            

        print('Cross validation loop number ' + str(Loop) + ' out of ' + str(outer_cv_number) + ' loops is finished\n')
        Loop = Loop + 1


    results = pd.DataFrame(outer_scores)

    df = pd.DataFrame(clf_list, columns=['Clf'])
    df['ACC %'] = np.round(results.mean(axis=1)*100, 2)
    df['STD %'] = np.round(results.std(axis=1)*100, 2)
    print('\nThe classifiers score after cross validation is:\n')
    print(df)
    print('\n')


    # Print selected hyperparameters for all classifiers
    for k in np.arange(5):
        S1 = pd.Series(ListOpt[k])
        S2 = pd.Series(S1.value_counts())
        print('\n Best hyperparameters for ' + clf_list[k] + ':\n')
        print(S2)
        top_param[k] = S2.index[0]   #Here the top hyperparameters for each classifier are chosen
    

    # Print selected features
    S = pd.Series(F_list)
    N = ( S.value_counts() / outer_cv_number ) * 100  #To find appearance precentage
    features_df = pd.DataFrame(N, columns = ['Appearance %'])
    features_df.reset_index(inplace=True)
    features_df = features_df.rename(columns = {'index':'Feature'})
    print('\n Best selected features: \n')
    print( features_df.head(50) )
    
    #This last part is just for Lasso
    if LF == 'Yes' and Met == 'Lasso':
        #print('\n', Lasso_features_df)
        Lasso_features_df.to_csv('./results/Lasso_features_df') # Exported with indices to seperate features in each outer cv loop
    
    return features_df, top_param



# Hyperparameters optimization for classifiers
def clf_h_p_opt(X_data, y_data):
    
    clf_list = ['LR', 'SVC', 'KNN', 'RF', 'NB'] #'QDA', 
    
    pipe = {}
    param_grid = {}
    clf_gs = {}  #Create an empty dictionry for grid search cross validation
    best_param = {}  #Create an empty dictionary for best parameters

    LR = LogisticRegression(random_state=0)
    pipe[0] = Pipeline([('std', StandardScaler()), ('LR', LR)])
    param_grid[0] = [{"LR__C":np.logspace(-2,2,num=5), "LR__penalty":["l2"], "LR__solver":['liblinear','newton-cg']}]

    SvC = SVC(random_state=0)
    pipe[1] = Pipeline([('std', StandardScaler()), ('SvC', SvC)])
    param_grid[1] = [{'SvC__kernel': ['linear'], 'SvC__C': np.logspace(-2,2,num=5)}]

    KNN = KNeighborsClassifier()
    pipe[2] = Pipeline([('std', StandardScaler()), ('KNN', KNN)])
    param_grid[2] = [{'KNN__n_neighbors': list(range(1, 10))}]


    RF = RandomForestClassifier(random_state=0)
    pipe[3] = Pipeline([('std', StandardScaler()), ('RF', RF)])
    param_grid[3] =  {'RF__max_depth': [10, 20, 30, 40, 50], 'RF__n_estimators': [10, 50, 100]}


    NB = GaussianNB()
    pipe[4] = Pipeline([('std', StandardScaler()), ('NB', NB)])
    param_grid[4] = [{'NB__var_smoothing': np.logspace(0,-9, num=10)}]
    
    
    # Here to find the best Hyperparameters for each clssifier by Grid Search cross validation
    print('\nBest parameters for all classifiers:\n')
    
    for i in np.arange(5):
        clf_gs[i] = GridSearchCV(estimator=pipe[i], param_grid=param_grid[i], cv=10, scoring='accuracy')
        clf_gs[i].fit(X_data, y_data.values.ravel())
        print(clf_list[i], ':')
        print(clf_gs[i].best_params_, '\n')
        best_param[i] = clf_gs[i].best_params_

    
    return best_param



# Classifiers Evaluation by cross Validation (10 folds)
def test_classifiers(X_data, y_data, best_param, ROC_AUC = 'None'):
    
    clf_list = ['LR', 'SVC', 'KNN', 'RF', 'NB'] #'QDA', 
    pipe = {}    
    
    #LogisticRegresion Classifier:
    LR = LogisticRegression(random_state=0)
    pipe[0] = Pipeline([('std', StandardScaler()), ('LR', LR)])
    pipe[0].set_params(**best_param[0])

    #SVC Classifier: 
    SvC = SVC(random_state=0)
    pipe[1] = Pipeline([('std', StandardScaler()), ('SvC', SvC)])
    pipe[1].set_params(**best_param[1])

    #KNN Classifier:
    KNN = KNeighborsClassifier()
    pipe[2] = Pipeline([('std', StandardScaler()), ('KNN', KNN)])
    pipe[2].set_params(**best_param[2])

    #Random Forest Classifier:
    RF = RandomForestClassifier(random_state=0)
    pipe[3] = Pipeline([('std', StandardScaler()), ('RF', RF)])
    pipe[3].set_params(**best_param[3])

    #Naive Bayes Classifier:
    NB = GaussianNB()
    pipe[4] = Pipeline([('std', StandardScaler()), ('NB', NB)])
    pipe[4].set_params(**best_param[4])
    
    
    
    Results = np.zeros((4,5))
    
    List_AUC = [[] for i in range(5)]
    List_y_test = []
        
    acc_df = pd.DataFrame()   #Empty data frame for accuracy
    rec_df = pd.DataFrame()   #Empty data frame for recall
    pre_df = pd.DataFrame()   #Empty data frame for precition
    f1s_df = pd.DataFrame()   #Empty data frame for f1_score
    
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_idx, valid_idx in cv.split(X_data, y_data):
  
        X_train = X_data.iloc[train_idx]
        y_train = y_data.iloc[train_idx].values.ravel()
        X_test = X_data.iloc[valid_idx]
        y_test = y_data.iloc[valid_idx].values.ravel()
        
        
        List_y_test.extend(y_test)
        
        
        for k in np.arange(5):
            
            pipe[k].fit(X_train, y_train)
            
            y_pred = pipe[k].predict(X_test)
            
            #It is better to use "decision_function" than "predict_proba" for logistic Regrassion and SVC to create AUC plot
            if k < 2:
                y_pro = pipe[k].decision_function(X_test)
                List_AUC[k].extend(y_pro)
            else:
                y_pro = pipe[k].predict_proba(X_test)
                List_AUC[k].extend(y_pro[:, 1])
            
            Results[0,k] = np.round(accuracy_score(y_pred, y_test), 2)
            Results[1,k] = np.round(recall_score(y_pred, y_test), 2)
            Results[2,k] = np.round(precision_score(y_pred, y_test), 2)
            Results[3,k] = np.round(f1_score(y_pred, y_test), 2)
            
                    
        
        ### RESULTS ###
        Results_df = pd.DataFrame(Results, columns=clf_list, index = ('Accuracy', 'Recall', 'Precision', 'F1_score'))
        
        acc_df = pd.concat([acc_df, Results_df.iloc[0]], axis=1, join='outer')
        rec_df = pd.concat([acc_df, Results_df.iloc[1]], axis=1, join='outer')
        pre_df = pd.concat([acc_df, Results_df.iloc[2]], axis=1, join='outer')
        f1s_df = pd.concat([acc_df, Results_df.iloc[3]], axis=1, join='outer')
                        
        
    total_df = pd.concat([
                        np.round(acc_df.T.mean(axis=0)*100, 2), np.round(acc_df.T.std(axis=0)*100, 2),
                        np.round(rec_df.T.mean(axis=0)*100, 2), np.round(rec_df.T.std(axis=0)*100, 2),
                        np.round(pre_df.T.mean(axis=0)*100, 2), np.round(pre_df.T.std(axis=0)*100, 2),
                        np.round(f1s_df.T.mean(axis=0)*100, 2), np.round(f1s_df.T.std(axis=0)*100, 2)
                        ], axis=1)
    
    total_df.columns = ['Accuracy %', 'STD %', 'Recall %', 'STD %', 'Precision %', 'STD %', 'F1_score %', 'STD %']
    
    
    
    #Plot ROC_AUC
    if ROC_AUC == 'ROC':
        
        plt.figure(figsize=(6, 6), dpi=150)
        
        for k in np.arange(5):

            clf_fpr, clf_tpr, _ = roc_curve(List_y_test, List_AUC[k])
            clf_auc = auc(clf_fpr, clf_tpr)

            plt.plot(clf_fpr, clf_tpr, label = clf_list[k] + ' (AUC = %0.3f)' % clf_auc)

        plt.plot([0,1], [0,1], linestyle='--', color='k')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')

        plt.legend()

        plt.show()
    
    
    return total_df



# Feature selection will be peformed by selecting features from 3 models (LassoCV, RidgeCV, ElasticNetCV)
# These methods use cross validation in selecting features and choose best model. CV folds are set to 10
def feature_selection_reg(X, y, Met = 'LassoCV'):
    
    if Met not in ['LassoCV', 'RidgeCV', 'ElasticNetCV']:
        print('Please, choose available feature selection method: \nLassoCV or RidgeCV or ElasticNetCV')
        return
    
    # Scale the features before feature selection
    scaler = StandardScaler()
    X_data = scaler.fit_transform(X)
    
    model = {}
    # Create regression object
    model['LassoCV'] = linear_model.LassoCV(cv=10)
    model['RidgeCV'] = linear_model.RidgeCV(cv=10)
    model['ElasticNetCV'] = linear_model.ElasticNetCV(cv=10)

    # Train the model using the training sets
    model[Met].fit(X_data, y.values.ravel())

    # Make predictions using the testing set
    y_pred = model[Met].predict(X_data)

    # Evaluate feature selection method
    MSE = mean_squared_error(y, y_pred)
    R2score = r2_score(y, y_pred)

    print('\nEvaluation of feature selection method by ' + str(Met) + ' :')
    print("Mean squared error: %.2f" % np.mean(MSE))
    print("Coefficient of determination: %.2f" % np.mean(R2score))

    # Find and sort features by their importances (coeffecients)
    coeff = np.abs(model[Met].coef_.ravel())
    Best_features = pd.DataFrame({'Feature':X.columns,'Coeff':coeff})
    Best_features = Best_features.sort_values(by='Coeff', ascending=False).reset_index(drop=True)
    Best_features = Best_features[Best_features['Coeff'] != 0].reset_index(drop=True)
    
    #To export the best features:
    #Best_features.to_csv('./results/Regression_Best_features_' + Met + '.csv', mode = 'w')
    
    print('\n', Best_features.head(50))
    
    return Best_features



# The following function to find best parameters for Random Forest Regressor by using the selected features
def RF_hyp_opt(X_new_chosen, y):
    
    RF = RandomForestRegressor(random_state=0)
    pipe = Pipeline([('scaler', StandardScaler()), ('RF', RF)])

    cv=KFold(n_splits=5, shuffle=True, random_state=0)

    param_grid = [{'RF__n_estimators': range(5,50,5),'RF__max_depth': range(5,50,5)}]

    clf = GridSearchCV(estimator = pipe, param_grid = param_grid, cv = cv, n_jobs=-1, scoring='r2')

    clf.fit(X_new_chosen, y.values.ravel())

    return clf.best_params_



# Regressors Evaluation by 10-fold cross Validation
def test_regressors(X_data, y_data, param):
     
    MSE = [[] for i in range(4)]
    R2score = [[] for i in range(4)]
    Results = [[] for i in range(4)]

    clf_list = ['RF', 'LassoCV', 'RidgeCV', 'ElasticNetCV']
    pipe = {}
    
    
    # Create regression objects & pipelines
    RF = RandomForestRegressor(random_state=0)
    pipe[clf_list[0]] = Pipeline([('scaler', StandardScaler()), ('RF', RF)])
    pipe[clf_list[0]].set_params(**param)

    L_CV = linear_model.LassoCV(max_iter=10000)
    pipe[clf_list[1]] = Pipeline([('scaler', StandardScaler()), ('LassoCV', L_CV)])

    R_CV = linear_model.RidgeCV(cv=5)
    pipe[clf_list[2]] = Pipeline([('scaler', StandardScaler()), ('RidgeCV', R_CV)])

    E_CV = linear_model.ElasticNetCV(max_iter=10000)
    pipe[clf_list[3]] = Pipeline([('scaler', StandardScaler()), ('ElasticNetCV', E_CV)])
            
    
    cv = KFold(n_splits=10, shuffle=True, random_state=0)

    for train_idx, test_idx in cv.split(X_data, y_data):
  
        X_train = X_data.iloc[train_idx]
        y_train = y_data.iloc[train_idx].values.ravel()
        X_test = X_data.iloc[test_idx]
        y_test = y_data.iloc[test_idx].values.ravel()
        

        for k, name in enumerate(clf_list):

            # Train the model using the training sets
            pipe[name].fit(X_train, y_train)

            # Make predictions using the testing set
            y_pred = pipe[name].predict(X_test)

            # Evaluation of regresion models
            MSE[k].append(mean_squared_error(y_test, y_pred))
            R2score[k].append(r2_score(y_test, y_pred))


    for k in np.arange(4):
        Results[k] = np.round([np.mean(MSE[k]), np.std(MSE[k]), np.mean(R2score[k]), np.std(R2score[k])],2)

    Results_df = pd.DataFrame(Results)
    Results_df.columns=['MSE', 'MSE STD', 'R2score', 'R2score STD']
    Results_df.index=clf_list

    print(Results_df)
    print("\nDetails:")
    print("MSE: Mean squared error")
    print("MSE STD: Standard deviation for mean squared error")
    print("R2score: Coefficient of determination")
    print("R2score STD: Standard deviation for coefficient of determination")

    return Results_df