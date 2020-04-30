# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import necessary packages
import os
import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
    
# Import the logging module.
import logging
    
# Import the sys module & suppress tracebacks.
import sys
sys.tracebacklimit = 0
    
# Change jupyter functionality to display all output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
    
# Import modules from sklearn
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn import ensemble
from sklearn.model_selection import (RandomizedSearchCV)
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.compose import ColumnTransformer
    
# Create and configure logger.
logging.basicConfig(format='%(asctime)s %(message)',
                        filemode='w')
    
# Create an object.
logger=logging.getLogger()
    
# Set the logger threshold.
logger.setLevel(logging.WARN)

# Define the import function
def import_data():
    """Checks the user-entered file path validity, imports the data, and returns a merged data frame.
    
    Returns
        An imported data set as a numpy array
    Raises
        FormatError: In case `path1 is not a valid file path`"""
    
    print("Please enter the path for the local folder that contains csv-format TEDS-A datasets you wish to import and merge.")
    print("Hit enter to submit (all .csv files in the folder will be imported); example: C:/users/name/desktop/datafolder")
    path1 = input()
    
    if not os.path.exists(path1):
        return logger.error("Error: the entered text is not a valid file path.")
    # The file path is not valid
    else:
        # The file path is valid, all is well: proceed
        read_files = glob.glob(os.path.join(path1, "*.csv"))
        np_array_values = []
        for files in read_files:
            # Imports the data
            TEDS_A = pd.read_csv(files)
            np_array_values.append(TEDS_A)
        merge_values = np.vstack(np_array_values)
        TEDS_A_merged = pd.DataFrame(merge_values)
        col_list = list(TEDS_A.columns)
        TEDS_A_merged.columns = col_list
        TEDS_A_merged[col_list] = TEDS_A_merged[col_list].astype('category')
    return(TEDS_A_merged)

# Create a function for exploratory data analysis
def EDA(TEDS_A_merged):
    """Generates basic exploratory data analsis aides based on user input.
    
    Returns
        Tables and plots."""
    
    # Query the user for the variables of interest
    print("You now have the option to explore some descritive info and associations.")
    print("Please enter the Y (dependent) and X(s) of interest (a smaller number will be easier to examine).")
    print("Enter the column/variable names separated by a space and not a comma, example: EDUC MARSTAT\n")
    print("Enter the Y variable\n")
    user_target = input()
    print("\nEnter the X variable(s)\n")
    user_X = input()
    user_X_list = list(user_X.split(" "))
    user_XY_list = list(user_X.split(" "))
    user_XY_list.append(user_target)
    
    # Create a temporary dataframe without NAs
    TEDS_Ana = TEDS_A_merged[user_XY_list]
    TEDS_Ana = TEDS_Ana.replace(-9, np.nan)
    TEDS_Ana = TEDS_Ana.dropna()
    col_list2 = list(TEDS_Ana.columns)
    TEDS_Ana[col_list2] = TEDS_Ana[col_list2].astype('category')
    
    # Get some descriptive info for the selected variables
    info1 = TEDS_Ana[user_target].describe()
    info2 = TEDS_Ana[user_X_list].describe()
    # Cleanly print the info
    pd.DataFrame(info1)
    pd.DataFrame(info2)
    
    return(user_XY_list, info1, info2, TEDS_Ana, user_target, user_X_list)

# Create a function to generate crosstabs
def crosstabs(data_frame, id_col, col_names):
    """Generates crosstabs based on the variables selected by the user.
    
    Returns
        A series of crosstabs in dictionary format."""
    
    crosstabs_out = {}
    for i in col_names:
        crosstabs_out['crosstabs_out_{}'.format(i)] = pd.crosstab(data_frame[id_col], data_frame[i])
    return (crosstabs_out)

# Create a function to generate visual crosstabs
def vis_crosstabs(data_frame, id_col, col_names):
    """Generates crosstabs based on the variables selected by the user.
    
    Returns
        A series of crosstabs in dictionary format."""
    
    for i in col_names:
        sns.heatmap(pd.crosstab([data_frame[id_col]], [data_frame[i]]), cmap="YlGnBu")
        plt.show()
    return()

def drop_data(TEDS_A_merged):
    """Queries the user for variables they wish to exclude.
    
    Returns
        A truncated data frame."""
        
    print("""You now have the option to keep specific variables from the data (type 'NA' to skip). 
Please use the TEDS-A codebook or the list generated to enter the names of variables you wish to keep.) 
Enter the column/variable names separated by a space and not a comma, example: CASEID EDUC MARSTAT\n""")
    
    print("Currently available columns: ", list(TEDS_A_merged.columns))
    print("\n")
    user_list = input()
        
    if user_list == "NA" or user_list == "na":
        return(print("No columns will be dropped."))
    else:
        keep_list = user_list.split(" ")
        TEDS_A = TEDS_A_merged.filter(keep_list)
        TEDS_A.replace(-9, np.nan, inplace=True)
    return(TEDS_A)

def imputation(data_frame):
    """Imputes missing data using sklearn's iterative imputer.
    
    Returns
        A data frame with imputed values."""
    
    imputer = IterativeImputer(max_iter=5, random_state=0)
    # Note that the imputation is set to round to 0 since the data is categorical
    TEDS_A_Imputed = np.round(imputer.fit_transform(data_frame), 0)
    TEDS_A_Imputed = pd.DataFrame(TEDS_A_Imputed)
    TEDS_A_Imputed.columns = list(data_frame.columns)
    return(TEDS_A_Imputed)

# Define the polynomial function.
def user_interactions(TEDS_A_Imputed):
    """Queries the user for variables they wish to generate interactions for.
    
    Returns
        A data frame with interaction terms."""
        
    print("""You now have the option to generate polynomial features for variables of interest (enter NA to skip). 
Please use the TEDS-A codebook or the list generated to enter the names of variables of interest.\n 
Enter the column/variable names separated by a space and not a comma, example: CASEID EDUC MARSTAT\n
Currently available columns: """, list(TEDS_A_Imputed.columns))
 
    print("\n")
    user_list2 = input()
        
    if user_list2 == "NA" or user_list2 == "na":
        return(print("No polynomial features will be generated."))
    else:
        feature_list = user_list2.split(" ")
        featuregen = TEDS_A_Imputed[feature_list]
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        TEDS_Af = poly.fit_transform(featuregen)
        TEDS_Af = pd.DataFrame(TEDS_Af)
        TEDS_Af.columns = poly.get_feature_names(featuregen.columns)
        TEDS_Af = TEDS_Af.drop(feature_list, axis=1)
        return(TEDS_Af)
    
# Note that there are some missing states based on reporting and supression
# This may cause the code to faulter in other years
# In future iterations of the pipeline, this may be addressed by providing a code for all possible values
def var_recode(TEDS_A_Imputed, user_target, TEDS_Af):
    """Recodes categorical variables present in the data.
    
    Returns
        A recoded data frame."""
        
    # Identify the variable types
    col_list = list(TEDS_A_Imputed.columns)
    col_list_cat = [s for s in col_list if s != 'CASEID' and s != 'ADMYR' and s != 'AGE' 
                    and s != 'EDUC' and s != 'ARRESTS']
    col_list_ord = [s for s in col_list if s == 'ADMYR' or s == 'AGE' 
                    or s == 'EDUC' or s == 'ARRESTS' and s != user_target]
    col_list_cat_f = list(TEDS_Af.columns)
    TEDS_A_Imputed[col_list_cat] = TEDS_A_Imputed[col_list_cat].astype('category')
    TEDS_A_Imputed[col_list_ord] = TEDS_A_Imputed[col_list_ord].astype('category')
    TEDS_A_Imputed['CASEID'] = TEDS_A_Imputed['CASEID'].astype('int64')
    TEDS_A_Imputed[user_target] = TEDS_A_Imputed[user_target].astype('category')
    TEDS_Af[col_list_cat_f] = TEDS_Af[col_list_cat_f].astype('category')

    if 'STFIPS' in col_list:

        # Recoding the States variable
        TEDS_A_Imputed['STFIPS'] = TEDS_A_Imputed['STFIPS'].replace([1,2,4,5,6,7,8,10,11,12,
                                                                     13,15,16,17,18,19,20,21,
                                                                     22,23,24,25,26,27,28,29,
                                                                     30,31,32,33,34,35,36,37,
                                                                     38,39,40,42,44,45,46,47,
                                                                     48,49,50,51,53,54,55,56,
                                                                     72], 
                                                                    ['Alabama','Alaska','Arizona', 
                                                                     'Arkansas','California','Colorado',
                                                                     'Connecticut','Delaware','District of Columbia',
                                                                     'Florida','Georgia','Hawaii','Idaho',
                                                                     'Illinois','Indiana','Iowa','Kansas',
                                                                     'Kentucky','Louisiana','Maine','Maryland',
                                                                     'Massachusetts','Michigan','Minnesota',
                                                                     'Mississippi','Missouri','Montana','Nebraska',
                                                                     'Nevada','New Hampshire','New Jersey',
                                                                     'New Mexico','New York','North Carolina',
                                                                     'North Dakota','Ohio','Oklahoma','Pennsylvania',
                                                                     'Rhode Island','South Carolina','South Dakota',
                                                                     'Tennessee','Texas','Utah','Vermont','Virginia',
                                                                     'Washington','West Virginia','Wisconsin',
                                                                     'Wyoming','Puerto Rico'])
        #Recast as a categorical variable
        TEDS_A_Imputed["STFIPS"] = TEDS_A_Imputed["STFIPS"].astype("category")
        
    if 'EDUC' in col_list:
        # Recoding the Education variable
        TEDS_A_Imputed['EDUC'] = TEDS_A_Imputed['EDUC'].replace([1,2,3,4,5],
                                                                ['8 years or less',
                                                                 '9–11 years','12 years (or GED)',
                                                                 '13–15 years','16 years or more'])

        #Recast as a categorical variable
        TEDS_A_Imputed['EDUC'] = TEDS_A_Imputed['EDUC'].astype("category")
        
    if 'AGE' in col_list:
        # Recoding the Age variable
        TEDS_A_Imputed['AGE'] = TEDS_A_Imputed['AGE'].replace([1,2,3,4,5,6,7,8,9,10,11,12],
                                                              ['12–14 years','15–17 years','18–20 years',
                                                               '21–24 years','25–29 years','30–34 years',
                                                               '35–39 years','40–44 years','45–49 years',
                                                               '50–54 years','55–64 years','65 years and older'])
        #Recast as a categorical variable
        TEDS_A_Imputed['AGE'] = TEDS_A_Imputed['AGE'].astype("category")
        
    if 'SERVICES' in col_list:
        # Recoding the Services variable
        TEDS_A_Imputed['SERVICES'] = TEDS_A_Imputed['SERVICES'].replace([1,2,3,4,5,6,7,8],
                                                                        ['Detox, 24-hour, hospital inpatient',
                                                                         'Detox, 24-hour, free-standing residential',
                                                                         'Rehab/residential, hospital (non-detox)',
                                                                         'Rehab/residential, (30 days or fewer)',
                                                                         'Rehab/residential, (more than 30 days)',
                                                                         'Ambulatory, intensive outpatient',
                                                                         'Ambulatory, non-intensive outpatient',
                                                                         'Ambulatory, detoxification'])
        #Recast as a categorical variable
        TEDS_A_Imputed['SERVICES'] = TEDS_A_Imputed['SERVICES'].astype("category")
        
    if 'RACE' in col_list:
        # Recoding the Race variable
        TEDS_A_Imputed['RACE'] = TEDS_A_Imputed['RACE'].replace([1,2,3,4,5,6,7,8,9],
                                                                ['Alaska Native (Aleut, Eskimo, Indian)',
                                                                 'American Indian (other than Alaska Native)',
                                                                 'Asian or Pacific Islander', 'Black or African American',
                                                                 'White','Asian','Other single race','Two or more races',
                                                                 'Native Hawaiian or Other Pacific Islander'])
        #Recast as a categorical variable
        TEDS_A_Imputed['RACE'] = TEDS_A_Imputed['RACE'].astype("category")
        
    if 'ETHNIC' in col_list:
        # Recoding the Ethnicity variable
        TEDS_A_Imputed['ETHNIC'] = TEDS_A_Imputed['ETHNIC'].replace([1,2,3,4,5],
                                                                    ['Puerto Rican','Mexican',
                                                                     'Cuban or other specific Hispanic',
                                                                     'Not of Hispanic or Latino origin',
                                                                     'Hispanic or Latino, specific origin not specified'])
        #Recast as a categorical variable
        TEDS_A_Imputed['ETHNIC'] = TEDS_A_Imputed['ETHNIC'].astype("category")
        
    if 'HLTHINS' in col_list:
        # Recoding the Health Insurance variable
        TEDS_A_Imputed['HLTHINS'] = TEDS_A_Imputed['HLTHINS'].replace([1,2,3,4],
                                                                      ['Private insurance',
                                                                       'Medicaid','Medicare',
                                                                       'None'])
        #Recast as a categorical variable
        TEDS_A_Imputed['HLTHINS'] = TEDS_A_Imputed['HLTHINS'].astype("category")
        
    if 'REGION' in col_list:
        # Recoding the Region variable
        TEDS_A_Imputed['REGION'] = TEDS_A_Imputed['REGION'].replace([1,2,3,4],
                                                                    ['U.S. territories',
                                                                     'Northeast','Midwest',
                                                                     'South','West'])
        #Recast as a categorical variable
        TEDS_A_Imputed['REGION'] = TEDS_A_Imputed['REGION'].astype("category")
        
    if 'DSMCRIT' in col_list:
        # Recoding the DSM Diagnosis variable
        TEDS_A_Imputed['DSMCRIT'] = TEDS_A_Imputed['DSMCRIT'].replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
                                                                      ['Alcohol-induced disorder','Substance-induced disorder',
                                                                       'Alcohol intoxication','Alcohol dependence',
                                                                       'Opioid dependence','Cocaine dependence',
                                                                       'Cannabis dependence','Other substance dependence',
                                                                       'Alcohol abuse','Cannabis abuse',
                                                                       'Other substance abuse','Opioid abuse','Cocaine abuse',
                                                                       'Anxiety disorders','Depressive disorders',
                                                                       'Schizophrenia/other psychotic disorders',
                                                                       'Bipolar disorders',
                                                                       'Attention deficit/disruptive behavior disorders',
                                                                       'Other mental health condition'])
        #Recast as a categorical variable
        TEDS_A_Imputed['DSMCRIT'] = TEDS_A_Imputed['DSMCRIT'].astype("category")
        
    return(TEDS_A_Imputed, col_list_cat, col_list_ord, col_list_cat_f)    
    
# Define the combination function.
def combine_levels(user_target, TEDS_A_Imputed):
    """Queries the user to combine target variable levels.
    
    Returns
        A target variable (in the data frame) with collapsed levels."""
        
    print("You now have the option to collapse your target variable levels (type 'NA' to skip).") 
    print("Please enter up to three categories for your new level recoding. The reference is the first entry.") 
    
    print("Currently available levels in {}: {}".format(user_target, list(TEDS_A_Imputed[user_target].values.unique())))
    print("\n")
    
    user_combined1 = input()

    if user_combined1 == "NA" or user_combined1 == "na":
        return(print("No columns will be dropped."))
    else:
        user_combined2 = input()
        user_combined3 = input()
        existing = list(TEDS_A_Imputed[user_target].values.unique())
        if user_combined3 == "NA" or user_combined3 == "na":
            existing2 = [1 if x != user_combined1 else 0 for x in existing]
        
            collapse = {} 
            for key in existing: 
                for value in existing2: 
                    collapse[key] = value 
                    existing2.remove(value) 
                    break
            TEDS_A_Imputed[user_target] = pd.DataFrame(TEDS_A_Imputed[user_target].map(collapse))
            return(TEDS_A_Imputed, user_combined1, user_combined2, user_combined3)
        else:
            existing2 = [1 if x == user_combined2 else x for x in existing]
            existing3 = [2 if x == user_combined3 else x for x in existing2]
            existing4 = [1 if x == 1 else 2 if x == 2 else 0 for x in existing3]
            collapse = {} 
            for key in existing: 
                for value in existing4: 
                    collapse[key] = value 
                    existing4.remove(value) 
                    break
            TEDS_A_Imputed[user_target] = pd.DataFrame(TEDS_A_Imputed[user_target].map(collapse))
            return(TEDS_A_Imputed, user_combined1, user_combined2, user_combined3)
  
def cleaning_pt2(TEDS_A_Imputed, TEDS_Af, col_list_cat, col_list_cat_f, col_list_ord):    
    enc = preprocessing.OrdinalEncoder()
    TEDS_Aic = TEDS_A_Imputed
    TEDS_Aic['CASEID'] = TEDS_A_Imputed['CASEID']
    temp_cat = pd.get_dummies(TEDS_A_Imputed[col_list_cat])
    temp_cat2 = pd.get_dummies(TEDS_Af[col_list_cat_f])
    temp_ord = pd.DataFrame(enc.fit_transform(TEDS_A_Imputed[col_list_ord]))
    temp_ord.columns = col_list_ord
    temp_full1 = temp_ord.join(temp_cat)
    temp_full2 = temp_full1.join(temp_cat2)
    TEDS_Aic = TEDS_A_Imputed[['CASEID']].copy()
    TEDS_Aicf = TEDS_Aic.join(temp_full2)
    return(TEDS_Aicf)

# Define the polynomial function.
def user_split(TEDS_Aicf, user_target):
    """Queries the user for the desired ratio of training to testing.
    
    Returns
        Training and testing data.
    Raises
        FormatError: In case `user_test is not an integer`"""
        
    print("You can now specify the percent split for training and testing data.\n")
    
    print("Enter the percentage of data you wish to be testing as an integer.",
          "Example: 20 would translate to a testing dataset of 20%.\n")
    user_test = int(input())
    
    TEDS_Aicf.columns = TEDS_Aicf.columns.astype(str)
    filter_col = [col for col in TEDS_Aicf if col.startswith(user_target)]
    y_list = TEDS_Aicf[filter_col]
    y_list = list(y_list.columns)
        
    if isinstance(user_test, str):
        return(print("Please enter an integer."))
    else:
        user_test2 = (user_test / 100)
        user_train = (100 - user_test)
        # Create the training and testing data
        TEDS_A_X = TEDS_Aicf.drop(columns=y_list)
        TEDS_A_Y = TEDS_Aicf[y_list]
        TEDS_X_train, TEDS_X_test, TEDS_Y_train, TEDS_Y_test = train_test_split(TEDS_A_X, TEDS_A_Y, 
                                                                            test_size = user_test2, random_state = 30)
        print("Training and testing datasets were created with {}% and {}% of the original data.".format(user_train, user_test))
        return(TEDS_X_train, TEDS_X_test, TEDS_Y_train, TEDS_Y_test, y_list)

def model_building(TEDS_X_train,  TEDS_Y_train, TEDS_X_test):
    # Default hyperparameters (baseline performance)
    # Import the random forest classifier
    rfc = ensemble.RandomForestClassifier()
    analysis=Pipeline(steps=([('Base', ensemble.RandomForestClassifier())]))
    
    analysis.fit(TEDS_X_train, TEDS_Y_train.values.ravel())
    Y_pred = analysis.predict(TEDS_X_test)
    
    # Tune the hyperparameters for the random forest classifier
    param_dist = {'n_estimators': list(np.linspace(100, 200, 10, dtype = int)),
                  'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                  'min_samples_leaf': [1, 2, 4]}
    random_grid_search = RandomizedSearchCV(estimator = rfc, param_distributions = param_dist, 
                              cv = 2, n_jobs = 4, verbose = 1, n_iter = 5)
    random_grid_search.fit(TEDS_X_train, TEDS_Y_train.values.ravel())
    best_grid_randomsearch = random_grid_search.best_estimator_
    Y_pred_randomsearch = best_grid_randomsearch.predict(TEDS_X_test)
    return(Y_pred, Y_pred_randomsearch, best_grid_randomsearch)

def model_eval(TEDS_Y_test, Y_pred, Y_pred_randomsearch):
    # Test the predictive capability of the fitted data
    mae = mean_absolute_error(TEDS_Y_test, Y_pred)
    acc1 = metrics.accuracy_score(TEDS_Y_test, Y_pred)
    print("The model performance for baseline model is:")
    print("---------------------------------------------")
    print('The mean absoulte error is {}'.format(mae))
    print('The accuracy score is {}'.format(acc1))
    print('\n\n')
    # Test the predictive capability of the fitted data
    mae_randomsearch = mean_absolute_error(TEDS_Y_test, Y_pred_randomsearch)
    acc2 = metrics.accuracy_score(TEDS_Y_test, Y_pred_randomsearch)
    print("The model performance for testing set from randomsearch search")
    print("--------------------------------------")
    print('The mean absoulte error is {}'.format(mae_randomsearch))
    print('The accuracy score is {}'.format(acc2))
    print('Improvement of {:0.2f}%.'.format( 100 * (acc2 - acc1) / acc1))
    return()
    
def model_confusion_matrix(user_combined1, user_combined2, user_combined3, 
                           TEDS_X_test, TEDS_Y_test, best_grid_randomsearch, TEDS_X_train, TEDS_Y_train):
    # Plot confusion matrix
    if user_combined3 == "NA" or user_combined3 == "na":
        class_names = [user_combined1, user_combined2]
    else:
        class_names = [user_combined1, user_combined2, user_combined3]
    classifier = best_grid_randomsearch.fit(TEDS_X_train, TEDS_Y_train.values.ravel())
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, TEDS_X_test, TEDS_Y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        plt.xticks(rotation=45)
    plt.show()
    return()

def export_fun(best_grid_randomsearch, TEDS_Aicf):
    final_model = best_grid_randomsearch
    filename = 'Final_TEDS_A_model.sav'
    pickle.dump(final_model, open(filename, 'wb'))
    print("The model has been saved as a pickle file: Final_TEDS_A_model.sav.")
    
    # Export the files as pickle for other purposes
    TEDS_Aicf.to_csv('TEDSA_2015_2017_Final')

    # Export the files as pickle for other purposes
    TEDS_Aicf.to_pickle('TEDSA_2015_2017_Final')
    print("The data has been saved as a .csv and pickle format: TEDSA_2015_2017_Final")
    return()