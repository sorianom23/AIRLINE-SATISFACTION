
# LIBRARIES:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import ConfusionMatrixDisplay


# GET TO KNOW THE DATA

def df_info(df):
    info = print(df.isna().sum(), df.info(), df.shape, list(df.columns), sep="\n")
    
    return(info)


# NULLS PERCENT

def nulls_percent(df):
    
    '''
    Shows percent of nulls in a data frame.
    
    Args:
        df: The dataframe we want to check out.
        
    Returns:
        A new df with 2columns:
        - 'column_name' with the name of the original df columns
        - 'nulls_percentage' with the percentage of nulls in every column
    '''
    nulls_percent = pd.DataFrame(df.isna().sum()/len(df)).reset_index()
    nulls_percent.columns = ['column_name', 'nulls_percentage']
    
    return nulls_percent


# PLOT NUMERICAL DISTRIBUTION

    def plot_distributions(df):
        for col in df:
            sns.displot(data[col])
            plt.show()


# UNIQUE VALUES FOR CATEGORICAL COLUMNS

def unique_values(df):

    categories = df.select_dtypes('O').columns.astype('category')

    for i in df[categories].columns:
        print(f'The categorical column  {i} has  {len(df[i].value_counts())}  unique values')


# PLOTTING RESULTS WITH CM
def plot_results(y_test,y_pred_test):
    print("The accuracy in the TEST set is: {:.2f}".format(accuracy_score(y_test,y_pred_test)))
    print("The precision in the TEST set is: {:.2f}".format(precision_score(y_test,y_pred_test, pos_label=1)))
    print("The recall in the TEST set is: {:.2f}".format(recall_score(y_test,y_pred_test, pos_label=1)))
    print("The F1 in the TEST set is: {:.2f}".format(f1_score(y_test,y_pred_test, pos_label=1)))
    print("The Kappa in the TEST set is: {:.2f}".format(cohen_kappa_score(y_test,y_pred_test))) 

    cm_test = confusion_matrix(y_test,y_pred_test)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)

    disp.plot()
    plt.show()


# FUNCTION TO READ DATA

def read_data(x):
    data = pd.read_csv(x)
    return data
