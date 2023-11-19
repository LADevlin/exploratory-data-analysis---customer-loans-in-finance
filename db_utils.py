import yaml
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine, engine
from statsmodels.graphics.gofplots import qqplot


class RDSDatabaseConnector:
    '''
    The class is used to represent an Amazon RDS Connection

    Attributes:
        credentials (.yaml): A dictionary containing connection details:
            RDS_HOST: The url link to the database
            RDS_PASSWORD: The password to the database
            RDS_USER: The username for the database
            RDS_DATABASE: The database connecting to
            RDS_PORT: The port for the database
    '''
    def __init__(self, credentials):
        '''
        See help(RDSDatabaseConnector) for accurate signature
        '''
        self.host = credentials['RDS_HOST']
        self.password = credentials['RDS_PASSWORD']
        self.user = credentials['RDS_USER']
        self.database = credentials['RDS_DATABASE']
        self.port = credentials['RDS_PORT']
    
    def initialise_sql(self):
        '''
        This function is used to create an SQL Engine

        Returns:
            engine: The SQL Engine
        '''
        # Create URL for engine
        url_object = engine.URL.create(
            'postgresql',
            username=self.user,
            password=self.password,
            host=self.host,
            database=self.database,
            port=self.port
        )
        # Create SQL Engine
        sql_engine = create_engine(url_object)
        return sql_engine

    def create_dataframe(self):
        '''
        This function is used to create a DataFrame from the Engine

        Returns:
            DataFrame: The DataFrame created from the engine
        '''
        # Extract the loan payments table from the initialised database
        df = pd.read_sql_table('loan_payments', self.initialise_sql())
        return df
    
    def create_csv(self):
        '''
        This function is used to create a .csv file from the Engine

        Outputs:
            .csv: The .csv file created from the DataFrame
        '''
        # Create csv file
        self.create_dataframe().to_csv('loan_payments.csv')

class DataTransform:
    '''
    This class is used to represent the Data Transform from csv to Data Frame

    Attributes:
        df (dataframe): A DataFrame created from a .csv file
        int_col (list): A list containing all columns to be converted to int64
        float_col (list): A list containing all columns to be converted to float64
        cat_col (list): A list containing all columns to be converted to category
        dat_col (list): A list containing all columns to be converted to datetime
    '''
    def __init__(self, df, int_col=None, float_col=None, cat_col=None, dat_col=None):
        '''
        See help(DataTransform) for accurate signature
        '''
        # Conversion of each type Int64, float64, category, datetime
        self.df = df
        self.df[int_col] = self.df[int_col].astype('Int64')
        self.df[float_col] = self.df[float_col].astype('float64')
        self.df[cat_col] = self.df[cat_col].astype('category')
        self.df[dat_col] = self.df[dat_col].apply(pd.to_datetime, format='mixed')

class DataFrameInfo:
    '''
    This class is a used to represent the information and processes for the DataFrame

    Attributes:
        df (dataframe): A dataframe with correct types

    Returns:
        .df (dataframe): Returns the dataframe
        .median (series): Returns the median of each numerical column in the dataframe
        .std (series): Returns the standard deviation of each numerical column in the dataframe 
        .mean (series): Returns the mean of each numerical column in the dataframe
        .mode (series): Returns the mode of each column in the database
        .shape (tuple): Returns the shape of the dataframe
        .null_count (series): Returns the count of null values for each column in the dataframe
        .null_pc (series): Returns the null percentage for each column in the dataframe
        .skew (series): Returns the skew of each numerical column in the dataframe
        .skew_columns (list): Returns a list of every numerical column with a skew higher than 0.5
    '''
    def __init__(self, df):
        '''
        See help(DataFrameInfo) for accurate signature
        '''
        self.df = df
        self.median = df.median(skipna=True, numeric_only=True)
        self.std = df.std(skipna=True, numeric_only=True)
        self.mean = df.mean(skipna=True, numeric_only=True)
        self.mode = df.mode()
        self.shape = df.shape
        self.null_count = df.isna().sum()
        self.null_pc = df.isna().sum()/len(df)
        self.skew = df.skew(numeric_only=True)        
        self.skew_columns = list(self.skew.loc[self.skew > 0.5].keys())[2:] #List of columns that skew is higher than 0.5 and ignores id and member_id

    def info(self):
        '''
        This function is used to show information on the dataframe

        Returns:
            Object showing information on dataframe
        '''
        return self.df.info()
    
    def median_col(self, column):
        '''
        This is a function to give a median of a given column

        Args:
            column (str): The column key
        
        Returns:
            int: The median of the column
        '''
        return self.df[column].median(skipna=True)

    def std_col(self, column):
        '''
        This is a function to give a standard deviation of a given column

        Args:
            column (str): The column key
        
        Returns:
            int: The mstandard deviation of the column
        '''
        return self.df[column].std(skipna=True)

    def mean_col(self, column):
        '''
        This is a function to give a mean of a given column

        Args:
            column (str): The column key
        
        Returns:
            int: The mean of the column
        '''
        return self.df[column].mean(skipna=True)
    
    def mode_col(self, column):
        '''
        This is a function to give a mode of a given column

        Args:
            column (str): The column key
        
        Returns:
            int: The mode of the column
        '''
        return self.df[column].mode()[0]
    
    def value_counts(self, column):
        '''
        This is a function to give the counts of each unique value of a given column

        Args:
            column (str): The column key
        
        Returns:
            series: The counts of each unique value of the column
        '''
        return self.df[column].value_counts()

    def nunique(self, column):
        '''
        This is a function to give the number of unique values of a given column

        Args:
            column (str): The column key
        
        Returns:
            series: The number of unique values of the column
        '''
        return self.df[column].nunique()
    
class DataFrameTransform(DataFrameInfo):
    '''
    This class is used to represent transformations on the dataframe

    Attributes:
        df (dataframe): dataframe to have transformations applied
    '''
    def __init__(self, df):
        '''
        See help(DataFrameTransform) for accurate signature
        '''
        super().__init__(df)

    def drop_cols(self, column):
        '''
        This is a function to drop columns provided

        Args:
            column (list/str): list or string of column(s) to be removed from dataframe

        Output:
            Will update dataframe
        '''
        self.df.drop(columns=column, inplace=True)
    
    def median_impute(self, column):
        '''
        This is a function to impute the median value of the column into null values

        Args:
            column (list): list of numerical columns to impute

        Output:
            Will update dataframe
        '''
        # Runs through every column in the list and imputes median value
        for col in column:
            self.df[col].fillna(value=self.median_col(col), inplace=True)
        
    def mode_impute(self, column):
        '''
        This is a function to impute the mode value of the column into null values

        Args:
            column (list): list of columns to impute

        Output:
            Will update dataframe
        '''
        # Runs through every column in the list and imputes mode value
        for col in column:
            self.df[col].fillna(value=self.mode_col(col), inplace=True)
    
    def ffill_impute(self, column):
        '''
        This is a function to impute via forward fill of the column into null values

        Args:
            column (list): list of columns to impute

        Output:
            Will update dataframe
        '''
        for col in column:
            self.df[col].ffill(inplace=True)
    
    def correct_skew(self):
        '''
        This is a function to reduce the skew via log transformation

        Output:
            original_df (dataframe) = the dataframe before transformation

        Returns:
            (dataframe) = the dataframe with reduced skew
        '''
        self.original_df = self.df.copy() # saves the original dataframe
        for col in self.skew_columns:
            self.df[col] = self.df[col].map(lambda x: np.log(x) if x > 0 else 0) # log transformation, ignores 0 values to prevent eroor
        return self.df
    
    def remove_outliers(self, column):
        '''
        This is a function to remove outliers from the dataframe

        Args:
            column (list of lists): list of (column_key, max_value) where rows that column key is greater than the max value will be removed

        Output:
            outliers_inc_df (dataframe) = the dataframe before transformation
            updates dataframe
        '''
        self.outliers_inc_df = self.df.copy() # saves the original dataframe
        for col in column:
            self.df.drop(self.df.loc[self.df[col[0]]>col[1]].index, inplace=True)

class Plotter(DataFrameTransform):
    '''
    This class is used to represent plotting information

    Attributes:
        df (dataframe): dataframe to plots applied
    '''

    def __init__(self, df):
        '''
        See help(Plotter) for accurate signature
        '''
        super().__init__(df)
    
    def plot_null(self):
        '''
        This function creates a plot of null values within the dataframe

        Outputs:
            Null matrix plot
        '''
        msno.matrix(pd.DataFrame(self.df))
        plt.show()
    
    def plot_dist_col(self, column):
        '''
        This function creates a histogram distribution plot for a given column

        Args:
            column (str): Column key to create a distribution plot
        
        Output:
            Histogram distribution plot
        '''
        self.df[column].hist(bins=100)
        plt.title(column)
        plt.show()

    def plot_dist(self):
        '''
        This function creates a histogram distribution plot for all numeric columns in dataframe
        
        Output:
            Histogram distribution plot
        '''
        titles = self.df.select_dtypes(include=np.number).keys() #list of numerical columns
        titles = titles.delete([0, 1]) # remove id and member_id
        fig, axes = plt.subplots(nrows=len(titles), ncols=1) # creates frame for subplots, a list of plots
        # Run through list of columns adding a subplot of histogram distribution plot for each column
        i = 0
        for title in titles:
            self.df[title].hist(bins=30, ax=axes[i])
            axes[i].set_title(title)
            i += 1
        plt.show()
    
    def plot_qq_col(self, column):
        '''
        This function creates a qq plot for a given column

        Args:
            column (str): Column key to create a qq plot
        
        Output:
            qq plot
        '''
        qqplot(self.df[column], scale=1, line='q', fit=True)
        plt.show()
    
    def plot_qq(self):
        '''
        This function creates a qq plot for all numeric columns in dataframe
        
        Output:
            qq plot
        '''
        titles = self.df.select_dtypes(include=np.number).keys() # create list of numerical column keys
        titles = titles.delete([0, 1]) # remove id and member_id
        fig, axes = plt.subplots(nrows=len(titles), ncols=1) # creates frame for subplots, list of plots
        i = 0
        for title in titles:
            try:
                qqplot(self.df[title], scale=1, line='q', fit=True, ax=axes[i])
                axes[i].set_title(title)
            except:
                print(f"Can't run {title}") # error prevention
            finally:
                i += 1
        plt.show()

    def plot_box_col(self, column):
        '''
        This function creates a box plot for a given column

        Args:
            column (str): Column key to create a box plot
        
        Output:
            box plot
        '''
        sns.boxplot(self.df[column])
        plt.show()
        
    def plot_box(self):
        '''
        This function creates a box plot for all numeric columns in dataframe
        
        Output:
            box plot
        '''
        titles = self.df.select_dtypes(include=np.number).keys() # creates list of numerical column keys
        titles = titles.delete([0, 1]) # delete id and member_id
        fig, axes = plt.subplots(nrows=(len(titles)//3)+1, ncols=3) # create a frame for subplots, will be 3 columns wide
        i = 0
        for title in titles:
            sns.boxplot(self.df[title], ax=axes[i // 3, i % 3]) # cycles through the subplot frame left to right, top to bottom
            axes[i // 3, i % 3].set_title(title)
            i += 1
        plt.show()
    
    def corr_matrix(self):
        '''
        This function creates a correlation matrix for all numeric columns in dataframe
        
        Output:
            correlation matrix
        '''
        #sets up correlation matrix to be readable
        plt.matshow(self.df.select_dtypes(include=np.number).corr())
        plt.xticks(range(self.df.select_dtypes(include=np.number).shape[1]), self.df.select_dtypes(include=np.number).columns, fontsize=14, rotation=45)
        plt.yticks(range(self.df.select_dtypes(include=np.number).shape[1]), self.df.select_dtypes(include=np.number).columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.show()


def load_credentials():
    '''
    This function opens a credentials.yaml file and converts to dictionary

    Returns:
        (dict) : dictionary containing credentials
    '''
    # open credentials.yaml in read mode
    with open('credentials.yaml', mode='r') as file:
        # load yaml as dictionary
        credentials_dict = yaml.safe_load(file)
        return credentials_dict

def create_sql_class():
    '''
    This function runs the credentials to the Database Connector

    Returns:
        (engine): sql engine
    '''
    return RDSDatabaseConnector(load_credentials)

def load_csv(file):
    '''
    This function reads a csv file and converts to a dataframe

    Args:
        file (.csv): a csv file to open
    
    Returns:
        (dataframe): dataframe from the given csv file
    '''
    # Load csv file as dataframe and return
    df = pd.read_csv(file, index_col=0)
    return df

def transform_data(file):
    '''
    This function converts a dataframe and converts column type

    Args:
        file (.csv): .csv file to be converted to dataframe
    
    Returns:
        (dataframe): dataframe that has correct data type columns
    '''
    # Input for which columns to be which type
    integer_columns = ['id', 'member_id', 'mths_since_last_record', 'open_accounts', 'total_accounts', 'mths_since_last_major_derog']
    float_columns = ['loan_amount', 'funded_amount', 'funded_amount_inv', 'int_rate', 'instalment', 'annual_inc', 'dti', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_payment_amount']
    category_columns = ['term', 'grade', 'sub_grade', 'employment_length', 'home_ownership', 'verification_status', 'loan_status', 'payment_plan', 'purpose', 'delinq_2yrs', 'inq_last_6mths', 'collections_12_mths_ex_med', 'policy_code', 'application_type']
    datetime_columns = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']

    # Assign to DataTransform Class
    df_transform = DataTransform(load_csv(file), integer_columns, float_columns, category_columns, datetime_columns)

    return df_transform.df

def transform_dataframe(df):
    '''
    This function performs all desired dataframe transformations

    Args:
        df (dataframe): dataframe to have transformations made

    returns:
        (dataframe): transformed dataframe
    '''
    # Columns to drop with large number of null values
    drop_columns = ['mths_since_last_delinq', 'mths_since_last_record', 'next_payment_date', 'mths_since_last_major_derog']
    # Columns to impute via median
    median_columns = ['funded_amount', 'int_rate']
    # COlumns to impute via forward fill
    ffill_columns = ['last_payment_date', 'last_credit_pull_date']
    # Columns to impute via mode (categorical)
    mode_columns = ['term', 'employment_length', 'collections_12_mths_ex_med']
    # Columns with outliers, (column_key, max_value)
    outlier_columns = [('loan_amount', 34500), ('funded_amount', 34200), ('funded_amount_inv', 34950), ('int_rate', 24), ('total_accounts', 62), ('total_rec_prncp', 34000)]
    # Columns that have high correlation with another column, to be dropped
    high_corr_columns = ['id', 'member_id', 'funded_amount', 'funded_amount_inv', 'instalment', 'total_accounts', 'total_rec_prncp', 'out_prncp_inv', 'total_payment_inv', 'total_rec_int', 'collection_recovery_fee']

    tf_df = DataFrameTransform(df.copy()) #Transforms on Copied dataframe to not lose original data

    # Apply transformations
    tf_df.drop_cols(drop_columns)
    tf_df.median_impute(median_columns)
    tf_df.ffill_impute(ffill_columns)
    tf_df.mode_impute(mode_columns)
    tf_df.remove_outliers(outlier_columns)
    tf_df.drop_cols(high_corr_columns)

    return tf_df

def current_state(df):
    '''
    This function prints an analysis of the current state of loans

    Args:
        df (dataframe): Dataframe to analyse
    
    Outputs:
        Total Loan Amount: Total Amount of Loans Provided
        Total Investor Funded Amount: Total Amount Funded by Investors
        Total Payments Made: Total Payments Received, Total Revenue
        Recovered Against Loan Amount: Percentage of Total Revenue against Total Loans Provided
        Recovered Against Investor Funded: Percentage of Total Revenue against Total Investor Funded Amount
    '''
    state_df = df[['loan_amount', 'funded_amount_inv', 'total_payment']].copy() # Creates copy of dataframe with needed columns
    #Create New columns with percent recovered for each member
    state_df['recovered_against_loan_(%)'] = df['total_payment']/df['loan_amount']
    state_df['recovered_against_inv_(%)'] = df['total_payment']/df['funded_amount_inv']
    #print outputs
    print(state_df)
    total_loan = round(df['loan_amount'].sum(), 2)
    total_inv = round(df['funded_amount_inv'].sum(), 2)
    total_payment = round(df['total_payment'].sum(), 2)
    print(f'Total Loan Amount: £{total_loan}')
    print(f'Total Investor Funded Amount: £{total_inv}')
    print(f'Total Payments Made: £{total_payment}')
    print(f'Recovered Against Loan Amount {round((total_payment/total_loan) * 100, 2)}%')
    print(f'Recovered Against Investor Funded {round((total_payment/total_inv) * 100, 2)}%')

def sixmnths_state(df):
    '''
    This function prints a predicted analysis for 6 months in the future

    Args: 
        df (dataframe): dataframe to be analysed
    
    Outputs:
        Scatter Plot: A Scatter Plot showing the predicted return percentage for each month up to 6 months into the future
        Return Percentage at 6 Months from Loan Amount: Percent 6 months into the future of percentage of total revenue against total loans
        Return Percentage at 6 Months from Invested Amount: Percent 6 months into the future of percentage of total revenue against investor funded amounts
    '''
    state_df = df[['loan_amount', 'funded_amount_inv', 'total_payment', 'out_prncp', 'last_payment_amount']].copy() # Creates copy of dataframe with needed columns
    # Initialise lists for plotting later
    recover_against_loan = [0] * 7
    recover_against_inv = [0] * 7
    recover_against_loan[0] = state_df['total_payment'].sum() / state_df['loan_amount'].sum()
    recover_against_inv[0] = state_df['total_payment'].sum() / state_df['funded_amount_inv'].sum()
    # Run for range where number is months in the future
    for i in range(6):
        state_df['out_prncp'] = state_df['out_prncp'] - state_df['last_payment_amount'] # Take away payment from left owed
        title = f'total payment +{i+1} mths' # Initialise title name variable
        state_df[title] = state_df['total_payment'] + state_df['last_payment_amount'] # New total payments is old total plus payment
        state_df[title] = state_df[title] + state_df['out_prncp'].map(lambda x: x if x < 0 else 0) # Takes away overpayment (If out_prncp is negative then overpaid)
        state_df['out_prncp'] = state_df['out_prncp'].map(lambda x: 0 if x < 0 else x) # Sets owed money back to 0 (if negative it's been corrected in the last step)
        state_df['total_payment'] = state_df[title] # updates total payment
        recover_against_loan[i+1] = state_df[title].sum() / state_df['loan_amount'].sum() # saves percentage return for loan to list
        recover_against_inv[i+1] = state_df[title].sum() / state_df['funded_amount_inv'].sum() # saves percentage return for investor to list
    print(state_df)
    # Creates Graph of return percentage
    months = [0, 1, 2, 3, 4, 5, 6] # initialise months
    plt.plot(months, recover_against_loan, color='blue')
    plt.plot(months, recover_against_inv, color='red')
    plt.xlabel('Months into Future')
    plt.ylabel('Percentage Recovered')
    plt.legend(['Recovered Against Loan', 'Recovered Against Investor'])
    plt.show()
    # Prints return percentages after 6 months
    print(f'Return Percentage at 6 months from Loan Amount: {round(recover_against_loan[6] * 100, 2)}%')
    print(f'Return Percentage at 6 months from Invested Amount: {round(recover_against_inv[6] * 100, 2)}%')

def calculate_loss(df):
    '''
    This function shows the current status of charged off loans

    Args:
        df (dataframe): Dataframe to be analysed
    
    Outputs:
        Number of Charged Off Loans: Total number of current charged off loans
        Percentage that is Charged off Loans: Percentage of total loans that have been charged off
        Total Payment of Charged Off Loans: Total Payment that has already been paid by charged off loans
    '''
    number_of_charged_off = df.loc[df['loan_status'] == 'Charged Off', 'loan_status'].count() #Counts series containing only charged off
    percent_of_charged_off = number_of_charged_off / len(df['loan_status']) #Uses previous value over total values
    total_payment_charged_off = round(df.loc[df['loan_status'] == 'Charged Off']['total_payment'].sum(), 2) #selects values of total payment where loan status is charged off
    # Prints outputs
    print(f'Number of Charged Off Loans: {number_of_charged_off}')
    print(f'Percentage that is Charged Off Loans: {round(percent_of_charged_off * 100, 2)}%')
    print(f'Total Payment of Charged Off Loans: £{total_payment_charged_off}')

def calculate_projected_loss(df):
    '''
    This function shows the current loss by Charged Off Loans by total that would have been taken from Loan

    Args:
        df (dataframe): Dataframe to be analysed
    
    Outputs:
        Total Revenue Lost: Total Revenue Lost from potential total loan with interest rate
        Percentage of Revenue Lost: Percentage of Revenue Lost compared with Total Potential Revenue
    '''
    df['term_years'] = df['term'].map(lambda x: 5 if x == '60 months' else 3) # Creates new column converting term into numeric years
    df['loan_w_interest'] = round(df['loan_amount'] * (1 + (df['int_rate'] / 100)) ** df['term_years'], 2) # Creates new column that shows total loan
    total_chargedoff_loans = df.loc[df['loan_status'] == 'Charged Off']['loan_w_interest'].sum() # Sums the total loan value for all charged off loans
    total_payment_chargeoff = round(df.loc[df['loan_status'] == 'Charged Off']['total_payment'].sum(), 2) # Sums the total payment for all charged off loans
    lost_revenue = round(total_chargedoff_loans - total_payment_chargeoff, 2) # THe lost revenue from unpaid loans
    total_revenue = df['loan_w_interest'].sum() # total potential revenue
    # Prints Outputs
    print(f'Total Revenue Lost £{lost_revenue}')
    print(f'Pecentage of Revenue Lost: {round((lost_revenue / total_revenue * 100), 2)}%')

def possible_loss(df):
    '''
    This function shows the potential possible loss if late payments default or charged off

    Args:
        df (dataframe): Dataframe to be analysed
    
    Outputs:
        Total Number of Late Status: Total number of late loans
        Percentage of Members in Late Status: Percentage of late loans against total loans
        Total Lost if Late Loans Charged Off: Total Loss of LAte Loans if no more payments
        TOtal Lost for Late, Default and Charged Off: Total Loss of Late Loans, Default and Charged Off
        Percentage Lost of Total Revenue: Total Lost as a percentage of Total Revenue
    '''
    df['term_years'] = df['term'].map(lambda x: 5 if x == '60 months' else 3) # Creates new column converting term into numeric years
    df['loan_w_interest'] = round(df['loan_amount'] * (1 + (df['int_rate'] / 100)) ** df['term_years'], 2) # Creates new column that shows total loan
    total_late_loans = df.loc[df['loan_status'].isin(['Late (31-120 days)', 'Late (16-30 days)'])]['loan_status'].count() # Counts the number of Late Loans
    late_loans_value = df.loc[df['loan_status'].isin(['Late (31-120 days)', 'Late (16-30 days)'])]['loan_w_interest'].sum() # Sums the total loan value for all late loans
    late_loans_paid = df.loc[df['loan_status'].isin(['Late (31-120 days)', 'Late (16-30 days)'])]['total_payment'].sum() # Sums the total payment for all late loans
    late_loans_loss = round(late_loans_value - late_loans_paid, 2) # The potential loss from all late loans
    total_lost_loan = df.loc[df['loan_status'].isin(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'Charged Off'])]['loan_w_interest'].sum() # Sums the total loan value for late, default and charged off loans
    total_lost_paid = df.loc[df['loan_status'].isin(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'Charged Off'])]['total_payment'].sum() # Sums the total payment for all late, default and charged off loans
    total_lost_value = round(total_lost_loan - total_lost_paid, 2) # The potential loss from all late, default and charged off loans
    # Prints Outputs
    print(f'Total Number of Late Status: {total_late_loans}')
    print(f'Percentage of Members in Late Status: {round(total_late_loans / (df["loan_status"]).count() * 100, 2)}%')
    print(f'Total Lost if Late Loans Charged Off: £{late_loans_loss}')
    print(f'Total Lost for Late, Default and Charged Off: £{total_lost_value}')
    print(f'Percentage Lost of Total Revenue: {round(total_lost_value / df["loan_w_interest"].sum() * 100, 2)}%')
    
def loss_indicators(df):
    '''
    This function gives a correlation matrix to determine what effect each variable has on loan_Status

    Args:
        df (dataframe): Dataframe to be analysed
    
    Outputs
        Correlation Matrix: Correlation matrix where the top row is loan_status
    '''
    # list all potential correlation columns to loan_status
    columns = ['loan_status', 'int_rate', 'grade', 'employment_length', 'annual_inc', 'verification_status', 'purpose', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'open_accounts', 'total_accounts', 'mths_since_last_major_derog', 'policy_code', 'application_type']
    loss_df = df[columns].copy() # Create new dataframe
    # Create dictionary of loan_status transformation ordered from Worst Result to Best (9 is placeholder)
    loan_status_dict = {
                        'Charged Off' : 0,
                        'Default' : 1,
                        'Late (31-120 days)' : 3,
                        'Late (16-30 days)' : 2,    
                        'In Grace Period' : 9,
                        'Does not meet the credit policy. Status:Charged Off' : 9,
                        'Does not meet the credit policy. Status:Fully Paid' : 9,
                        'Current' : 9,                                                                                                                                                                      
                        'Fully Paid' : 4                                               
                    }
    loss_df.replace({'loan_status' : loan_status_dict}, inplace=True) # Use dictionary to convert loan_status to numerical
    # Converts all other string columns to numerical
    for col in columns:
        if type(loss_df[col][0]) == str:
            code, uniques = pd.factorize(loss_df[col], sort=True) # Converts the categorical data to numeric
            loss_df[col] = code # Replaces dataframe columns with new numerical values
    # Keep the relevant rows (Late payments, defaults, charged off and fully paid)
    loss_df = loss_df.loc[df['loan_status'].isin(['Late (31-120 days)', 'Late (16-30 days)', 'Charged Off', 'Default', 'Fully Paid'])]
    # Plot correlation matrix
    plt.matshow(loss_df.corr())
    plt.show()  

def initialise():
    '''
    This function creates, formats, transforms and sets up for plotting

    Returns:
        (dataframe): Raw dataframe before any transformations
        (dataframe): Transformed dataframe that can be plotted
    '''
    loan_payments_file = 'loan_payments.csv' # from created .csv
    transformed_df = transform_data(loan_payments_file) # turns csv into dataframe
    loan_df = transform_dataframe(transformed_df) # transforms csv
    return transformed_df, Plotter(loan_df.df) # returns plottable dataframe

if __name__ == '__main__':
    df, plot_df = initialise()
    sixmnths_state(df)
