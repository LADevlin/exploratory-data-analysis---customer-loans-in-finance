import yaml
import pandas as pd
from sqlalchemy import create_engine, engine


class RDSDatabaseConnector:
    
    def __init__(self, credentials):
        self.host = credentials['RDS_HOST']
        self.password = credentials['RDS_PASSWORD']
        self.user = credentials['RDS_USER']
        self.database = credentials['RDS_DATABASE']
        self.port = credentials['RDS_PORT']
    
    def initialise_sql(self):
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
        df = pd.read_sql_table('loan_payments', self.initialise_sql())
        return df
    
    def create_csv(self):
        self.create_dataframe().to_csv('loan_payments.csv')
        



def load_credentials():
    # open credentials.yaml in read mode
    with open('credentials.yaml', mode='r') as file:
        # load yaml as dictionary
        credentials_dict = yaml.safe_load(file)
        return credentials_dict


def load_csv():
    df = pd.read_csv('loan_payments.csv')
    return df

credentials = load_credentials()

#test = RDSDatabaseConnector(credentials)
#test.create_csv()

df = load_csv()

print(df.info())

