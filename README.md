# Exploratory Data Analysis - Customer Loans in Finance

## Introduction

This project is a practice study on a real world example:
- Imagine I am working for a large financial institution, where managing loans is a critical component of business operations.
- To ensure informed decisions are made about loan approvals and risk is efficiently managed, my task is to gain a comprehensive understanding of the loan portfolio data.
- My task is to perform exploratory data analysis on the loan portfolio, using various statistical and data visualisation techniques to uncover patterns, relationships, and anomalies in the loan data.
- This information will enable the business to make more informed decisions about loan approvals, pricing, and risk management.
- By conducting exploratory data analysis on the loan data, I aim to gain a deeper understanding of the risk and return associated with the business' loans.
- Ultimately, my goal is to improve the performance and profitability of the loan portfolio.

## Usage

### create_sql_class

Creates a class to connect to the RDS with load_credentials (below) output as an argument

### load_credentials

Converts the YAML file into a dictionary, intended for use to gather credentials to connect with RDS

### load_csv

Converts the csv file into a pandas dataframe, the csv file will be generated from the RDSDatabaseConnector Class

### .initialise_sql

Converts the class arguments into a usable URL, creates an sql engine from this url

### .create_dataframe

Converts the gathered SQL file into a pandas dataframe

### .create_csv

Creates a .csv file from the pandas dataframe

## Full loans dataset schema

- **id**: unique id of the loan
- **member_id**: id of the member to took out the loan
- **loan_amount**: amount of loan the applicant received
- **funded_amount**: The total amount committed to the loan at the point in time 
- **funded_amount_inv**: The total amount committed by investors for that loan at that point in time 
- **term**: The number of monthly payments for the loan
- **int_rate**: Interest rate on the loan
- **instalment**: The monthly payment owned by the borrower
- **grade**: LC assigned loan grade
- **sub_grade**: LC assigned loan sub grade
- **employment_length**: Employment length in years.
- **home_ownership**: The home ownership status provided by the borrower
- **annual_inc**: The annual income of the borrower
- **verification_status**: Indicates whether the borrowers income was verified by the LC or the income source was verified
- **issue_date:** Issue date of the loan
- **loan_status**: Current status of the loan
- **payment_plan**: Indicates if a payment plan is in place for the loan. Indication borrower is struggling to pay.
- **purpose**: A category provided by the borrower for the loan request.
- **dti**: A ratio calculated using the borrowerâ€™s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowerâ€™s self-reported monthly income.
- **delinq_2yr**: The number of 30+ days past-due payment in the borrower's credit file for the past 2 years.
- **earliest_credit_line**: The month the borrower's earliest reported credit line was opened
- **inq_last_6mths**: The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
- **mths_since_last_record**: The number of months since the last public record.
- **open_accounts**: The number of open credit lines in the borrower's credit file.
- **total_accounts**: The total number of credit lines currently in the borrower's credit file
- **out_prncp**: Remaining outstanding principal for total amount funded
- **out_prncp_inv**: Remaining outstanding principal for portion of total amount funded by investors
- **total_payment**: Payments received to date for total amount funded
- **total_rec_int**: Interest received to date
- **total_rec_late_fee**: Late fees received to date
- **recoveries**: post charge off gross recovery
- **collection_recovery_fee**: post charge off collection fee
- **last_payment_date**: Last month payment was received
- **last_payment_amount**: Last total payment amount received
- **next_payment_date**: Next scheduled payment date
- **last_credit_pull_date**: The most recent month LC pulled credit for this loan
- **collections_12_mths_ex_med**: Number of collections in 12 months excluding medical collections
- **mths_since_last_major_derog**: Months since most recent 90-day or worse rating
- **policy_code**: publicly available policy_code=1 new products not publicly available policy_code=2
- **application_type**: Indicates whether the loan is an individual application or a joint application with two co-borrowers
