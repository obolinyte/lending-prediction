# LendingClub's loan data - EDA & ML

LendingClub [dataset]( https://storage.googleapis.com/335-lending-club/lending-club.zip) contains customer (borrower) and loan information for accepted loans as well as for rejected loan applications over the period ranging from 2007 to 2018. Data is distributed into two files:

- Accepted loan data contains more than 2M observations (customers) and has 151 features related with borrower personal and financial information, credit rating and credit history, current loan status, payment details, etc.
- Rejected loan data has more than 27M observations (customers) and only 9 features: application date, amount requested, loan purpose, policy code, borrower risk (fico) score and debt-to-income ratio, state, zip code and employment length.

The period chosen for this project is **from 2017 to 2018**.

- Based on the project requirements there are 4 ML goals (targets):
    1. Classify loan applications into accepted or rejected
    2. Predict the grade for the loan
    3. Predict the sub-grade for the loan
    4. Predict interest rate for the loan
- EDA is structured according to the ML targets:
  1. The first part of EDA is focused on investigating and validating each feature impact on distinguishing loan application as accepted or rejected. 
  2. In the second part of EDA, loan grade, sub-grade and interest rate are explored and analyzed to understand feature importance and data relationships/associations.

## Exploratory Data Analysis

EDA and statistical inference part of this project can be found in [here](./notebooks/335_EDA.ipynb).

## Machine Learning

ML part of this project can be found [here](./notebooks/335_ML.ipynb).

## API

API part of this project can be found [here](./loan_prediction_api).

## Inspiration

Learning @TuringCollege