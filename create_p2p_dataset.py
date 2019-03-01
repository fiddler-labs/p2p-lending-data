"""
This script takes a raw lending club dataset, and makes it into a processed
version that is more suited to ML modeling.
"""


import logging
import pathlib
import re

import numpy as np
import pandas as pd
import yaml


LOG_LEVEL = logging.INFO
LOG = logging.getLogger(__name__)

RAW_DATSET_FILENAME = 'accepted_2007_to_2018Q3.csv.gz'
SCRIPT_DIR = pathlib.Path(__name__).parent
BROWSE_COLUMN_FILEPATH = SCRIPT_DIR / 'browse_cols_from_data_dictionary.txt'
RAW_DATA_DIR = SCRIPT_DIR / 'raw_data'
PROCESSED_DATA_DIR = SCRIPT_DIR / 'p2p_loans_470k/'

# defines which variables are categorical (we omit loan_status, since
# we'll process it later)
IMPORT_DTYPES = {
    'term': 'category',
    'grade': 'category',
    'sub_grade': 'category',
    'emp_length': 'category',
    'home_ownership': 'category',
    'purpose': 'category',
    'addr_state': 'category',
    'application_type': 'category',
    'disbursement_method': 'category',
    'debt_settlement_flag': 'category',
    'settlement_status': 'category',
}

DATETIME_FIELDS = ['issue_d', 'settlement_date', 'earliest_cr_line']

LABEL_FIELDS = [
    # target variable for creditworthiness modeling
    'loan_status',

    # additional loan metadata
    'issue_d',
    'zip_code_prefix',
    'grade',
    'sub_grade',

    # information on dollars received from the investment
    'installment',
    'int_rate',
    'collection_recovery_fee',
    'recoveries',
    'debt_settlement_flag',
    'settlement_amount',
    'settlement_date',
    'settlement_percentage',
    'settlement_status',
    'settlement_term',
    'total_pymnt',
    'total_pymnt_inv',
    'total_rec_int',
    'total_rec_late_fee',
    'total_rec_prncp',
]


def main():
    """
    Imports the raw data, keeps only the columns of interest for
    creditworthiness, and filters down to 3-year single-applicant loans that
    were issued at least 3 years ago and are now either fully-paid or
    charged-off.

    We also drop all loans issued before June 01 2012, since a lot of fields
    were not collected/reported for loans issued before then.

    We drop all columns which, in our best guess, would not be available to
    investors when they are evaluating the creditworthiness of an applicant.
    This is primarily accomplished by keeping only the columns that the data
    dictionary lists as "browse" columns, i.e. information that an investor
    browsing through loans would see.

    Joint-application loans, like five-year loans, are removed. This leads to
    a number of fields (those with the prefix 'sec_' being full of only missing
    values. We drop these fields and all others that contain no information.
    """
    LOG.info('Processing p2p loan data!')
    dataset_filepath = RAW_DATA_DIR / RAW_DATSET_FILENAME

    # address slight mis-formatting in the input data
    skip_rows = get_skip_rows(dataset_filepath)

    LOG.info('Loading raw data from CSV...')
    df = pd.read_csv(dataset_filepath, low_memory=False, index_col='id',
                     skiprows=skip_rows, parse_dates=DATETIME_FIELDS,
                     infer_datetime_format=True, dtype=IMPORT_DTYPES)

    LOG.info('Processing the data...')
    filter_time_range(df)
    filter_to_3yr_loans(df)
    remove_joint_loans(df)
    remove_unclean_loans(df)
    merge_fico_range_fields(df)

    # remove the 'xx' from a zipcode prefix in format '123xx' but leave
    # as a string type to preserve leading zeros
    df['zip_code_prefix'] = df['zip_code'].str[:3]
    df.drop(columns=['zip_code'], inplace=True)

    remove_uninformative_fields(df)
    remove_non_predictive_features(df)

    # get set of columns available at time of loan browsing
    loan_browse_cols = get_loan_browse_columns()
    all_columns = set(df.columns)
    LOG.debug(f'Columns in browse but not CSV:'
              f'\n{loan_browse_cols - all_columns}')

    # determine which columns are features, accounting for the merging of
    # the fico range fields
    features = (all_columns & loan_browse_cols) | {'fico_range_midpoint'}

    # drop irrelevant columns (not features or labels)
    drop_cols = all_columns - (features | set(LABEL_FIELDS))
    LOG.debug(f'Columns dropped because not feature or label:\n{drop_cols}')
    df.drop(columns=drop_cols, inplace=True)

    # split into train and test set
    train_df, test_df = train_test_split(df)
    del df

    # split features from labels
    train_features = train_df.drop(columns=LABEL_FIELDS)
    test_features = test_df.drop(columns=LABEL_FIELDS)
    train_labels = train_df.loc[:, LABEL_FIELDS].copy()
    test_labels = test_df.loc[:, LABEL_FIELDS].copy()
    LOG.debug(f'Feature fields:\n{set(train_features.columns)}')
    del train_df, test_df

    # update the categorical dtypes in both train and test so that only
    # categories present in the training data are kept
    # (otherwise we'd be leaking categorical information from the test set
    #  into the training set)
    for field in train_features.select_dtypes('category'):
        train_features[field].cat.remove_unused_categories(inplace=True)
        test_features[field] = test_features[field].astype(
            train_features[field].dtype)

    # write to disk
    LOG.info('Writing the processed data to disk...')
    train_features.to_csv(
        PROCESSED_DATA_DIR / 'train' / 'train_features.csv.zip')
    train_labels.to_csv(PROCESSED_DATA_DIR / 'train' / 'train_labels.csv.zip')
    test_features.to_csv(PROCESSED_DATA_DIR / 'test' / 'test_features.csv.zip')
    test_labels.to_csv(PROCESSED_DATA_DIR / 'test' / 'test_labels.csv.zip')

    # also export the data schema as yaml files
    LOG.info('Writing schemas to disk as well')
    create_schema_file(train_features,
                       PROCESSED_DATA_DIR / 'feature_schema.yaml')
    create_schema_file(train_labels,
                       PROCESSED_DATA_DIR / 'label_schema.yaml')

    LOG.info('Done!')


def filter_time_range(df, start_after='2012-06-01', end_before='2015-09-01'):
    """Removes loans before 2012-06-01 and after 2015-09-01. Loans before
    June 2012 are missing a lot of information, and loans on or after
    September 2015 have not had a chance to complete their 3-year term."""
    LOG.info(f'Removing rows for loans issued before {start_after} or issued '
             f'on or after {end_before}')
    df.drop(index=df[df['issue_d'] >= end_before].index, inplace=True)
    df.drop(index=df[df['issue_d'] < start_after].index, inplace=True)


def filter_to_3yr_loans(df):
    """Removes rows associated with loans that are for a 5yr term.
    Removes the now-meaningless 'term' column."""
    LOG.info('Removing loans that are not for a 3-year term')
    df.drop(index=df[df['term'] != ' 36 months'].index, inplace=True)
    df.drop(columns=['term'], inplace=True)


def remove_joint_loans(df):
    """Remove loans that fall under a joint application.
    Removes the now-meaningless 'application_type' column."""
    LOG.info('Removing joint-application loans')
    joint_loans_index = df[df['application_type'] != 'Individual'].index
    df.drop(index=joint_loans_index, inplace=True)
    df.drop(columns=['application_type'], inplace=True)


def remove_unclean_loans(df):
    """Removes loans that even after 3 years are not either paid or
    charged-off and loans for which the funded amount does not equal the loan
    amount.

    One could argue that dropping loans that were extended beyond the original
    term would skew the distribution of the dataset, but these cases are
    very rare in the data, so dropping them has very little impact of the
    overall distribution of the data. It just makes for a cleaner, truly binary
    target variable in creditworthiness modeling.
    """
    LOG.info('Removing loans that are incomplete at end of term or not '
             'fully funded')
    df.drop(
        index=df[
            ~df['loan_status'].isin({'Fully Paid', 'Charged Off'})].index,
        inplace=True)
    df['loan_status'] = df['loan_status'].astype('category')
    not_fully_funded = df['loan_amnt'] != df['funded_amnt']
    df.drop(index=df[not_fully_funded].index, inplace=True)
    df.drop(columns=['funded_amnt'], inplace=True)


def remove_uninformative_fields(df):
    """After heavy cleaning, some of the fields left in the dataset track
    information that was never recorded for any of the loans in the dataset.
    These fields have only a single unique value or are all NaN, meaning
    that they are entirely uninformative. We drop these fields."""
    is_single = df.apply(lambda s: s.nunique()).le(1)
    single = df.columns[is_single].tolist()
    LOG.info('Dropping useless fields')
    LOG.debug(f'Useless fields dropped:\n{single}')
    df.drop(columns=single, inplace=True)


def merge_fico_range_fields(df):
    """The data includes the applicant's fico score as a range between two
    fields. Here we replace the range with a single value: the midpoint of
    the range."""
    fico_fields = ['fico_range_high', 'fico_range_low']
    df['fico_range_midpoint'] = df[fico_fields].mean(axis=1)
    df.drop(columns=fico_fields, inplace=True)


def remove_non_predictive_features(df):
    """The features include some fields that are not predictive, like URL and
    the 'initial_list_status' (which is randomized
    (see https://blog.lendingclub.com/investor-updates-and-enhancements/))

    Here we drop those fields."""
    non_ml_features = {'url', 'initial_list_status'}
    df.drop(columns=non_ml_features, inplace=True)


def train_test_split(df, test_start='2015-04'):
    """Splits dataset approx 75% train 25% test by making everything issued on
    or after 2015-04 a test set row.
    """
    is_in_test_set = df['issue_d'] >= test_start
    df_train, df_test = df[~is_in_test_set].copy(), df[is_in_test_set].copy()
    return df_train, df_test


def create_schema_file(df, output_path):
    """Writes a map from column name to column datatype to a YAML file for a
    given dataframe. The schema format is as keyword arguments for the pandas
    `read_csv` function."""
    # ensure file exists
    output_path = pathlib.Path(output_path)
    output_path.touch(exist_ok=True)

    # get dtypes schema
    datatype_map = {}
    datetime_fields = []
    for name, dtype in df.dtypes.iteritems():
        if 'datetime64' in dtype.name:
            datatype_map[name] = 'object'
            datetime_fields.append(name)
        else:
            datatype_map[name] = dtype.name

    schema = dict(dtype=datatype_map, parse_dates=datetime_fields,
                  index_col='id')
    # write to YAML file
    with output_path.open('w') as yaml_file:
        yaml.dump(schema, yaml_file)


def get_skip_rows(path_to_dataset):
    """Finds corrupted rows to skip in import.
    https://www.kaggle.com/lukemerrick/demonstrating-corrupted-mal-formed-rows
    """
    LOG.info('Finding misformatted rows to skip on import...')
    df = pd.read_csv(path_to_dataset, usecols=['id'], low_memory=False)
    num_id = pd.to_numeric(df['id'], errors='coerce')
    skip_rows = np.where(num_id.isna())[0] + 1
    LOG.debug(f'Misformatted row indices:\n{skip_rows}')
    del df

    return skip_rows


def camel_to_snake(variable_name):
    """CamelCase to snake_case conversion.

    Source: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case  # noqa
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', variable_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_loan_browse_columns():
    """Returns a list of columns that represent data available when browsing
    for a loan on Lending Club, i.e. information that is generally available
    before a loan is funded."""

    # read in a list of columns taken from the data dictionary
    with BROWSE_COLUMN_FILEPATH.open() as text_file:
        loan_browse_cols = [name.strip() for name in text_file.readlines()]

    # convert from camelCase names in the data dictionary to snake_case names
    # found in the CSV file
    loan_browse_cols = {camel_to_snake(x) for x in loan_browse_cols}

    # manually fix cases where the camel_to_snake guessed wrong
    loan_browse_cols = loan_browse_cols - {'acc_open_past24_mths',
                                           'inq_last6_mths',
                                           'sec_app_inq_last_6mths',
                                           'delinq2_yrs',
                                           'mths_since_most_recent_inq'}
    loan_browse_cols = loan_browse_cols | {'acc_open_past_24mths',
                                           'inq_last_6mths',
                                           'sec_app_inq_last_6mths',
                                           'delinq_2yrs',
                                           'mths_since_recent_inq'}

    return loan_browse_cols


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=LOG_LEVEL, format=log_fmt)
    main()
