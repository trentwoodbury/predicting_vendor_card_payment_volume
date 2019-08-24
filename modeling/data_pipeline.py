'''
This is a class that contains a series of methods required to get the data into
a proper train-test split. These methods should be run sequentially.

A lot of this code is copied directly over from EDA/EDA.ipynb
'''

# ################################# IMPORTS ####################################
import datetime
import numpy as np
import pandas as pd

# ################################# CONSTANTS ##################################
CONVERSIONS = {
    'EUR': 1,
    'GBP': 1.14,
    'DKK': 0.13,
    'NOK': 0.10,
    'SEK': 0.10
}

# ################################# FUNCTIONS #################################
class data_pipeline():

    def __init__(self):
        pass

    def read_in_and_join_data(self):
        '''
        Reads in the training/label data and joins together organization start dates
        with the other training data.

        RETURNS a dataframe of the joined data.
        '''
        # read in and join data
        organization_names = pd.read_csv('../data/organization_ids.csv')
        payment_data = pd.read_csv('../data/payment_ids.csv')
        joined_df = organization_names.merge(payment_data, on='organization_id', how='inner')
        self.joined_df = joined_df


    def create_time_to_purchase(self):
        '''
        Convert dates from strings into timestamps and add in time from organization
        creation to purchase.

        RETURNS updated dataframe
        '''
        joined_df = self.joined_df
        joined_df['organization_created'] = (
            joined_df['organization_created']
            .apply(lambda x: datetime.datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S'))
        )
        joined_df['timestamp'] = (
            joined_df['timestamp']
            .apply(lambda x: datetime.datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S'))
        )
        joined_df['days_from_creation_to_purchase'] = [
            (row['timestamp'] - row['organization_created']).days for _, row in joined_df.iterrows()
        ]
        self.joined_df = joined_df


    def convert_currencies_to_euros(self):
        '''
        Converts all currencies into Euros and drops original currency values.

        RETURNS updated dataframe.
        '''
        joined_df = self.joined_df
        joined_df['amount_in_euros'] = [
            row['amount'] * CONVERSIONS[row['currency_id']] for _, row in joined_df.iterrows()
        ]
        joined_df.drop('amount', inplace=True, axis=1)
        self.joined_df = joined_df


    def creating_period_labels(self):
        '''
        Creates labels for the training perios and for when we're in evaluation month
        1-3. Adds these labels to the dataframe. Also filters out all rows that contain
        purchases occuring outside of the training/3-month window.

        RETURNS updated dataframe.
        '''
        joined_df = self.joined_df
        which_month = []
        for _, row in joined_df.iterrows():
            # month 0 is the training period.
            if row['days_from_creation_to_purchase'] <= 14:
                which_month.append(0)
            elif row['days_from_creation_to_purchase'] <= 44:
                which_month.append(1)
            elif row['days_from_creation_to_purchase'] <= 74:
                which_month.append(2)
            elif row['days_from_creation_to_purchase'] <= 104:
                which_month.append(3)
            else:
                which_month.append(4)

        joined_df['which_month'] = which_month
        joined_df = joined_df.loc[joined_df['which_month'] < 4, :]

        self.joined_df = joined_df


    def filter_out_invalid_transactions(self):
        '''
        RETURNS valid_df, new dataframe with only valid transactions.
        '''
        valid_df = self.joined_df.loc[self.joined_df.payment_type == 'Valid', :]
        self.valid_df = valid_df


    def group_dataset_by_order_and_month(self):
        '''
        Groups together the valid_df taking the transaction sum and count by
            month, vendor pair.
        RETURNS grouped_df, pandas dataframe
        '''
        grouped_df = (
        self.valid_df
        .loc[:, ['organization_id', 'amount_in_euros', 'which_month']]
        .groupby(['organization_id', 'which_month']).agg(['sum', 'count'])
        .reset_index()
        )
        grouped_df.columns = ['organization_id', 'which_month', 'cpv', 'transaction_count']

        self.grouped_df = grouped_df


    def filter_and_impute_organization_data(self):
        '''
        This function filters out organizations that don't have any transactions during the
        training period (which_month = 0). This function also fills in 0 for any months
        that are missing for a given organization.

        RETURNS filtered_grouped_df, pandas dataframe with columns [organization_id, which_month, amount_in_euros]
            and a new row for each month for the organization_id.
        '''
        grouped_df = self.grouped_df
        filtered_grouped_df = grouped_df.copy(deep=True)

        # create values for first row
        current_organization = grouped_df.loc[0, 'organization_id']
        current_month = -1
        skip_org = False

        for idx, row in grouped_df.iterrows():

            if row.organization_id == current_organization:

                if skip_org:
                    continue

                if row.which_month == current_month + 1:
                    pass

                # Case when there was a month (or two) with no purchases
                else:
                    for month in range(current_month + 1, row.which_month + 1):
                        new_row = pd.DataFrame(
                            data={
                                'organization_id': [row.organization_id],
                                'which_month': [month],
                                'cpv': [0],
                                'transaction_count': [0]
                            }
                        )
                        filtered_grouped_df = filtered_grouped_df.append(new_row)

                current_month += 1

            # case when we're switching new a new organization id
            else:

                # Fill in the remainder of missing months with 0's
                if current_month < 3 and not skip_org:

                    for month in range(current_month + 1, 4):
                        new_row = pd.DataFrame(
                            data={
                                'organization_id': [current_organization],
                                'which_month': [month],
                                'cpv': [0],
                                'transaction_count': [0]
                            }
                        )
                        filtered_grouped_df = filtered_grouped_df.append(new_row)

                # Reset all the variables
                current_organization = row.organization_id
                current_month = 0
                skip_org = False

                # Filter out companies with no training data
                if row.which_month != 0:

                    filtered_grouped_df = (
                        filtered_grouped_df
                        .loc[filtered_grouped_df['organization_id'] != current_organization, :]
                    )
                    skip_org = True

                else:
                    pass

        # Finally, let's handle the last row. We're just going to remove it for simplicity.
        if idx == grouped_df.index[-1]:
            filtered_grouped_df = (
                filtered_grouped_df
                .loc[filtered_grouped_df['organization_id'] != current_organization, :]
            )

        filtered_grouped_df = (
            filtered_grouped_df
            .groupby(['organization_id', 'which_month'])
            .sum()
            .reset_index()
        )
        self.filtered_grouped_df = filtered_grouped_df


    def split_up_data(self):
        '''
        This function splits our data into 4 pieces:
            (1) training datasets 1 & 2 (for grouped and ungrouped features)
            (2) month 1
            (3) month 2
            (4) month 3
        INPUT
            filtered_grouped_df: pandas df, output of the function filter_and_impute_organization_data()
            valid_df: pandas df, output of the function filter_out_invalid_transactions()
        RETURNS
            training_dataset_grouped: pandas dataframe of data from the training period.
                This contains the grouped features and labels.
            training_dataset_ungrouped: pandas dataframe of the ungrouped data from the training period.
        '''
        filtered_grouped_df = self.filtered_grouped_df
        non_grouped_df = self.valid_df

        month_1 = filtered_grouped_df.loc[filtered_grouped_df['which_month'] == 1, ['cpv', 'organization_id']]
        month_2 = filtered_grouped_df.loc[filtered_grouped_df['which_month'] == 2, ['cpv', 'organization_id']]
        month_3 = filtered_grouped_df.loc[filtered_grouped_df['which_month'] == 3, ['cpv', 'organization_id']]
        month_1.columns, month_2.columns, month_3.columns = ['month_1_cpv', 'organization_id'], ['month_2_cpv', 'organization_id'], ['month_3_cpv', 'organization_id']

        training_dataset_grouped = filtered_grouped_df.loc[filtered_grouped_df['which_month'] == 0, ['organization_id', 'cpv', 'transaction_count']]
        training_dataset_grouped = training_dataset_grouped.merge(month_1, how='left', on='organization_id')
        training_dataset_grouped = training_dataset_grouped.merge(month_2, how='left', on='organization_id')
        training_dataset_grouped = training_dataset_grouped.merge(month_3, how='left', on='organization_id')

        # Fill in missing month labels with 0's since they are null due to no transactions.
        training_dataset_grouped = training_dataset_grouped.fillna(0)

        training_dataset_ungrouped = non_grouped_df.loc[non_grouped_df['which_month'] == 0, ['organization_id', 'currency_id', 'device_type']]

        self.training_dataset_grouped, self.training_dataset_ungrouped = training_dataset_grouped, training_dataset_ungrouped


    def format_ungrouped_training_dataset(self):
        '''
        This gets training_dataset_ungrouped into dummy variables so that it is ready for the model.
        RETURNS training_dataset_ungrouped, pandas dataframe
        '''
        # drop duplicates
        training_dataset_ungrouped = self.training_dataset_ungrouped.drop_duplicates()
        dummy_vars = pd.get_dummies(training_dataset_ungrouped.loc[:, ['currency_id', 'device_type']], prefix=['currency_', 'device_'])
        training_dataset_ungrouped = pd.concat((training_dataset_ungrouped.loc[:, ['organization_id']], dummy_vars), axis=1)

        self.training_dataset_ungrouped = training_dataset_ungrouped


    def merge_training_data(self):
        '''
        Merges together grouped and ungrouped datasets. Imputes missing
        values.
        RETURNS dataset, pandas dataframe of our data with the labels.
        '''
        dataset = self.training_dataset_grouped.merge(
            self.training_dataset_ungrouped,
            how='left',
            on='organization_id'
        )
        dataset.fillna(0) # impute 0's for missing currency/device items

        self.dataset = dataset


    def perform_full_data_pipeline(self):
        '''
        performs all the sequntial steps necessary to get the training and test data.
        RETURNS
            training_dataset, pandas dataframe of training data
            month_1 - month_3: pandas dataframe of organization-ids and their associated
                CPV values for that month.
        '''
        self.read_in_and_join_data()
        self.create_time_to_purchase()
        self.convert_currencies_to_euros()
        self.creating_period_labels()
        self.filter_out_invalid_transactions()
        self.group_dataset_by_order_and_month()
        self.filter_and_impute_organization_data()
        self.split_up_data()
        self.format_ungrouped_training_dataset()
        self.merge_training_data()
