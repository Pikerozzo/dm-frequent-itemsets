import pandas as pd
import ast
import time
import os
from datetime import timedelta

from utils import *
from algorithms import *


def load_data(transaction_filepath=os.path.join(os.path.curdir, '..', 'data', 'transactions_train.csv'), items_filepath=os.path.join(os.path.curdir, '..', 'data', 'products.csv'), subsample_size=100_000, subsample_seed=1492):
    try:
        all_transactions = pd.read_csv(transaction_filepath)
    except FileNotFoundError:
        print(f'ERROR: could not find data file(s) with path \'{transaction_filepath}\'. Stopping . . .')
        exit()
    try:
        all_items = pd.read_csv(items_filepath)
    except FileNotFoundError:
        print(f'ERROR: could not find data file(s) with path \'{items_filepath}\'. Stopping . . .')
        all_items = None

    if subsample_size > 0:
        transactions = all_transactions.sample(subsample_size, random_state=subsample_seed)
    else:
        transactions = all_transactions
    transactions['product_id'] = transactions['product_id'].apply(ast.literal_eval)
    transactions['product_id'] = transactions['product_id'].apply(sorted)

    return transactions, all_items


def main():  
    # Load transaction dataset  
    transactions, items = load_data(subsample_size=100_000)
    
    # Create the algorithm instance
    algorithm = MultiHash(tot_megabytes=500, tot_hash_tables=2)

    # Find frequent pairs 
    algorithm.find_frequent_itemsets(transactions=transactions.product_id, itemset_size=2, min_sup_norm=0.01)

    # Print some results
    tot_freq_items = len(algorithm.frequent_itemsets)
    print(f'Found {tot_freq_items} frequent itemsets with size {algorithm.itemset_size}')

    # Print the top 5 frequent itemsets
    algorithm.frequent_itemsets = sorted(algorithm.frequent_itemsets, key=lambda x: x[1], reverse=True)
    
    if items is None:
        for i in range(min(5, tot_freq_items)):
            print(f'{algorithm.frequent_itemsets[i][0]} has support {algorithm.frequent_itemsets[i][1]}')
    else:
        itemsets_with_names = Utils.get_item_names(algorithm.frequent_itemsets, items)    
        for i in range(min(5, tot_freq_items)):        
            print(f'{itemsets_with_names[i]} has support {algorithm.frequent_itemsets[i][1]}')


if __name__ == '__main__':
    start = time.time()
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        print('Stopping . . .\n')
    
    total_time = time.time() - start
    print(f'Total exec. time = {str(timedelta(seconds=total_time))[:-3]}')