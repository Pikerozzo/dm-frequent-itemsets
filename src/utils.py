import pandas as pd
import resource
import numpy as np
import psutil

class Utils:    
    def PCY_MultiHash_hash_func(itemset=[], itemset_hash=np.array([]), seeds=np.array([]), m=100):
        if len(itemset_hash) == 0:
            itemset_hash = np.array([hash(tuple(i)) for i in itemset])
        
        return np.array([(itemset_hash * s) % m for s in seeds],dtype=int)

    def get_item_names(itemsets, items:pd.DataFrame):
        return [items.loc[items['product_id'].isin(itemset[0])]['product_name'].values for itemset in itemsets]
    
    def get_pcy_hashtable_size(megabytes=1000, memory_percentage=1):
        soft, _ = resource.getrlimit(resource.RLIMIT_AS)        
        available_memory = 0
        
        if soft == resource.RLIM_INFINITY:
            available_memory = psutil.virtual_memory().available
        else:
            available_memory = soft

        if megabytes > 0 and (megabytes * 2**20) < available_memory:
            target_memory_size = megabytes * 2**20   
        else:
            if memory_percentage > 1 or memory_percentage < 0:
                memory_percentage = 0.25
            target_memory_size = int(memory_percentage * available_memory)
        
        max_num_elements = target_memory_size // np.dtype(np.int32).itemsize
        
        return max_num_elements    
    
    def remove_small_transactions(transactions, k):
        return [transaction for transaction in transactions if len(transaction) > k]