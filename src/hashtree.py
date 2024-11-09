from itertools import combinations
from tqdm import tqdm
import numpy as np

from utils import Utils

class HashTreeNode:
    def __init__(self, is_leaf=True, max_leaf_size=5, node_level=0, hash_index=-1):
        self.is_leaf = is_leaf
        self.node_level = node_level
        self.max_leaf_size = max_leaf_size
        self.k = 0
        self.hash_index = hash_index
        self.children = {}
        self.itemsets = {}

    def __hash_func(self, item):
        return (item - 1) % self.k      
    
    def add_to_node(self, candidate):
        if self.k == 0:     
           self.k = len(candidate)
                   
        if len(self.itemsets) >= self.max_leaf_size and self.node_level < self.k:
            self.__split_node()

        candidate = sorted(candidate)

        if self.is_leaf:
            self.itemsets[tuple(candidate)] = 0
        else:
            item = candidate[self.node_level]
            item_hash = self.__hash_func(item)
            if item_hash not in self.children:
                self.children[item_hash] = HashTreeNode(max_leaf_size=self.max_leaf_size, node_level=(self.node_level + 1), hash_index=item_hash)
                
            self.children[item_hash].add_to_node(candidate)

    def __split_node(self):
        self.is_leaf = False

        for itemset in list(self.itemsets.keys()):
            self.itemsets.pop(itemset)
            item_hash = self.__hash_func(itemset[self.node_level])

            if item_hash not in self.children:
                self.children[item_hash] = HashTreeNode(max_leaf_size=self.max_leaf_size, node_level=(self.node_level + 1), hash_index=item_hash)
                
            self.children[item_hash].add_to_node(itemset)        

    def __recursive_support_counting_helper(self, transaction, depth=0):     
        transaction_len = len(transaction)
        if self.is_leaf:
            for itemset, _  in self.itemsets.items():
                if all(item in transaction for item in itemset):
                    self.itemsets[itemset] += 1
        else:
            min_i = max([self.node_level, depth])
            max_i = transaction_len - (self.k - self.node_level) + 1
            for i in range(min_i, max_i):
                
                item_hash = self.__hash_func(transaction[i])
                
                if item_hash in self.children:
                    self.children[item_hash].__recursive_support_counting_helper(transaction, depth=i+1)

    def increment_support_recursive(self, transactions):
        progress_bar = tqdm(total=len(transactions), desc="Support Counting - Recursive")
        for transaction in transactions:
            transaction_len = len(transaction)
            if transaction_len < self.k:
                progress_bar.update(1)
                continue

            self.__recursive_support_counting_helper(transaction)
            
            progress_bar.update(1)
        progress_bar.close()

    def increment_support_iterative(self, transactions):     
        progress_bar = tqdm(total=len(transactions), desc="Support Counting - Iterative")
        for transaction in transactions:
            transaction_len = len(transaction)
            if transaction_len < self.k:
                progress_bar.update(1)
                continue

            hashed_items = {item: self.__hash_func(item) for item in transaction}
            itemsets = combinations(transaction, self.k)
            for itemset in itemsets:
                node = self
                for item in itemset:
                    item_hash = hashed_items[item]

                    if item_hash in node.children:
                            node = node.children[item_hash]
                    else:                        
                        break
    
                if node.is_leaf and (itemset in node.itemsets.keys()):
                    node.itemsets[itemset] += 1
            
            progress_bar.update(1)
        progress_bar.close()


class HashTree:
    def __init__(self, max_leaf_size=5):
        self.root: HashTreeNode = None
        self.candidate_size = 0
        self.max_leaf_size = max_leaf_size
        self.support_counts = {}

    def insert_candidates(self, candidates):
        progress_bar = tqdm(total=len(candidates), desc="HashTree Insertion")

        for candidate in candidates:

            if self.candidate_size == 0:
                self.candidate_size = len(candidate)

            if not self.root:
                self.root = HashTreeNode(max_leaf_size=self.max_leaf_size)

            self.root.add_to_node(candidate)

            progress_bar.update(1)
        progress_bar.close()

    def count_supports(self, transactions, use_recursive_method=False):
        if use_recursive_method:
            self.root.increment_support_recursive(transactions)
        else:
            self.root.increment_support_iterative(transactions)

    def count_supports_PCY_MultiHash(self, transactions, hash_func_seeds, table_size=1_000, min_sup=0, use_recursive_method=False, target_itemset_size=2):
        tot_hash_funcs = len(hash_func_seeds)
        table = np.zeros((tot_hash_funcs, table_size), dtype=np.int32)        

        self.count_supports(transactions, use_recursive_method)

        # if current candidate itemset size is the target one, no need to build the (k+1)-itemset bitmap
        if self.root.k == target_itemset_size:
            return table >= min_sup

        progress_bar = tqdm(total=len(transactions), desc="BitMap Construction")
        for transaction in transactions:

            if len(transaction) < (self.root.k + 1):
                progress_bar.update(1)
                continue
            
            itemset_hashes = np.array([hash(itemset) for itemset in combinations(transaction, (self.root.k + 1))])

            hash_table_indices  = Utils.PCY_MultiHash_hash_func(itemset_hash=itemset_hashes, seeds=hash_func_seeds, m=table_size)
            for i in range(len(hash_func_seeds)):
                unique_indices, counts = np.unique(hash_table_indices[i], return_counts=True)
                table[i][unique_indices] += counts
            progress_bar.update(1)
        progress_bar.close()      

        return table >= min_sup

    def get_supports(self, node=None, min_sup=0):
        if not node:
            node = self.root
        
        if node.is_leaf:
            itemsets = node.itemsets.items()
            if min_sup > 0:
                itemsets = dict((k, v) for k, v in itemsets if v >= min_sup)
            
            self.support_counts.update(itemsets)
            return node.itemsets
        else:
            for i, child in node.children.items():
                self.get_supports(child, min_sup=min_sup)

        return self.support_counts

    def clear(self):
        self.root: HashTreeNode = None
        self.candidate_size = 0
        self.support_counts = {}