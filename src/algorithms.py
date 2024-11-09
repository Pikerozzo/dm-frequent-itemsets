from abc import abstractmethod
import numpy as np
from enum import Enum
import random
from tqdm import tqdm
import itertools
from itertools import combinations
from math import factorial
from collections import Counter, defaultdict
import time
import sympy

from hashtree import HashTree
from utils import Utils
from bitmap import BitMap


class Algorithm:
    def __init__(self):
        self.min_sup = 0
        self.frequent_itemsets = []
        self.itemset_size = 0

    @abstractmethod
    def find_frequent_itemsets(self, transactions, itemset_size=3, min_sup_norm=0.01):
        pass
        
    @abstractmethod
    def get_name(self):
        pass


class BruteForce(Algorithm):
    def __init__(self) -> None:
        super().__init__()

    def find_frequent_itemsets(self, transactions, itemset_size=3, min_sup_norm=0.01):
        self.target_itemset_size = itemset_size
        transaction_len = len(transactions)
        self.min_sup = min_sup_norm * transaction_len
        print(f'Using min_sup={self.min_sup}\n')

        print('Generating candidates . . .')
        # candidate generation - all items, O(N^2)
        items = set(itertools.chain(*transactions))
        candidates = combinations(items, self.target_itemset_size)  
        print('DONE.')              

        tot_candidates = factorial(len(items)) / (factorial(self.target_itemset_size) * factorial(len(items) - self.target_itemset_size))            
        progress_bar = tqdm(total=tot_candidates, desc="Support Counting")
        # support counting -  O(N*M), N = # of transactions, M = # of candidates
        support_counts = Counter()
        for candidate in candidates:
            for transaction in transactions:
                if set(candidate).issubset(transaction):
                    support_counts[candidate] += 1
            progress_bar.update(1)
        progress_bar.close()

        print('Extracting Frequent Itemsets . . .')
        frequent_itemsets = [[list(itemset), count] for itemset, count in support_counts.items() if count >= self.min_sup]
        print('DONE.')

        self.frequent_itemsets = frequent_itemsets
        if len(self.frequent_itemsets) > 0:
            self.itemset_size = len(self.frequent_itemsets[0][0])
        
    def get_name(self):
        return 'Brute Force'


class CountingMethod(Algorithm):
    def __init__(self) -> None:
        super().__init__()

    def find_frequent_itemsets(self, transactions, itemset_size=3, min_sup_norm=0.01, verbose=False):
        self.target_itemset_size = itemset_size
        transaction_len = len(transactions)
        self.min_sup = min_sup_norm * transaction_len
        print(f'Using min_sup={self.min_sup}\n')

        progress_bar = tqdm(total=len(transactions), desc="Candidate Generation")
        # candidate generation - basket per basket, O(N_i^2)
        support_counts = Counter()
        for transaction in transactions:

            if len(transaction) >= self.target_itemset_size:
                candidates = list(combinations(transaction, self.target_itemset_size))                
                support_counts.update(Counter(candidate for candidate in candidates))

            progress_bar.update(1)
        progress_bar.close()
                        
        print('Extracting Frequent Itemsets . . .')
        frequent_itemsets = [[list(itemset), count] for itemset, count in support_counts.items() if count >= self.min_sup]
        print('DONE.')

        self.frequent_itemsets = frequent_itemsets
        if len(self.frequent_itemsets) > 0:
            self.itemset_size = len(self.frequent_itemsets[0][0])

    def get_name(self):
        return 'Counting Method'


class Generation_Mode(Enum):
    BRUTE_FORCE = 0
    Fk_MINUS_1_F1 = 1
    Fk_MINUS_1_Fk_MINUS_1 = 2

class Pruning_Mode(Enum):
    BRUTE_FORCE = 0
    Fk_MINUS_1_F1 = 1
    Fk_MINUS_1_Fk_MINUS_1 = 2

class Support_Counting_Mode(Enum):
    BRUTE_FORCE = 0
    HASH_TREE = 1


class Apriori(Algorithm):
    def __init__(self, generation_mode:Generation_Mode=Generation_Mode.Fk_MINUS_1_Fk_MINUS_1, pruning_mode:Pruning_Mode=Pruning_Mode.Fk_MINUS_1_Fk_MINUS_1, support_counting_mode:Support_Counting_Mode=Support_Counting_Mode.HASH_TREE) -> None:        
        super().__init__()
        self.generation_mode = generation_mode
        self.pruning_mode = pruning_mode
        self.support_counting_mode = support_counting_mode

    def __brute_force_generation(self, items, k):
        return list(combinations(items, k))

    def __Fk_minus_1_F1_generation(self, Fk_minus_1, F_1, display_progress=False):
        C_k = []
        
        progress_bar = None
        if display_progress:
            progress_bar = tqdm(total=len(Fk_minus_1), desc="Candidate Generation")

        for itemset_k_minus_1 in Fk_minus_1:
            last_item = itemset_k_minus_1[-1]

            new_candidates = [itemset_k_minus_1 + single_item for single_item in F_1 if single_item[0] not in itemset_k_minus_1 and single_item[0] > last_item]
            C_k.extend(new_candidates)

            if display_progress:
                progress_bar.update(1)

        if display_progress:
            progress_bar.close()

        return C_k

    def __Fk_minus_1_Fk_minus_1_generation(self, Fk_minus_1, display_progress=False):
        C_k = []

        Fk_minus_1_len = len(Fk_minus_1)

        progress_bar = None
        if display_progress:
            progress_bar = tqdm(total=Fk_minus_1_len, desc="Candidate Generation")
        for i in range(Fk_minus_1_len):
            itemset_1 = Fk_minus_1[i]
            last_1 = itemset_1[-1]
            prefix_1 = itemset_1[:-1]

            for j in range(i+1, Fk_minus_1_len):
                itemset_2 = Fk_minus_1[j]
                
                if last_1 != itemset_2[-1] and prefix_1 == itemset_2[:-1]:
                    last_2 = Fk_minus_1[j][-1]
                    suffix = [min(last_1, last_2), max(last_1, last_2)]

                    C_k.append(prefix_1 + suffix)

            if display_progress:
                progress_bar.update(1)

        if display_progress:
            progress_bar.close()
        
        return C_k

    def __brute_force_pruning(self, C_k, Fk_minus_1, display_progress=False):
        Fk = []
        Fk_minus_1_set = {tuple(xi) for xi in Fk_minus_1}        
        progress_bar = None
        if display_progress:
            progress_bar = tqdm(total=len(C_k), desc="Pruning")

        for candidate in C_k:
            prune_candidate = False
            for subset in combinations(candidate, len(candidate) - 1):
                if subset not in Fk_minus_1_set:
                    prune_candidate = True
                    break

            if not prune_candidate:
                Fk.append(candidate)

            if display_progress:
                progress_bar.update(1)

        if display_progress:
            progress_bar.close()
            
        return Fk
    
    def __Fk_minus_1_F1_pruning(self, C_k, Fk_minus_1, display_progress=False):
        Fk = []
        Fk_minus_1 = set([tuple(xi) for xi in Fk_minus_1])
        
        progress_bar = None
        if display_progress:
            progress_bar = tqdm(total=len(C_k), desc="Pruning")
        
        for candidate in C_k:
            last_item = tuple(candidate[-1])

            prune_candidate = False
            for subset in combinations(candidate[:-1], len(candidate) - 2):
                candidate_subset = subset + last_item

                if candidate_subset not in Fk_minus_1:
                    prune_candidate = True
                    break

            if not prune_candidate:
                Fk.append(candidate)

            if display_progress:
                progress_bar.update(1)

        if display_progress:
            progress_bar.close()
            
        return Fk
    
    def __Fk_minus_1_Fk_minus_1_pruning(self, C_k, Fk_minus_1, display_progress=False):
        Fk = []
        Fk_minus_1_set = {tuple(xi) for xi in Fk_minus_1}
        progress_bar = None
        if display_progress:
            progress_bar = tqdm(total=len(C_k), desc="Pruning")

        for candidate in C_k:
            last_items = tuple(candidate[-2:])

            prune_candidate = False
            for subset in combinations(candidate[:-2], len(candidate) - 3):
                candidate_subset = subset + last_items

                if candidate_subset not in Fk_minus_1_set:
                    prune_candidate = True                    
                    break

            if not prune_candidate:
                Fk.append(candidate)

            if display_progress:
                progress_bar.update(1)

        if display_progress:
            progress_bar.close()

        return Fk

    def __brute_force_support_counting(self, C_k, transactions, display_progress=False):
        support_counts = defaultdict(int)
        candidates_sets = [frozenset(candidate) for candidate in C_k]
        
        if display_progress:
            progress_bar = tqdm(total=len(transactions), desc="Support Counting")

        for transaction in transactions:
            transaction_set = frozenset(transaction)
            for candidate_set in candidates_sets:
                if candidate_set <= transaction_set:
                    support_counts[tuple(candidate_set)] += 1
                    
            if display_progress:
                progress_bar.update(1)

        if display_progress:
            progress_bar.close()

        return support_counts
    
    def hash_tree_support_counting_mode(self, hash_tree:HashTree, transactions):
        hash_tree.count_supports(transactions, use_recursive_method=False)

    def __hash_tree_support_counting(self, C_k, transactions, max_leaf_size=50):
        # create hash tree and insert candidates      
        hash_tree = HashTree(max_leaf_size=max_leaf_size)
        hash_tree.insert_candidates(C_k)

        # count supports with mode
        self.hash_tree_support_counting_mode(hash_tree=hash_tree, transactions=transactions)

        # get support counts
        support_counts = hash_tree.get_supports()
        hash_tree.clear()

        return support_counts
    
    def __extract_F1(self, transactions):
        support_counts = Counter(item for transaction in transactions for item in transaction)
        return [[itemset] for itemset, count in support_counts.items() if count >= self.min_sup]

    def generate_with_mode(self, F_k, F_1, display_progress=False):        
        C_k = []
        match self.generation_mode:
            case Generation_Mode.BRUTE_FORCE:
                C_k = self.__brute_force_generation(F_1)

            case Generation_Mode.Fk_MINUS_1_F1:
                C_k = self.__Fk_minus_1_F1_generation(F_k, F_1, display_progress=display_progress)

            case Generation_Mode.Fk_MINUS_1_Fk_MINUS_1:
                C_k = self.__Fk_minus_1_Fk_minus_1_generation(F_k, display_progress=display_progress)
                
        return C_k

    def prune_with_mode(self, C_k, F_k, display_progress=False):
        pruned_C_k = []
        match self.pruning_mode:
            case Pruning_Mode.BRUTE_FORCE:
                pruned_C_k = self.__brute_force_pruning(C_k=C_k, Fk_minus_1=F_k, display_progress=display_progress)

            case Pruning_Mode.Fk_MINUS_1_F1:
                pruned_C_k = self.__Fk_minus_1_F1_pruning(C_k=C_k, Fk_minus_1=F_k, display_progress=display_progress)

            case Pruning_Mode.Fk_MINUS_1_Fk_MINUS_1:
                pruned_C_k = self.__Fk_minus_1_Fk_minus_1_pruning(C_k=C_k, Fk_minus_1=F_k, display_progress=display_progress)

        return pruned_C_k

    def count_supports_with_mode(self, C_k, transactions, display_progress=False):
        support_counts = []
        match self.support_counting_mode:
            case Support_Counting_Mode.BRUTE_FORCE:
                support_counts = self.__brute_force_support_counting(C_k, transactions, display_progress=display_progress)

            case Support_Counting_Mode.HASH_TREE:
                support_counts = self.__hash_tree_support_counting(C_k, transactions)

        return support_counts

    def find_frequent_itemsets(self, transactions, itemset_size=3, min_sup_norm=0.01):        
        transaction_len = len(transactions)
        self.min_sup = min_sup_norm * transaction_len
        print(f'Using min_sup={self.min_sup}\n')
        
        k = 1
        self.target_itemset_size = itemset_size
        
        print('Extracting F1... ')
        F_1 = self.__extract_F1(transactions, self.min_sup)
        print(f'Done.')

        F_k = F_1.copy()
        frequent_itemsets = F_1
        self.itemset_size = 0
        if len(F_k) > 0:
            self.itemset_size += 1

        while len(F_k) > 0 and k != self.target_itemset_size:
            transactions = Utils.remove_small_transactions(transactions, k)
            print(f'\n REMOVED {transaction_len - len(transactions)} SMALL (with size <= {k}) TRANSACTIONS\n')

            k += 1
            print(f'Mining Frequent Itmesets of size {k} . . .\n')            

            C_k = self.generate_with_mode(F_k, F_1, display_progress=True)
            print(f'  Total Generated = {len(C_k)}')

            if k > 2:
                C_k = self.prune_with_mode(C_k, F_k, display_progress=True)
                print(f'  After Pruning   = {len(C_k)}')

            if len(C_k) == 0:
                print(f'PRUNED OUT ALL TRANSACTIONS. FOUND NO FREQUENT ITEMSETS WITH SIZE {self.target_itemset_size}, USING min_sup = {self.min_sup}.')
                break
            
            support_counts = self.count_supports_with_mode(C_k=C_k, transactions=transactions)
            F_k = [[list(itemset), count] for itemset, count in support_counts.items() if count >= self.min_sup]
            
            if len(F_k) > 0:
                self.itemset_size += 1
                frequent_itemsets = F_k
                F_k = [itemset_with_count[0] for itemset_with_count in F_k]
                print(f'  Total frequent {k}-itemsets  = {len(F_k)}.')
            else:
                print(f'FOUND NO FREQUENT ITEMSETS WITH SIZE {self.target_itemset_size}, USING min_sup = {self.min_sup}.')
                break
        
        self.frequent_itemsets = frequent_itemsets
        print(f'STOPPING. LARGEST ITEMSETS FOUND HAVE SIZE {self.itemset_size}.\n')

    def get_algorithm_name(self):
        return 'Apriori'
    
    def get_name(self):
        return f'{self.get_algorithm_name()} - (Generation: {self.generation_mode.name}, Pruning: {self.pruning_mode.name}, Support Conting: {self.support_counting_mode.name})'


class PCY(Apriori):    
    def __init__(self, generation_mode:Generation_Mode=Generation_Mode.Fk_MINUS_1_Fk_MINUS_1, pruning_mode:Pruning_Mode=Pruning_Mode.Fk_MINUS_1_Fk_MINUS_1, support_counting_mode:Support_Counting_Mode=Support_Counting_Mode.HASH_TREE, tot_table_size=0, tot_megabytes=1000) -> None:
        super().__init__(generation_mode, pruning_mode, support_counting_mode)      
        if tot_table_size == 0:
            tot_table_size = Utils.get_pcy_hashtable_size(megabytes=tot_megabytes,memory_percentage=0.15)
        
        self.tot_table_size = tot_table_size
        self.table_shape = (1,self.tot_table_size)        
        self.bitmap:BitMap = BitMap(self.table_shape)        
        self.hash_func_seeds = [1]

    def __extract_F1(self, transactions, min_sup):
        support_counts = Counter(item for transaction in transactions for item in transaction)
        table = np.zeros(self.table_shape, dtype=np.int16)

        progress_bar = tqdm(total=len(transactions), desc="F1 Extraction")
        for transaction in transactions:

            if len(transaction) < 2:
                progress_bar.update(1)
                continue

            pair_hashes = np.array([hash(pair) for pair in combinations(transaction, 2)])

            hash_table_indices  = Utils.PCY_MultiHash_hash_func(itemset_hash=pair_hashes, seeds=self.hash_func_seeds, m=self.tot_table_size)       
            for i in range(len(self.hash_func_seeds)):
                unique_indices, counts = np.unique(hash_table_indices[i], return_counts=True)
                table[i][unique_indices] += counts                

            progress_bar.update(1)
        progress_bar.close()

        self.bitmap.set_from_bool_array(table >= min_sup)

        return [[itemset] for itemset, count in support_counts.items() if count >= self.min_sup]

    def hash_tree_support_counting_mode(self, hash_tree:HashTree, transactions):
        self.bitmap.set_from_bool_array(hash_tree.count_supports_PCY_MultiHash(transactions, hash_func_seeds=self.hash_func_seeds, table_size=self.tot_table_size, min_sup=self.min_sup, target_itemset_size=self.target_itemset_size))

    def __generate_candidates_pcy(self, Fk_minus_1, display_progress=False):
        C_k = []

        Fk_minus_1_len = len(Fk_minus_1)

        progress_bar = None
        if display_progress:
            progress_bar = tqdm(total=Fk_minus_1_len, desc="Candidate Generation")
        
        for i in range(Fk_minus_1_len):
            itemset_1 = Fk_minus_1[i]
            last_1 = itemset_1[-1]
            prefix_1 = itemset_1[:-1]

            for j in range(i+1, Fk_minus_1_len):
                itemset_2 = Fk_minus_1[j]
                
                if last_1 != itemset_2[-1] and prefix_1 == itemset_2[:-1]:
                    last_2 = Fk_minus_1[j][-1]
                    suffix = [min(last_1, last_2), max(last_1, last_2)]

                    candidate = prefix_1 + suffix
                    candidate_hash = hash(tuple(candidate))
                    discard = False
                    for i, seed in enumerate(self.hash_func_seeds):
                        if not self.bitmap.at(i, Utils.PCY_MultiHash_hash_func(itemset_hash=np.array([candidate_hash]), seeds=np.array([seed]), m=self.tot_table_size)[0][0]):
                            discard = True
                            break
                    if not discard:
                        C_k.append(candidate)

            if display_progress:
                progress_bar.update(1)

        if display_progress:
            progress_bar.close()

        self.bitmap.clear_bitmap()
        
        return C_k

    def find_frequent_itemsets(self, transactions, itemset_size=3, min_sup_norm=0.01, verbose=False):      
        transaction_len = len(transactions)
        self.min_sup = min_sup_norm * transaction_len
        print(f'Using min_sup={self.min_sup}\n')              

        k = 1
        self.target_itemset_size = itemset_size

        F_1 = self.__extract_F1(transactions, self.min_sup)        
        F_k = F_1.copy()
        frequent_itemsets = F_1
        self.itemset_size = 0
        if len(F_k) > 0:
            self.itemset_size += 1       
        
        while len(F_k) > 0 and k != self.target_itemset_size: 
            transactions = Utils.remove_small_transactions(transactions, k)
            print(f'\n REMOVED {transaction_len - len(transactions)} SMALL (with size <= {k}) TRANSACTIONS\n')

            k += 1
            print(f'Mining Frequent Itmesets of size {k} . . .\n')            

            C_k = self.__generate_candidates_pcy(F_k, F_1)
            C_k_len = len(C_k)
            print(f'  Total Generated = {C_k_len}')

            if k > 2:
                C_k = self.prune_with_mode(C_k, F_k)
                print(f'  After Pruning   = {len(C_k)}')

            if len(C_k) == 0:
                print(f'PRUNED OUT ALL TRANSACTIONS. FOUND NO FREQUENT ITEMSETS WITH SIZE {self.target_itemset_size}, USING min_sup = {self.min_sup}.')
                break
            
            support_counts = self.count_supports_with_mode(C_k=C_k, transactions=transactions)
            F_k = [[list(itemset), count] for itemset, count in support_counts.items() if count >= self.min_sup]                

            if len(F_k) > 0:
                self.itemset_size += 1
                frequent_itemsets = F_k
                F_k = [itemset_with_count[0] for itemset_with_count in F_k]            
                print(f'  Total frequent {k}-itemsets  = {len(F_k)}.')
            else:
                print(f'FOUND NO FREQUENT ITEMSETS WITH SIZE {self.target_itemset_size}, USING min_sup = {self.min_sup}.')
                break

        self.frequent_itemsets = frequent_itemsets
        print(f'STOPPING. LARGEST ITEMSETS FOUND HAVE SIZE {self.itemset_size}.\n')

    def get_algorithm_name(self):
        return 'PCY'    


class MultiHash(PCY):
    def __init__(self, generation_mode:Generation_Mode=Generation_Mode.Fk_MINUS_1_Fk_MINUS_1, pruning_mode:Pruning_Mode=Pruning_Mode.Fk_MINUS_1_Fk_MINUS_1, support_counting_mode:Support_Counting_Mode=Support_Counting_Mode.HASH_TREE, tot_tables_size = 0, tot_hash_tables = 2, tot_megabytes=1000) -> None:
        super().__init__(generation_mode, pruning_mode, support_counting_mode, tot_tables_size)   
        if tot_tables_size == 0:
            tot_tables_size = Utils.get_pcy_hashtable_size(megabytes=tot_megabytes, memory_percentage=0.15)        
        self.tot_table_size = round(tot_tables_size / tot_hash_tables)        
        
        self.table_shape = (tot_hash_tables, self.tot_table_size)
        self.bitmap:BitMap = BitMap(self.table_shape)
        self.hash_func_seeds = []
        self.set_hash_func_seeds(tot_hash_tables, self.tot_table_size)
        
    def set_hash_func_seeds(self, tot_seeds, table_size, base_rand_seed=1492):
        random.seed(base_rand_seed)
        while len(self.hash_func_seeds) < tot_seeds:
            seed = sympy.nextprime(random.randint(1,table_size))
            if seed not in self.hash_func_seeds:
                self.hash_func_seeds.append(seed)

    def get_algorithm_name(self):
        return 'MultiHash'