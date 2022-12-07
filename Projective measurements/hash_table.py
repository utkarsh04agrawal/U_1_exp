import numpy as np
import matplotlib.pyplot as pl
from itertools import combinations
import os
import pickle


def get_states(N,Q):
    states = []
    for positions in combinations(range(N), Q):
        p = [0] * N

        for i in positions:
            p[i] = 1

        states.append(tuple(p))
    return states

def get_hash_table(N,Q):
    hash_table = {}
    reverse_hash_table = {}
    filename = 'sep_data/hash_tables/hash_table_N='+str(N)+'_Q='+str(Q)
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            hash_table, reverse_hash_table = pickle.load(f)
        return hash_table, reverse_hash_table
    else:
        states = get_states(N,Q)
        print(len(states))
        for i in enumerate(states):
            hash_table[i[1]] = i[0]
            reverse_hash_table[i[0]] = i[1]
        with open(filename,'wb') as f:
            pickle.dump([hash_table, reverse_hash_table], f)
        return hash_table, reverse_hash_table


