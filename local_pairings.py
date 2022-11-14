import os
import pickle
import hash_table as ht
import numpy as np
import importlib
importlib.reload(ht)
from hash_table import get_hash_table



def get_pairings(N, hash_table):

    pairings = {}

    for i in range(0, N, 1):
        for j in range(0, i+1, 1):
            if (j, i) not in pairings:
                pairings[(j, i)] = {}
                pairings[(j, i)][(0, 0)] = []
                pairings[(j, i)][(0, 1)] = []
                pairings[(j, i)][(1, 0)] = []
                pairings[(j, i)][(1, 1)] = []

    for state in hash_table:
        for i in range(0, N, 1):
            for j in range(0, i + 1, 1):
                local_state = (state[j],state[i])
                if local_state == (0,1):
                    continue
                pairings[(j, i)][local_state].append(hash_table[state])
                if local_state == (1, 0):
                    conj_state = list(state)
                    conj_state[j] = 0
                    conj_state[i] = 1
                    pairings[(j, i)][(0, 1)].append(hash_table[tuple(conj_state)])

    return pairings


def get_even_local_pairings(N,hash_table):
    pairings = {}

    for i in range(0, N - 1, 2):
        if (i,i+1) not in pairings:
            pairings[(i,i+1)] = {}
        pairings[(i, i + 1)][(0, 0)] = []
        pairings[(i, i + 1)][(0, 1)] = []
        pairings[(i, i + 1)][(1, 0)] = []
        pairings[(i, i + 1)][(1, 1)] = []

    for state in hash_table:
        for i in range(0,N-1,2):
            local_state = state[i:i+2]
            pairings[(i,i+1)][local_state].append(hash_table[state])

    return pairings


def get_odd_local_pairings(N,hash_table):
    pairings = {}

    for i in range(1, N, 2):
        next_site = (i + 1) % N
        if (i, next_site) not in pairings:
            pairings[(i, next_site)] = {}
        pairings[(i, next_site)][(0, 0)] = []
        pairings[(i, next_site)][(0, 1)] = []
        pairings[(i, next_site)][(1, 0)] = []
        pairings[(i, next_site)][(1, 1)] = []

    for state in hash_table:
        for i in range(0, N - 1, 2):
            next_site = (i + 1) % N
            local_state = state[i:i + 2]
            pairings[(i, next_site)][local_state].append(hash_table[state])

    return pairings


def local_pairings(N, Q, hash_table):
    filename = 'sep_data/pairing_data/pairings_N='+str(N)
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            pairs = pickle.load(f)
        if Q not in pairs:
            pairs_Q = get_pairings(N, hash_table)
            pairs[Q] = pairs_Q
            with open(filename, 'wb') as f:
                pickle.dump(pairs, f)
        return pairs[Q]

    else:
        pairs = {}
        pairs_Q = get_pairings(N, hash_table)
        pairs[Q] = pairs_Q
        with open(filename,'wb') as f:
            pickle.dump(pairs,f)
        return pairs[Q]