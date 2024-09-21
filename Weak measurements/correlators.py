"""
This code calculates classical-quantum correlators.
"""

import os
import sys
sys.path.append('/Users/utkarshagrawal/Documents/Postdoc/U_1_exp/Weak measurements')
import pickle
import time
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as pl
from itertools import combinations

from circuit_numpy_utils import get_circuit, get_diffusion_constants

import sep_decoder_2, biased_decoder, quantum_decoder

CLASSICAL_ALGO = 'QUANTUM'
assert CLASSICAL_ALGO in ['SEP','BIASED_SEP','QUANTUM']

if CLASSICAL_ALGO == 'SEP':
    classical_dynamics = sep_decoder_2.sep_dynamics_2
elif CLASSICAL_ALGO == 'BIASED_SEP':
    classical_dynamics = biased_decoder.biased_dynamics
elif CLASSICAL_ALGO == 'QUANTUM':
    classical_dynamics = quantum_decoder.quantum_dynamics_2


hardware_data = True

root_direc = "data/qiskit_data/measurement_data_all_qubits_special_depth_ratio=0.5"
if hardware_data: root_direc = '../Quantinum/Weak measurements/data/hardware_data/measurement_data_all_qubits_noisy_depth_ratio=0.5'

def get_params(file:str):
    index = file.index('L=')
    index2 = file.index('_depth')
    index3 = file.index('_Q=')
    index4 = file.index('_p=')
    index5 = file.index('_seed=')
    L = int(file[index+2:index2])
    depth = int(file[index2+7:index3])
    Q = int(file[index3+3:index4])
    p = float(file[index4+3:index5])
    return L,depth,Q,p


def get_correlation_matrix(data):
    final_time_data = data[-1,:]
    L = len(final_time_data)
    corr_mat = np.eye(L)
    for i in range(L):
        for j in range(i+1,L):
            corr_mat[i,j] = final_time_data[i]*final_time_data[j]
            corr_mat[j,i] = corr_mat[i,j]
    return corr_mat


def get_classical_correlation_matrix(state,L):
    assert state.shape == (2,)*L

    if np.sum(state) == 0:
        return False
    
    corr_mat = np.eye(L)
    if CLASSICAL_ALGO != 'QUANTUM':
        state = state/np.sum(state)
        for i in range(L):
            for j in range(i+1,L):
                state_copy = state.copy()
                state_copy = np.swapaxes(state_copy,i,0)
                state_copy = np.swapaxes(state_copy,j,1)
                state_copy[0,1] = -state_copy[0,1]
                state_copy[1,0] = -state_copy[1,0]
                corr_mat[i,j] = np.sum(state_copy)
                corr_mat[j,i] = corr_mat[i,j]
    else:
        state = state/np.sum(np.abs(state)**2)**0.5
        for i in range(L):
            for j in range(i+1,L):
                state_copy = state.copy()
                state_copy = np.swapaxes(state_copy,i,0)
                state_copy = np.swapaxes(state_copy,j,1)
                state_copy[0,1] = -state_copy[0,1]
                state_copy[1,0] = -state_copy[1,0]
                corr_mat[i,j] = np.sum(state_copy*np.conj(state))
                corr_mat[j,i] = corr_mat[i,j]

    return corr_mat

def get_quantum_avg_z(data):
    final_time_data = data[-1,:]
    return final_time_data

def get_classical_avg_z(state,L):
    assert state.shape == (2,)*L

    if np.sum(state) == 0:
        return False

    final_data = np.zeros(L)
    for x in range(L):
        state_copy = state.copy()
        state_copy = np.swapaxes(state_copy,x,0)
        final_data[x] = np.sum(state_copy[1]) - np.sum(state_copy[0])

    return final_data


L_list = [6,8,10,12,14]

zz_correlation_data = {}
z_correlation_data = {}
fluc_correlation_data = {}
for L in L_list:
    if L not in zz_correlation_data:
        zz_correlation_data[L] = {}
        z_correlation_data[L] = {}
        fluc_correlation_data[L] = {}
    if CLASSICAL_ALGO == 'SEP':
        extra_para = 5 #t_scr
    elif CLASSICAL_ALGO == 'BIASED_SEP':
        circuit = get_circuit(L=L,depth=int(0.5*L),scrambling_type='Special',seed=1,t_scram=5)
        extra_para = get_diffusion_constants(circuit_layer=circuit)
    else:
        extra_para = get_circuit(L,int(0.5*L),scrambling_type='Special',seed=1,t_scram=5)


    for file in os.listdir(root_direc):
        L_file,depth,Q,p = get_params(file)
        if L_file != L:
            continue
        
        if (Q,p) not in zz_correlation_data[L]: zz_correlation_data[L][(Q,p)] = []
        if (Q,p) not in z_correlation_data[L]: z_correlation_data[L][(Q,p)] = []
        if (Q,p) not in fluc_correlation_data[L]: fluc_correlation_data[L][(Q,p)] = []
        if not hardware_data:
            with open(root_direc+'/'+file,'rb') as f:
                data_raw,_,_ = pickle.load(f)
        else:
            with open(root_direc+'/'+file,'rb') as f:
                data_raw = pickle.load(f)

        start = time.time()
        for data in data_raw:
            # data[0] is the trajectory data
            # data[1] is the # of times data[0] had occured

            quantum_cor = get_correlation_matrix(data[0])
            quantum_avg = get_quantum_avg_z(data[0])

            classical_data = classical_dynamics(data[0],Q,p,extra_para,neel_state=True,decoding_protocol=3)
            if classical_data is False:
                continue
            else:
                p0,classical_state = classical_data
            classical_cor = get_classical_correlation_matrix(classical_state,L)
            classical_avg = get_classical_avg_z(classical_state,L)


            # print(quantum_avg,classical_avg)
            z_cor = np.kron(quantum_avg,classical_avg)
            
            zz_correlation_data[L][(Q,p)].extend([quantum_cor*classical_cor]*data[1])

            z_correlation_data[L][(Q,p)].extend([z_cor.reshape((L,L))]*data[1])

            fluc = quantum_avg - classical_avg
            fluc_correlation = np.kron(fluc,fluc).reshape((L,L))
            fluc_correlation_data[L][(Q,p)].extend([fluc_correlation]*data[1])

        print("L={},p={},Q={},time={}".format(L,p,Q,time.time()-start))
    hardware_label = ''
    if hardware_data: hardware_label = 'hardware_'
    with open('correlator_data/'+hardware_label+'ZZ_data_'+CLASSICAL_ALGO,'wb') as f:
        pickle.dump(zz_correlation_data,f)
    with open('correlator_data/'+hardware_label+'Z_data_'+CLASSICAL_ALGO,'wb') as f:
        pickle.dump(z_correlation_data,f)
    with open('correlator_data/'+hardware_label+'fluc_data_'+CLASSICAL_ALGO,'wb') as f:
        pickle.dump(fluc_correlation_data,f)




