"""
This code runs the SEP decoder without sparse matrices but useing tensor structure
"""

import os
import pickle
import time
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as pl
from itertools import combinations


# Function to get the initial state used in the Qiskit simulation
def initial_state(L,Q,neel_state=True):
    

    state = np.zeros((2,)*L)
    for positions in combinations(range(L), Q):
        p = [0] * L

        for i in positions:
            p[i] = 1

        state[tuple(p)] = 1 
    state = state/np.sum(state)
    return state

    if not neel_state:
        filename = 'Weak measurements/data/scrambled_states/L='+str(L)+'_T='+str(2*L)+'_Q='+str(Q)
        with open(filename,'rb') as f:
            state = pickle.load(f)
        state = np.abs(np.asarray(state).reshape((2,)*(L+1))[0,:])**2
        state = np.transpose(state,range(L)[::-1]) # reversing the order of qubits as qiskit has qubit #0 at the right end
    else:
        state = np.zeros((2,)*L)
        confi = [0]*L
        for x in range(0,Q,1):
            confi[2*x] = 1
        state[tuple(confi)] = 1
    
    #
    #
    return state


def transfer(x,T,outcomes,state_Q,log_Z,state_zero,L,theta,debug=False):
    
    if not state_zero:
        
        if x%2 == 1: # transfer step in odd layer
            state_Q = np.moveaxis(state_Q,0,-1)
        state_Q = state_Q.reshape((4,)*(L//2))
        state_Q = np.tensordot(T,state_Q,axes=(-1,x//2))
        
        if outcomes[0] == 1:
            state_Q[(0,1),:] = 0
            state_Q[(2,3),:] = np.sin(theta)**2 * state_Q[(2,3),:]
        elif outcomes[0] == -1:
            state_Q[(2,3),:] = np.cos(theta)**2 *state_Q[(2,3),:]

        if outcomes[1] == 1:
            state_Q[(0,2),:] = 0
            state_Q[(1,3),:] = np.sin(theta)**2 * state_Q[(1,3),:]
        elif outcomes[1] == -1:
            state_Q[(1,3),:] = np.cos(theta)**2 *state_Q[(1,3),:]

        state_Q = np.moveaxis(state_Q,0,x//2)
        state_Q = state_Q.reshape((2,)*L)
        if x%2 == 1:
            state_Q = np.moveaxis(state_Q,-1,0)


        sQ = np.sum(state_Q)
        if sQ == 0:
            state_zero = True
            log_Z.append(-np.inf)
        else:
            state_zero = False
            log_Z.append(np.log(sQ))
            state_Q = state_Q/sQ
    else:
        log_Z.append(-np.inf)
    
    return state_Q, state_zero


# def sep_dynamics_2(data,Q,theta,half_depth,neel_initial_state=False,decoding_protocol=0):

#     """
#     input:
#         - decoding_protocol
#             0: No active decoding. Post-select such that both P_Q and P_Q1 are 0
#             1: Postselect on trajectories where P_suc != 0, i.e P_Q != 0
#             2: Postselect on trajectories where last layer has total charge = Q
#             3: Union of 2 and 1
#     """

#     (depth,L) = data.shape
    
#     Q2 = Q-1
#     if Q<L//2:
#         Q2 = Q+1

#     if decoding_protocol == 2 or decoding_protocol == 3:
#         # print(data[:,:],Q)
#         if np.sum((data[-1,:]+1)) != 2*Q:
#             return False

#     total = 0
#     p_success = []
    
#     initial_state_Q = initial_state(L,Q,neel_state=neel_initial_state)
#     initial_state_Q2 = initial_state(L,Q2,neel_state=neel_initial_state)

#     N=1
#     traj = data
#     state_Q_is_zero = False
#     state_Q2_is_zero = False
#     state_Q = initial_state_Q.copy()
#     state_Q2 = initial_state_Q2.copy()

#     T = np.eye(4)
#     T[1,1] = 1/2
#     T[2,2] = 1/2
#     T[1,2] = 1/2
#     T[2,1] = 1/2

#     log_Z = []
#     log_Z2 = []
#     total += N
#     decoder_depth = L
#     if half_depth == True:
#         decoder_depth = L//2
#     for t in range(decoder_depth)[:-1]:
#         if t%2 == 0: #even layer
#             for x in range(0,L-1,2):
#                 outcomes = (traj[t,x],traj[t,x+1])
            
#                 state_Q, state_Q_is_zero = transfer(x,T,outcomes,state_Q,log_Z,state_Q_is_zero,L,theta,debug=False)

#                 state_Q2, state_Q2_is_zero = transfer(x,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,theta)
                
#                 if state_Q_is_zero:
#                     if decoding_protocol == 1 or decoding_protocol == 3:
#                         return False
#                     if state_Q2_is_zero:
#                         return False
#         else:
#             for x in range(1,L-1,2):
#                 outcomes = (traj[t,x],traj[t,x+1])
#                 debug=False

#                 state_Q, state_Q_is_zero = transfer(x,T,outcomes,state_Q,log_Z,state_Q_is_zero,L,theta,debug=False)

#                 state_Q2, state_Q2_is_zero = transfer(x,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,theta)

#                 if state_Q_is_zero:
#                     if decoding_protocol == 1 or decoding_protocol == 3:
#                         return False
#                     if state_Q2_is_zero:
#                         return False

#         # print(t,time.time()-start,total)
#         if state_Q_is_zero:    
#             return False          
#             break
    
#         if state_Q2_is_zero:
#             break
        
#     if state_Q_is_zero:
#         p_Q = 0
#     elif state_Q2_is_zero:
#         p_Q = 1
#     else:
#         ratio_p = np.exp(np.sum(log_Z) - np.sum(log_Z2))
#         p_Q = 1/(1+1/ratio_p)
#         p_Q2 = 1/(1+ratio_p)
    
#     return p_Q

def sep_dynamics_2(data,Q,theta,neel_initial_state=True, decoding_protocol=0):
    

    """
    input:
        - data (_type_): 2d array of shape (depth,L) holding values of the outcomes
        - Q (_type_): initial charge of the quantum state which was used to generate the outcomes
        - neel_initial_state (bool, optional): Defaults to True. This argument is redundant now!
        - decoding_protocol
            0: No active decoding. Post-select such that both P_Q and P_Q1 are 0
            1: Postselect on trajectories where P_suc != 0, i.e P_Q != 0
            2: Postselect on trajectories where last layer has total charge = Q
            3: Union of 2 and 1

    Returns:
        p_Q: Probility of the initial charge being Q (the true charge) in the SEP dynamics
    """

    (depth,L) = data.shape
    
    Q2 = Q-1
    if Q<L//2:
        Q2 = Q+1

    if decoding_protocol == 2 or decoding_protocol == 3:
        print(data[:,:],Q)
        if np.sum((data[-1,:]+1)) != 2*Q:
            return False

    total = 0
    p_success = []
    
    initial_state_Q = initial_state(L,Q,neel_state=neel_initial_state)
    initial_state_Q2 = initial_state(L,Q2,neel_state=neel_initial_state)

    N=1
    traj = data
    state_Q_is_zero = False
    state_Q2_is_zero = False
    state_Q = initial_state_Q.copy()
    state_Q2 = initial_state_Q2.copy()

    T = np.eye(4)
    T[1,1] = 1/2
    T[2,2] = 1/2
    T[1,2] = 1/2
    T[2,1] = 1/2

    log_Z = []
    log_Z2 = []
    total += N
    for t in range(depth)[:-1]:
        if t%2 == 0: #even layer
            for x in range(0,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
            
                state_Q, state_Q_is_zero = transfer(x,T,outcomes,state_Q,log_Z,state_Q_is_zero,L,theta,debug=False)

                state_Q2, state_Q2_is_zero = transfer(x,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,theta)
                
                if state_Q_is_zero:
                    if decoding_protocol == 1 or decoding_protocol == 3:
                        return False
                    if state_Q2_is_zero:
                        return False
        else:
            for x in range(1,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
                debug=False

                state_Q, state_Q_is_zero = transfer(x,T,outcomes,state_Q,log_Z,state_Q_is_zero,L,theta,debug=False)

                state_Q2, state_Q2_is_zero = transfer(x,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,theta)

                if state_Q_is_zero:
                    if decoding_protocol == 1 or decoding_protocol == 3:
                        return False
                    if state_Q2_is_zero:
                        return False

        # print(t,time.time()-start,total)
        if state_Q_is_zero:              
            break
    
        if state_Q2_is_zero:
            break
        
    if state_Q_is_zero:
        p_Q = 0
    elif state_Q2_is_zero:
        p_Q = 1
    else:
        ratio_p = np.exp(np.sum(log_Z) - np.sum(log_Z2))
        p_Q = 1/(1+1/ratio_p)
        p_Q2 = 1/(1+ratio_p)
    
    return p_Q



def get_sep_data(Q,L,p,depth_ratio,is_noisy,filedir,half_depth,decoding_protocol):
    depth = int(L*depth_ratio)
    p_suc = []
    seed=1
    if depth_ratio != 1:
        depth_label= "_depth_ratio="+str(depth_ratio)
    else:
        depth_label = ""

    if is_noisy:
        noisy_label = '_noisy'
    else:
        noisy_label = ''

    
    filename = filedir +'/L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p=' + str(p)+ '_seed='+str(seed)
    with open(filename,'rb') as f:
        data_raw = pickle.load(f)

    faulty_traj = 0
    total_traj = 0
    for data in data_raw:
        result = sep_dynamics_2(data[0],Q,p,half_depth,decoding_protocol=decoding_protocol)
        if result is False:
            faulty_traj += data[1]
        else:
            p_suc.extend([result]*data[1])
        total_traj += data[1]

    return p_suc,faulty_traj/total_traj


p_list = np.round(np.linspace(0,np.pi/2,10),3)[:]
p_list = np.round(np.linspace(0.2,0.6*np.pi/2,15),3)[:]

L_list = [6,8,10,12,14][:4]

def collect_sep_data(L_list,p_list,decoding_protocol,depth_ratio=1,half_depth=False,is_noisy=True,final_direc="data/emulator_sep_data/",outcomes_filedir = 'data/emulator_data/measurement_data_all_qubits'):
    p_suc_dic = {}

    if not os.path.isdir(final_direc):
        os.makedirs(final_direc)
    final_file = final_direc + 'seed=1_all_qubits'

    final_file = final_file + '_decoding_protocol='+str(decoding_protocol)

    if half_depth:
        final_file = final_file + '_half_linear'

    if is_noisy:
        final_file = final_file + '_noisy'


    for L in L_list:
        p_suc_dic[L] = {}
        for p in p_list:
            p_suc_dic[L][p] = {}
            for Q in [L//2,L//2-1][:]:

                start = time.time()
                p_suc_dic[L][p][Q],temp = get_sep_data(Q,L,p,depth_ratio=depth_ratio,half_depth=half_depth, is_noisy=is_noisy,filedir=outcomes_filedir,decoding_protocol=decoding_protocol)
                print(L,p,Q," frac of faulty traj:",temp," time=",time.time()-start)
        with open(final_file,'wb') as f:
            pickle.dump(p_suc_dic,f)

