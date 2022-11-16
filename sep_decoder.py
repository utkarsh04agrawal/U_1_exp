import os
import pickle
import time
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as pl
import importlib
import hash_table as ht
import local_pairings as lp
importlib.reload(ht)
importlib.reload(lp)

def TransferMatrix(i,j,pairings,dim,L,Q,debug=False):
    """
    i,j are the sites to apply transfer matrix to
    pairings are indices in the hash table with some particular value at i,j
    dim is size of hash table
    outcomes is list of measurement outcomes at i,j. 0 is no measurement, +1,-1 are respectively 1,0 measurements outcomes
    """

    if i>j:
        temp = i
        i = j
        j = temp
    
    # state_ab is the list of indices of the configuration space where charge at sites i,j are equal to a,b respectively
    states_00 = pairings[(i, j)][(0, 0)]
    states_10 = pairings[(i, j)][(1, 0)]
    states_01 = pairings[(i, j)][(0, 1)]
    states_11 = pairings[(i, j)][(1, 1)]

    fac00, fac01, fac10, fac11 = 1,1,1,1
    T = sparse.dok_matrix((dim,dim),dtype=float)
    T[states_00, states_00] = 1*fac00
    T[states_01, states_01] = 0.5*fac01
    T[states_10, states_10] = 0.5*fac10
    T[states_01, states_10] = 0.5*fac01
    T[states_10, states_01] = 0.5*fac10
    T[states_11, states_11] = 1*fac11

    return T.tocsr()


def load_transfer_matrices(pairings,dim,L,Q):
    filename = 'sep_data/transfer_matrix/L='+str(L)+'_Q='+str(Q)
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            transfer_matrices = pickle.load(f)
    else:
        transfer_matrices = {}

    added_something = False
    for x in range(0,L-1,2):
        if (x,x+1) not in transfer_matrices:
            T = TransferMatrix(x,x+1,pairings,dim,L,Q)
            transfer_matrices[(x,x+1)] = T
            added_something = True
    for x in range(1,L-1,2): #BC are open
        if (x,x+1) not in transfer_matrices:
            T = TransferMatrix(x,x+1,pairings,dim,L,Q)
            transfer_matrices[(x,x+1)] = T
            added_something = True
    
    if added_something:
        with open(filename,'wb') as f:
            pickle.dump(transfer_matrices,f)
    
    return transfer_matrices



# Function to get the initial state used in the Qiskit simulation
def initial_state(L,Q,hash_table,neel_state=True):
    #
    if not neel_state:
        filename = 'data/scrambled_states/L='+str(L)+'_T='+str(2*L)+'_Q='+str(Q)
        with open(filename,'rb') as f:
            state = pickle.load(f)
        state = np.asarray(state).reshape((2,)*L)
        state = np.transpose(state,range(L)[::-1]) # reversing the order of qubits as qiskit has qubit #0 at the right end
    else:
        state = np.zeros((2,)*L)
        confi = [0]*L
        for x in range(0,Q,1):
            confi[2*x] = 1
        state[tuple(confi)] = 1
    
    #
    #
    dim = len(hash_table)
    classical_state = np.zeros(dim)
    for confi,index in hash_table.items():
        classical_state[index] = np.abs(state[confi])**2
    return classical_state



# Function to load measurement data obtained from Qiskit simulation
def load_measurement_data(filename):
    """
    data is a list of different trajectories. For example, data[i] = (measurement_array,m_locs,param_list). measurement_array is a tuple:(array,int) and stores a trajectory stored as a tuple of an array Arr of shape (depth,L) and number of times it occured, int; the entries of the array are 0 if no measurement has been performed at the space time point, or -1 or +1 where the sign corresponds to measurement outcome 0 or 1.
    """
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return data
    

def transfer(x,dim,pairings,transfer_matrices,outcomes,state_Q,log_Z,state_zero,L,Q,debug=False):
    if not state_zero:
        T = transfer_matrices[(x,x+1)]
        state_Q = T.dot(state_Q)
        
        if outcomes[0] !=0:
            proj1 = 1 - (outcomes[0]+1)//2
            state_Q[pairings[(x,x)][(proj1,proj1)]] = 0
        if outcomes[1] !=0:
            proj2 = 1 - (outcomes[1]+1)//2
            state_Q[pairings[(x+1,x+1)][(proj2,proj2)]] = 0
        
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


def trial(L):
    Q = L//2
    Q2 = L//2 - 1
    hash_table,_ = ht.get_hash_table(L,Q)
    hash_table2,_= ht.get_hash_table(L,Q2)
    pairings = lp.local_pairings(L,Q,hash_table)
    pairings2 = lp.local_pairings(L,Q2,hash_table2)
    dim = len(hash_table)
    dim2 = len(hash_table2)

    seed_state = [1,0]*(L//2)
    initial_state_Q = np.zeros(dim)
    initial_state_Q[hash_table[tuple(seed_state)]] = 1
    initial_state_Q2 = np.zeros(dim2)
    seed_state[0] = 0
    initial_state_Q2[hash_table2[tuple(seed_state)]] = 1

    transfer_matrices = load_transfer_matrices(pairings,dim,L,Q)
    transfer_matrices2 = load_transfer_matrices(pairings2,dim2,L,Q2)

    p_success = []
    depth = L
    traj = np.zeros((L,L))
    
    state_Q_is_zero = False
    state_Q2_is_zero = False
    state_Q = initial_state_Q.copy()
    state_Q2 = initial_state_Q2.copy()
    log_Z = []
    log_Z2 = []

    for t in range(depth)[:]:
        
        if t%2 == 0: #even layer
            
            for x in range(0,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
                
                state_Q, state_Q_is_zero = transfer(x,dim,pairings,transfer_matrices, outcomes,state_Q,log_Z,state_Q_is_zero,L,Q,debug=False)
                
                state_Q2, state_Q2_is_zero = transfer(x,dim2,pairings2,transfer_matrices2 ,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,Q2)

                if state_Q_is_zero and state_Q2_is_zero:
                    print('hurray2',np.sum(state_Q2),np.sum(state_Q))
            
        else:
            for x in range(1,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
                debug=False
                # print(outcomes,t,x)
                state_Q, state_Q_is_zero = transfer(x,dim,pairings,transfer_matrices, outcomes,state_Q,log_Z,state_Q_is_zero,L,Q,debug=False)

                state_Q2, state_Q2_is_zero = transfer(x,dim2,pairings2,transfer_matrices2 ,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,Q2)

                if state_Q_is_zero and state_Q2_is_zero:
                    print('hurray2',np.sum(state_Q2),np.sum(state_Q))
        
    
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
    p_success.extend([p_Q])
    # print(np.exp(log_Z),np.exp(log_Z2))
      
    return p_success


def sep_dynamics(L,depth,Q,p,seed=1,neel_initial_state=True,what_type='nothing_fixed'):
    Q2 = Q-1
    if Q<L//2:
        Q2 = Q+1
    hash_table,_ = ht.get_hash_table(L,Q)
    hash_table2,_= ht.get_hash_table(L,Q2)
    pairings = lp.local_pairings(L,Q,hash_table)
    pairings2 = lp.local_pairings(L,Q2,hash_table2)
    dim = len(hash_table)
    dim2 = len(hash_table2)
    
    if not neel_initial_state:
        scrambling_label='_scrambled'
    else:
        scrambling_label = ""
    if what_type=="nothing_fixed":
        filename = 'data/measurement_data'+scrambling_label+'/L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p='+str(p)+'_seed='+str(seed)
    elif what_type == "unitaries_fixed":
        filename = 'data/measurement_data_unitaries_fixed'+scrambling_label+'/param_seed=100/L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p='+str(p)+'_seed='+str(seed)
    data = load_measurement_data(filename)

    total = 0
    p_success = []

    transfer_matrices = load_transfer_matrices(pairings,dim,L,Q)
    transfer_matrices2 = load_transfer_matrices(pairings2,dim2,L,Q2)
    
    initial_state_Q = initial_state(L,Q,hash_table,neel_state=neel_initial_state)
    initial_state_Q2 = initial_state(L,Q2,hash_table2,neel_state=neel_initial_state)

    for measurements,m_loc,param in data[:]:
        traj,N = measurements[0]
        state_Q_is_zero = False
        state_Q2_is_zero = False
        state_Q = initial_state_Q.copy()
        state_Q2 = initial_state_Q2.copy()
        log_Z = []
        log_Z2 = []
        total += N
        for t in range(depth)[:-1]:
            
            if t%2 == 0: #even layer
                for x in range(0,L-1,2):
                    outcomes = (traj[t,x],traj[t,x+1])
                
                    state_Q, state_Q_is_zero = transfer(x,dim,pairings,transfer_matrices, outcomes,state_Q,log_Z,state_Q_is_zero,L,Q,debug=False)

                    state_Q2, state_Q2_is_zero = transfer(x,dim2,pairings2,transfer_matrices2 ,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,Q2)
                    
                    if state_Q_is_zero and state_Q2_is_zero:
                        print('hurray2',np.sum(state_Q2),np.sum(state_Q))
            else:
                for x in range(1,L-1,2):
                    outcomes = (traj[t,x],traj[t,x+1])
                    debug=False

                    state_Q, state_Q_is_zero = transfer(x,dim,pairings,transfer_matrices, outcomes,state_Q,log_Z,state_Q_is_zero,L,Q,debug=False)

                    state_Q2, state_Q2_is_zero = transfer(x,dim2,pairings2,transfer_matrices2 ,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,Q2)

                    if state_Q_is_zero and state_Q2_is_zero:
                        print('hurray2',np.sum(state_Q2),np.sum(state_Q))
                   

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
        p_success.extend([p_Q]*N)
    return p_success


def run(filename,neel_initial_state=True,what_type='nothing_fixed'):
    aa = {}
    p_list = [0.05,0.1,0.13,0.16,0.2,0.25,0.3]
    # p_list = [0.13]
    L_list = [8,10,12]
    for L in L_list[:]:
        aa[L] = {}
        for p in p_list[:]:
            print(p)
            aa[L][p] = {}
            start = time.time()
            aa[L][p][L//2] = sep_dynamics(L,L,L//2,p,seed=1,neel_initial_state=neel_initial_state,what_type=what_type)
            aa[L][p][L//2-1] = sep_dynamics(L,L,L//2-1,p,seed=1,neel_initial_state=neel_initial_state,what_type=what_type)
            print(p,' completed, time', time.time()-start)

        with open(filename,'wb') as f:
            pickle.dump(aa,f)
    return aa


filename = 'sep_data/nothing_fixed'
aa = run(filename,neel_initial_state=False,what_type='nothing_fixed')
with open(filename,'wb') as f:
    pickle.dump(aa,f)