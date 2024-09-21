"""
This code runs the SEP decoder without sparse matrices and instead uses tensor structure
"""

import os
import pickle
import time
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as pl
from itertools import combinations
from circuit_generation_multiple_ancilla import get_circuit_parameters
print(os.getcwd())
# Function to get the initial state used in the Qiskit simulation
def initial_state(L,Q,neel_state=True):
    
    if neel_state is True:
        state = np.zeros((2,)*L)
        confi = [0]*L
        for x in range(0,Q,1):
            confi[2*x] = 1
        state[tuple(confi)] = 1

    else:
        state = np.zeros((2,)*L)
        for positions in combinations(range(L), Q):
            p = [0] * L

            for i in positions:
                p[i] = 1

            state[tuple(p)] = 1 
        state = state/np.sum(state)
    return state

    # Redundant code below. I am fixing the initial to uniform superposition of all states compatible with the initial charge Q


def measurement(state,x,outcome,theta):
    #measuring site x
    state = np.swapaxes(state,x,0)

    if outcome == 1:
        state[0,:] = 0
        state[1,:] = np.sin(theta)**2 * state[1,:]
    elif outcome == -1:
        state[1,:] = np.cos(theta)**2 * state[1,:] 
    
    state = np.swapaxes(state,x,0)
    return state 


def transfer(x,y,T,outcomes,state_Q,log_Z,state_zero,L,theta,debug=False):
    """_summary_

    Args:
        x (_type_): transfer matrix is applied on qubit x, x+1
        T (_type_): Transfer matrix
        outcomes (_type_): measurement outcomes at x, x+1. outcome has 3 possible values, 0: no measurement, 1: measured outcome is 1, -1: measured outcome is 0
        state_Q (_type_): state of the system
        log_Z (_type_): list of log of the partition functions in the previous steps
        state_zero (Boolean): if true then state_Q has zero partition function/weight; the measurement outcomes are compatible with the total charge.
        L (_type_): system size
        theta: quantify the strength of the weak measurement. theta=pi/2 is proj. measurement
        debug (bool, optional): Defaults to False.

    Returns:
        state_Q: transferred state
        state_zero: if true, the partition function becomes zero
    """

    if not state_zero:
        state_Q = np.swapaxes(state_Q,y,(x+1)%L) # moving y-axis to the right of the x-axis

        if x%2 == 1: # transfer step in odd layer
            state_Q = np.moveaxis(state_Q,0,-1)
        state_Q = state_Q.reshape((4,)*(L//2))
        state_Q = np.tensordot(T,state_Q,axes=(-1,x//2))
        
        # if outcomes[0] == 1:
        #     state_Q[(0,1),:] = 0
        #     state_Q[(2,3),:] = np.sin(theta)**2 * state_Q[(2,3),:]
        # elif outcomes[0] == -1:
        #     state_Q[(2,3),:] = np.cos(theta)**2 *state_Q[(2,3),:]

        # if outcomes[1] == 1:
        #     state_Q[(0,2),:] = 0
        #     state_Q[(1,3),:] = np.sin(theta)**2 * state_Q[(1,3),:]
        # elif outcomes[1] == -1:
        #     state_Q[(1,3),:] = np.cos(theta)**2 *state_Q[(1,3),:]


        state_Q = np.moveaxis(state_Q,0,x//2)
        state_Q = state_Q.reshape((2,)*L)
        if x%2 == 1:
            state_Q = np.moveaxis(state_Q,-1,0)
        state_Q = np.swapaxes(state_Q,y,(x+1)%L)

        state_Q = measurement(state_Q,x,outcomes[0],theta)
        state_Q = measurement(state_Q,y,outcomes[1],theta)

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


def boundary_measurement(state,outcomes,theta,L,log_Z,state_zero):
    # Left boundary
    state = measurement(state,0,outcomes[0],theta)
    # Right boundary
    state = measurement(state,L-1,outcomes[1],theta)
    sQ = np.sum(state)
    if sQ == 0:
        state_zero = True
        log_Z.append(-np.inf)
    else:
        state_zero = False
        log_Z.append(np.log(sQ))
        state = state/sQ
    return state, state_zero
print('test')

def sep_scrambling(state,log_Z,L,depth,t_scr,T_matrix,seed):
    scr_param,param_list = get_circuit_parameters(seed=seed,t_scram=t_scr,L=L,depth=depth,scrambling_type='Special')
    for i in range(len(scr_param)//2):
        swap_indices = scr_param[2*i+1]
        for j in range(0,len(swap_indices),2):
            x = swap_indices[j]
            y = swap_indices[j+1]
            state, _ = transfer(x,y,T=T_matrix,outcomes=[0,0],state_Q=state,log_Z=log_Z,state_zero=False,L=L,theta=0,debug=False)

    return state


def sep_dynamics_2(data,Q,theta,t_scr,neel_state=True, decoding_protocol=0):

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
        # print(data[:,:],Q)
        if np.sum((data[-1,:]+1)) != 2*Q:
            return False

    total = 0
    p_success = []
    
    initial_state_Q = initial_state(L,Q,neel_state=neel_state)
    initial_state_Q2 = initial_state(L,Q2,neel_state=neel_state)

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

    if neel_state:
        state_Q = sep_scrambling(state=state_Q,log_Z=log_Z,L=L,depth=depth,t_scr=t_scr,T_matrix=T,seed=1)
        state_Q2 = sep_scrambling(state=state_Q2,log_Z=log_Z2,L=L,depth=depth,t_scr=t_scr,T_matrix=T,seed=1)
        
    for t in range(depth)[:-1]:
        if t%2 == 0: #even layer
            for x in range(0,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
            
                state_Q, state_Q_is_zero = transfer(x,x+1,T,outcomes,state_Q,log_Z,state_Q_is_zero,L,theta,debug=False)

                state_Q2, state_Q2_is_zero = transfer(x,x+1,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,theta)
                
                if state_Q_is_zero:
                    if decoding_protocol == 1 or decoding_protocol == 3:
                        return False
                    if state_Q2_is_zero:
                        # print(x,t)
                        return False
                

        else:
            for x in range(1,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
                debug=False

                state_Q, state_Q_is_zero = transfer(x,x+1,T,outcomes,state_Q,log_Z,state_Q_is_zero,L,theta,debug=False)

                state_Q2, state_Q2_is_zero = transfer(x,x+1,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,theta)


               
            # assert np.sum(state_Q)!=0 and not state_Q_is_zero, print('first')
            outcomes = (traj[t,0],traj[t,L-1])
            state_Q, state_Q_is_zero = boundary_measurement(state_Q,outcomes,theta,L,log_Z,state_Q_is_zero)
            # assert np.sum(state_Q)!=0, print('second',state_Q_is_zero,np.sum(state_Q))
            # assert np.sum(state_Q)!=0 and not state_Q_is_zero, print('first')

            state_Q2, state_Q2_is_zero = boundary_measurement(state_Q2,outcomes,theta,L,log_Z2,state_Q2_is_zero)
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
    
    return p_Q, state_Q


def get_sep_data(Q,L,p,depth_ratio,scrambling_type,is_noisy,t_scr,decoding_protocol=0):
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

    if scrambling_type is None:
        scrambling_label = '_no_scrambling'
        noisy_label = ''
    elif scrambling_type == 'Normal':
        scrambling_label = '_normal'
    elif scrambling_type == 'Special':
        scrambling_label = '_special'
    else:
        print("Scrambled input argument not recognized. It should be either \'Normal\', \'Special\' or None")
        return

    filedir = 'Weak measurements/data/'+aniclla_label+'qiskit_data/measurement_data_all_qubits'+ scrambling_label + noisy_label + depth_label+'/'
    
    filename = filedir +'/L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p=' + str(p)+ '_seed='+str(seed)+'.imdat'
    with open(filename,'rb') as f:
        data_raw,_,_ = pickle.load(f)

    faulty_traj = 0
    total_traj = 0
    for data in data_raw:
        result,_ = sep_dynamics_2(data[0],Q,p,t_scr=t_scr,neel_initial_state=neel_state,decoding_protocol=decoding_protocol)
        if result is False:
            faulty_traj += data[1]
        else:
            p_suc.extend([result]*data[1])
        total_traj += data[1]

    return p_suc,faulty_traj/total_traj

neel_state = False
neel_label = ''
if neel_state:
    neel_label = '_neel'

MULTIPLE_ANCILLA = False
aniclla_label = ''
if MULTIPLE_ANCILLA:
    aniclla_label = 'multiple_ancilla/'


def run(t_scram=5):
    p_list = np.round(np.linspace(0,np.pi/2,10),3)[:]
    p_list = np.round(np.linspace(0.2,0.6*np.pi/2,15),3)[:]
    p_list = list(np.round(np.linspace(0.2*np.pi/2,0.8*np.pi/2,15),3))
    p_list.extend(list(np.array([0.001,0.01,0.02,0.05,0.1,0.15])*np.pi/2))
    L_list = [6,8,10,12,14,16][:5]
    p_suc_dic = {}
    scrambling_type = 'Special'
    depth_ratio=0.5
    if depth_ratio != 1:
        depth_label = '_depth_ratio='+str(depth_ratio) + '/'
    else:
        depth_label = '/'
    

    for is_noisy in [False,True]:
        for decoding_protocol in [0,3][:]:
            
            final_file = 'Weak measurements/data/'+aniclla_label+'sep_data'+neel_label + depth_label+ 'seed=1_all_qubits'
            individual_files = 'Weak measurements/data/'+aniclla_label+'sep_data' + neel_label + depth_label

            if t_scram == 2:
                final_file = 'Weak measurements/t_scram=2/data/'+aniclla_label+'sep_data'+neel_label + depth_label+ 'seed=1_all_qubits'
                individual_files = 'Weak measurements/t_scram=2/data/'+aniclla_label+'sep_data'+neel_label + depth_label

            if scrambling_type == 'Special':
                final_file = final_file + '_special_scrambled'
                individual_files = individual_files + 'special/'
            elif scrambling_type == 'Normal':
                final_file = final_file + '_normal_scrambled'
                individual_files = individual_files + 'normal/'

            if is_noisy:
                final_file = final_file + '_decoding_protocol='+str(decoding_protocol)
                final_file = final_file + '_noisy'
                individual_files = individual_files + 'noisy_decoding_protocol='+str(decoding_protocol)+'/'
            else:
                individual_files = individual_files + 'not_noisy/'


            for L in L_list:
                files = individual_files + 'L='+str(L)+'/'
                if not os.path.isdir(files):
                    os.makedirs(files)
                p_suc_dic[L] = {}
                for p in p_list:
                    files_p = files + 'p='+str(p)
                    p_suc_dic[L][p] = {}
                    data = {}
                    try:
                        with open(files_p,'rb') as f:
                            data = pickle.load(f)
                        print(L,p,data.keys(),len(data[L//2]))
                    except:
                        data = {}
                        for Q in [L//2,L//2-1][:]:
                            start = time.time()
                            data[Q],temp = get_sep_data(Q,L,p,depth_ratio=depth_ratio,scrambling_type=scrambling_type,is_noisy=is_noisy, t_scr=t_scram,decoding_protocol=decoding_protocol)
                            print(L,p,Q," frac of faulty traj:",temp," time=",time.time()-start,'\n',"is_noisy:",is_noisy,' decoding_protocol:',decoding_protocol)
                        with open(files_p,'wb') as f:
                            pickle.dump(data,f)
                    p_suc_dic[L][p] = data

                with open(final_file,'wb') as f:
                    pickle.dump(p_suc_dic,f)


if __name__ == '__main__':
    print('running')
    # run(t_scram=5)
# run(t_scram=2)

