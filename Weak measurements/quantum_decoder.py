import os
import time
import numpy as np
import pickle
import circuit_numpy_utils
from itertools import combinations

def initial_state(L,neel_state):
    if neel_state:
        state = np.zeros((2,)*L)
        Q_state = [1,0]*(L//2)
        Q2_state = [1,0]*(L//2 - 1) + [0,0]
        state[tuple(Q_state)] = 1/2**0.5
        state[tuple(Q2_state)] = 1/2**0.5
    else:
        state_Q = np.zeros((2,)*L)
        for positions in combinations(range(L), L//2):
            p = [0] * L
            for i in positions:
                p[i] = 1
            state_Q[tuple(p)] = 1 
        state_Q = state_Q/np.sum(np.abs(state_Q)**2)**0.5
        state_Q2 = np.zeros((2,)*L)
        for positions in combinations(range(L), L//2-1):
            p = [0] * L
            for i in positions:
                p[i] = 1
            state_Q2[tuple(p)] = 1
        state_Q2 = state_Q2/np.sum(np.abs(state_Q2)**2)**0.5

        state = (state_Q+state_Q2)/2**0.5
    return state

def separate_initial_state(L,Q):
    if neel_state:
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
        state = state/np.sum(np.abs(state)**2)**0.5
    return state


def measurement(state,x,outcome,theta):
    state = np.swapaxes(state,x,0)
    if outcome == 1:
        state[0,:] = 0
        state[1,:] = -1j * np.sin(theta) * state[1,:]
    elif outcome == -1:
        state[1,:] = np.cos(theta) * state[1,:] 
    state = np.swapaxes(state,x,0)
    return state  


def unitary_measurement(x,y,U,outcomes,state_Q,log_Z,state_zero,L,theta,do_measure,debug=False):
    """_summary_

    Args:
        x,y(_type_): transfer matrix is applied on qubit x, y

        U (_type_): Unitary to be applied at x,y
        outcomes (_type_): measurement outcomes at x, x+1. outcome has 3 possible values, 0: no measurement, 1: measured outcome is 1, -1: measured outcome is 0
        state_Q (_type_): state of the system
        log_Z (_type_): list of log of the partition functions in the previous steps
        state_zero (Boolean): if true then state_Q has zero partition function/weight; the measurement outcomes are compatible with the total charge.
        L (_type_): system size
        theta: quantify the strength of the weak measurement. theta=pi/2 is proj. measurement
        do_measure: Boolean variable. If true, perform measurement
        debug (bool, optional): Defaults to False.

    Returns:
        state_Q: transferred state
        state_zero: if true, the partition function becomes zero
    """

    if not state_zero:
              
        state_Q = np.swapaxes(state_Q,y,(x+1)%L) # moving y-axis to the right of the x-axis

        if x%2 == 1:
            state_Q = np.moveaxis(state_Q,0,-1)
        state_Q = state_Q.reshape((4,)*(L//2))
        state_Q = np.tensordot(U,state_Q,axes=(-1,x//2))

        if not do_measure:
            state_Q = np.moveaxis(state_Q,0,x//2)
            state_Q = state_Q.reshape((2,)*L)
            if x%2 == 1:
                state_Q = np.moveaxis(state_Q,-1,0)
            state_Q = np.swapaxes(state_Q,y,(x+1)%L)
            return state_Q, state_zero

        # print('Hey!',np.sum(np.abs(state_Q)[(0,1),:]**2))
        # if outcomes[0] == 1:
        #     state_Q[(0,1),:] = 0
        #     state_Q[(2,3),:] = -1j*np.sin(theta) * state_Q[(2,3),:]
        # elif outcomes[0] == -1:
        #     state_Q[(2,3),:] = np.cos(theta) *state_Q[(2,3),:]

        # if outcomes[1] == 1:
        #     state_Q[(0,2),:] = 0
        #     state_Q[(1,3),:] = -1j * np.sin(theta) * state_Q[(1,3),:]
        # elif outcomes[1] == -1:
        #     state_Q[(1,3),:] = np.cos(theta) *state_Q[(1,3),:]
        
        state_Q = np.moveaxis(state_Q,0,x//2)
        state_Q = state_Q.reshape((2,)*L)
        if x%2 == 1:
            state_Q = np.moveaxis(state_Q,-1,0)
        state_Q = np.swapaxes(state_Q,y,(x+1)%L)

        state_Q = measurement(state_Q,x,outcomes[0],theta)
        state_Q = measurement(state_Q,y,outcomes[1],theta)

        sQ = np.sum(np.abs(state_Q)**2)
        if sQ == 0:
            # print('Ahhhhhhh!',x,y,U,outcomes,np.all(state_Q==0))
            state_zero = True
            log_Z.append(-np.inf)
        else:
            state_zero = False
            log_Z.append(np.log(sQ))
            state_Q = state_Q/sQ**0.5
    else:
        log_Z.append(-np.inf)
    
    return state_Q, state_zero


def boundary_measurement(state,outcomes,theta,L,log_Z,state_zero):
    # Left boundary
    state = measurement(state,0,outcomes[0],theta)
    state = measurement(state,L-1,outcomes[1],theta)
    sQ = np.sum(np.abs(state)**2)
    if sQ == 0:
        # print('Ahhhhhhh!',x,y,U,outcomes,np.all(state_Q==0))
        state_zero = True
        log_Z.append(-np.inf)
    else:
        state_zero = False
        log_Z.append(np.log(sQ))
        state = state/sQ**0.5
    return state, state_zero



def get_indices(L,Q):
    """
    Get indices of the configuration with total charge equal to Q
    """
    count = 0
    p_list = []
    for positions in combinations(range(L), Q):
            p = [0] * L

            for i in positions:
                p[i] = 1
            count += 1
            p_list.append(np.array(p))
    
    indices = tuple(np.array(p_list).T)
    return indices

def get_probability(state,L,Q,indices=[]):
    """
    This function return probability of having charge Q in the state
    Args:
        state ((2,)*L shape array): state of the quantum system
        L (_type_): system size
        Q (_type_): Charge

    Returns:
        prob: probability of having charge Q in the state
    """
    if not indices:
        indices = get_indices(L,Q)
    prob = np.sum(np.abs(state[indices])**2)
    return prob


def quantum_dynamics_2(data,Q,theta,U_list, neel_state=True, decoding_protocol=0):
    """
    input:
        - data (_type_): 2d array of shape (depth,L) holding values of the outcomes
        - Q (_type_): initial charge of the quantum state which was used to generate the outcomes
        - neel_initial_state (bool, optional): Defaults to True. This argument is redundant now!
        - decoding_protocol
            0: No active decoding. Post-select such that both P_Q and P_Q1 are not 0
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

    indices_Q = get_indices(L,Q)
    indices_Q2 = get_indices(L,Q2)

    if decoding_protocol == 2 or decoding_protocol == 3:
        if np.sum((data[-1,:]+1)) != 2*Q:
            return False

    total = 0
    p_success = []
    
    state_is_zero = False
    state = initial_state(L,neel_state)

    N=1
    traj = data
    state_Q_is_zero = False
    state_Q2_is_zero = False
    state_Q = separate_initial_state(L,Q)
    state_Q2 = separate_initial_state(L,Q2)

    T = len(U_list) # includes t_scr and depth

    log_Z = []
    log_Z2 = []
    total += N
    t_scr = T - depth
    for t in range(T)[:-1]:
        for x,y,U in U_list[t]:
            if t >= T-depth:
                outcomes = (traj[t-t_scr,x],traj[t-t_scr,y])
        
                state_Q, state_Q_is_zero = unitary_measurement(x,y,U,outcomes,state_Q,log_Z,state_Q_is_zero,L,theta,do_measure=True,debug=False)
                
                state_Q2, state_Q2_is_zero = unitary_measurement(x,y,U,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,theta,do_measure=True,debug=False)
                
                # state,_ = unitary_measurement(x,y,U,outcomes,state,[],state_is_zero,L,theta,do_measure=True)
            
            else: #scrambling step
                if not neel_state:
                    continue
                outcomes = None
                state_Q, state_Q_is_zero = unitary_measurement(x,y,U,outcomes,state_Q,log_Z,state_Q_is_zero,L,theta,do_measure=False,debug=False)

                state_Q2, state_Q2_is_zero = unitary_measurement(x,y,U,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,theta,do_measure=False,debug=False)

                # state,_ = unitary_measurement(x,y,U,outcomes,state,[],state_is_zero,L,theta,do_measure=False)
        if t>t_scr and (t-t_scr)%2 == 1:
            outcomes = (traj[t-t_scr,0],traj[t-t_scr,L-1])

            state_Q, state_Q_is_zero = boundary_measurement(state_Q,outcomes,theta,L,log_Z,state_Q_is_zero)

            state_Q2, state_Q2_is_zero = boundary_measurement(state_Q2,outcomes,theta,L,log_Z2,state_Q2_is_zero)

            # state, _ = boundary_measurement(state,outcomes,theta,L,[],False)


        # p_Q_com = get_probability(state,L,Q,indices=indices_Q)
        # p_Q2_com = get_probability(state,L,Q2,indices=indices_Q2)
        # if p_Q_com == 0:
        #     if decoding_protocol == 1 or decoding_protocol == 3:
        #         return False
        #     if p_Q2_com == 0:
        #         return False
        # print('Combined:', Q,p_Q_com,Q2,p_Q2_com)

        if state_Q_is_zero:
            if decoding_protocol == 1 or decoding_protocol == 3:
                return False
            if state_Q2_is_zero:
                return False
        if state_Q_is_zero:
            p_Q = 0
            p_Q2 = 1
        elif state_Q2_is_zero:
            p_Q = 1
            p_Q2 = 0
        else:
            ratio_p = np.exp(np.sum(log_Z) - np.sum(log_Z2))
            p_Q = 1/(1+1/ratio_p)
            p_Q2 = 1/(1+ratio_p)
        # print('Separate:', Q,p_Q,Q2,p_Q2,ratio_p)

        # if round(p_Q_com/p_Q,6) != 1:
        #     print('Alert!',p_Q,p_Q_com)
        # p_success.append(p_Q_com)
                    
    return p_Q, state_Q


def get_data(Q,L,p,depth_ratio,scrambling_type,is_noisy,decoding_protocol=0):
    depth = int(L*depth_ratio)
    p_suc = []
    seed=1
    U_list = circuit_numpy_utils.get_circuit(L,int(depth_ratio*L),scrambling_type=scrambling_type,seed=seed,t_scram=5)
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
        result = quantum_dynamics_2(data[0],Q,p,U_list=U_list,decoding_protocol=decoding_protocol,neel_state=neel_state)

        if result is not False:
            result,_ = result
        if result is False:
            faulty_traj += data[1]
        else:
            p_suc.extend([np.array(result)]*data[1])
        total_traj += data[1]

    return p_suc,faulty_traj/total_traj

neel_state = True
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
            
            final_file = 'Weak measurements/data/'+aniclla_label+'quantum_decoder'+neel_label+depth_label+'seed=1_all_qubits'
            individual_files = 'Weak measurements/data/'+aniclla_label+'quantum_decoder'+neel_label + depth_label

            if t_scram == 2:
                final_file = 'Weak measurements/t_scram=2/data/'+aniclla_label+'quantum_decoder/seed=1_all_qubits'
                individual_files = 'Weak measurements/t_scram=2/data/'+aniclla_label+'quantum_decoder/'
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
                            data[Q],temp = get_data(Q,L,p,depth_ratio,scrambling_type=scrambling_type,is_noisy=is_noisy,decoding_protocol=decoding_protocol)
                            print(L,p,Q," frac of faulty traj:",temp," time=",time.time()-start,'\n',"is_noisy:",is_noisy,' decoding_protocol:',decoding_protocol)
                        with open(files_p,'wb') as f:
                            pickle.dump(data,f)
                    p_suc_dic[L][p] = data

                with open(final_file,'wb') as f:
                    pickle.dump(p_suc_dic,f)

if __name__ == '__main__':
    run(t_scram=5)
# run(t_scram=2)
