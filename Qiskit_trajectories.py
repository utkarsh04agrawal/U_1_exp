import time
import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import pickle
import os

## Global parameters
PARAMS_PER_GATE = 6 # number parameters for general U1 2q gate

# Function to get Haar random U(1) gate
def random_U_1_gate():
    U_11_11 = np.exp(-1j*2*np.pi*np.random.uniform())
    U_00_00 = np.exp(-1j*2*np.pi*np.random.uniform())
    U_01 = qk.quantum_info.random_unitary(2)
    U = np.zeros((4,4),dtype=complex)
    U[0,0] = U_11_11
    U[3,3] = U_00_00
    U[1:3,1:3] = U_01
    return U


# Function to create 2q gate
def u1gate(circ,gate_params,q1,q2,debug=False):
    """
    inputs: 
        circ = qiskit circuit containing q1,q2
        gate_parmas = np.array of PARAMS_PER_GATE=6 floats
        q1,q2 qiskit qubits
    returns:
        nothing, adds u1 gate directly to circ
    """
    if debug: # for debugging circuits, just put in czs for ease of visualizing
        circ.cz(q1,q2) 
    else:
        # U = qk.extensions.UnitaryGate(random_U_1_gate(),label=r'$U$')
        # circ.append(U,[q1,q2])
        # arbitrary z rotations
        circ.rz(gate_params[0],q1)
        circ.rz(gate_params[1],q2)

        # XX+YY,ZZ rotations
        circ.rz(np.pi/2,q2)
        circ.cnot(q2,q1)
        circ.rz(2*gate_params[2]-np.pi/2,q1)
        circ.ry(np.pi/2-2*gate_params[3],q2)
        circ.cnot(q1,q2)
        circ.ry(-np.pi/2+2*gate_params[3],q2)
        circ.cnot(q2,q1)
        circ.rz(-np.pi/2,q1)

        # arbitrary z rotations    
        circ.rz(gate_params[4],q1)
        circ.rz(gate_params[5],q2)

        # circ.save_unitary()
    return circ


# Function to get scrambled the initial state
def scrambled_state(depth,L,Q,debus=False,BC='PBC'):
    """
    L is system size. Depth is the depth of scrambling circuit. Q is total charge and is less than equal to int(L/2)
    """
    assert Q <= int(L/2)
    T_scram = depth
    filename = 'data/scrambled_states/L='+str(L)+'_T='+str(T_scram)+'_Q='+str(Q)
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            state = pickle.load(f)
        return state

    q = qk.QuantumRegister(L, 'q')
    c = qk.ClassicalRegister(1,'c')
    circ = qk.QuantumCircuit(q,c)

    count = 0

    while count < Q:
        circ.x(q[2*count])
        count += 1
    
    ### To study purification and sharpening, we need to add extra q_reg to circuit at location 0 and exceute below lines to entangle it with system
    # circ.h(0)
    # circ.cnot(0,1)
    # circ.cnot(0,2) # delete this line to study sharpening
    
    for t in range(T_scram):
        for i in range(0,L-1,2):
            U = qk.extensions.UnitaryGate(random_U_1_gate(),label=r'$U$')
            circ.append(U,[q[i],q[i+1]])
        for i in range(1,L-1,2):
            U = qk.extensions.UnitaryGate(random_U_1_gate(),label=r'$U$')
            circ.append(U,[q[i],q[i+1]])
        
        if BC=='PBC' and L%2==0:
            U = qk.extensions.UnitaryGate(random_U_1_gate(),label=r'$U$')
            circ.append(U,[q[1],q[-1]])
            
            
    backend = qk.Aer.get_backend('statevector_simulator')
    job = qk.execute(circ,backend=backend)
    state = np.asarray(job.result().data()['statevector'])
    with open(filename,'wb') as f:
        pickle.dump(state,f)
    return state
    

# Function to generate a random circuit
def generate_u1mrc(L,depth,m_locs,params,Q,debug=False,scrambled=False):
    """
    inputs:
        - L, int, system size
        - depth, int, number of circuit layers (one layer = even or odd bond gates, not both)
        - m_locs, np.array of bools, m_locs[i,j]=1 => add measurement after layer i and on qubit j
        - params, nested list of circuit parameters, 
            params[i][j] is an np.array of PARAMS_PER_GATE=6 floats
            specifying the circuit parameters for the jth gate in layer i (counting from the left of the circuit)
        - debug, bool, if True replaces u1 gates with cz gates and adds barriers so that you can visualize more easily
    outputs:
        - qiskit circuit of appropriate
    """
    assert Q <= L//2
    qreg = qk.QuantumRegister(L,'q')
    creg_list = [qk.ClassicalRegister(L,'c'+str(j)) for j in range(depth)] # for measurement outcomes

    # add the registers to the circuit
    circ = qk.QuantumCircuit(qreg)
    for reg in creg_list:
        circ.add_register(reg)
    
    if scrambled:
        initial_state = scrambled_state(2*L,L,Q)
        circ.initialize(initial_state)
    else:
        for i in range(0,Q,1):
            circ.x(qreg[2*i])

    # create the circuit layer-by-layer
    for i in range(depth):
        # gates
        if i%2 == 0: # even layer
            for j in range(L//2):
                circ = u1gate(circ,params[i][j],qreg[2*j],qreg[2*j+1],debug=debug)
                # circ.save_unitary()
        else: # odd layer
            for j in range(1,L//2):
                circ = u1gate(circ,params[i][j-1],qreg[2*j-1],qreg[2*j],debug=debug)
                # circ.save_unitary()
        # mid-circuit measurements
        if i<depth-1:
            for j in range(L):    
                if m_locs[i,j]:
                    # measure (implemented as measure + reset + classically-controlled X)
                    circ.measure(qreg[j],creg_list[i][j])
                    # circ.reset(qreg[j])
                    # circ.x(qreg[j]).c_if(creg_list[i][j], 1)
        if debug: circ.barrier()
        # circ.save_statevector(str(i))

    # final measurements
    circ.measure(qreg,creg_list[i])
    # circ.save_statevector(str(i+1))
    return qreg, creg_list, circ


## this function take measurement locations and outcomes to return an array of size (depth,L) with outcomes marked as +1,-1 if measured, otherwise 0
def outcome_history(circuit_results,L,depth,p_locations):
    outcomes = list(circuit_results.get_counts().items())
    measurement_data = []
    for outcome_str, count in outcomes:
        outcome_str = list(reversed(outcome_str))
        outcome_list = []
        for s in outcome_str[:]:
            if not outcome_list:
                outcome_list.append([])
            if s != ' ':
                outcome_list[-1].append(2*int(s)-1)
            else:
                outcome_list.append([])
        # print(outcome_str,'\n',outcome_list)
    #     print(p_locations, results.get_counts(circ))
        
        measurement_array = np.zeros((depth,L))

        for t in range(depth):
            if t < depth - 1:
                loc = np.where(p_locations[t])[0]
            else:
                loc = range(0,L,1)
            # print(loc,outcome_list[t])
            measurement_array[t,loc] = [outcome_list[t][i] for i in loc]
        measurement_data.append((measurement_array,count))
        
    return measurement_data


def quantum_trajectories(L,depth,Q,p,shots,m_locs, param_list,scrambled=False):
    # Create a circuit instance
    # Circuit geometry

    # Draw circuit
    # print('Remember to set debug=FALSE to generate actual circuits...')

    _,_,circ = generate_u1mrc(L,depth,m_locs,param_list,Q,debug=False,scrambled=scrambled)
    # circ.draw(output='mpl',scale=1)
    simulator = qk.Aer.get_backend('aer_simulator')
    circ = qk.transpile(circ, simulator)

    result = simulator.run(circ,shots=shots).result()
    if np.sum(m_locs) == 0:
        measurement_array = [(np.zeros((depth,L)),shots)]
    else:
        measurement_array = outcome_history(result, L, depth, m_locs)

    return measurement_array

# print(measurement_array[2])


## This changes measurement locations shot to shot
def get_trajectories(L,depth,Q,p,shots,seed,scrambled=False):
    if not scrambled:
        filedir = 'data/measurement_data/'
    else:
        filedir = 'data/measurement_data_scrambled/'

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    filename = filedir+'L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p='+str(p)+'_seed='+str(seed)
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            data = pickle.load(f)
    else:
        data = []

    for seed in range(len(data)+1,shots,1):
        rng = np.random.default_rng(seed=seed)
        # Random parts
        # measurement locations
        # entry [i,j] is layer i, qubit j
        # 1 denotes measurement, 0 = no measurement
        m_locs = rng.binomial(1,p,L*(depth-1)).reshape((depth-1,L))

        # generate random circuit parameters
        # each layer has L//2
        #params = 4*np.pi*np.random.rand(depth,L//2,PARAMS_PER_GATE)
        param_list = [[4*np.pi*rng.uniform(0,1,PARAMS_PER_GATE) 
                    for j in range(L//2-(i%2))] # there are either L//2 or L//2-1 gates for even/odd layers resp.
                    for i in range(depth)]

    

        new_trajectories = quantum_trajectories(L,depth,Q,p,1,m_locs,param_list,scrambled=scrambled)
        data.append([new_trajectories,m_locs,param_list])

    with open(filename,'wb') as f:
        pickle.dump(data,f)


## This changes measurement locations (but not unitaries) shot to shot
def get_trajectories_unitaries_fixed(L,depth,Q,p,shots,seed,param_list,param_seed,scrambled=False):
    if not scrambled:
        filename_dir = 'data/measurement_data_unitaries_fixed/param_seed='+ str(param_seed )+'/'
    else:
        filename_dir = 'data/measurement_data_unitaries_fixed_scrambled/param_seed='+ str(param_seed )+'/'
    if not os.path.isdir(filename_dir):
        os.makedirs(filename_dir)

    filename = filename_dir + 'L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p='+str(p)+'_seed='+str(seed)
    
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            data = pickle.load(f)
    else:
        data = []

    for seed in range(len(data)+1,shots,1):
        rng = np.random.default_rng(seed=seed)
        # Random parts
        # measurement locations
        # entry [i,j] is layer i, qubit j
        # 1 denotes measurement, 0 = no measurement
        m_locs = rng.binomial(1,p,L*(depth-1)).reshape((depth-1,L))

        new_trajectories = quantum_trajectories(L,depth,Q,p,1,m_locs,param_list,scrambled=scrambled)
        data.append([new_trajectories,m_locs,param_list])

    with open(filename,'wb') as f:
        pickle.dump(data,f)
    

L_list = [8,10,12]
p_list = [0.05,0.1,0.13,0.16,0.2,0.25,0.3]


def collect_nothing_fixed_data(L_list,p_list,samples,depth_ratio=1,scrambled=False):
    for L in L_list:
        for p in p_list:
            start = time.time()
            get_trajectories(L,L*depth_ratio,L//2,p,samples,1,scrambled=scrambled)
            get_trajectories(L,L*depth_ratio,L//2 - 1,p,samples,1,scrambled=scrambled)
            print(L,p,time.time()-start)


def collect_unitary_fixed_data(L_list,p_list,param_seed,samples,depth_ratio=1,scrambled=False):
    filename_dir = 'data/measurement_data_unitaries_fixed/param_seed='+ str(param_seed )+'/'

    if not os.path.isdir(filename_dir):
        os.makedirs(filename_dir)
    
    param_file = filename_dir+'param_array'
    if os.path.isfile(param_file):
        with open(param_file,'rb') as f:
            param_array = pickle.load(f)
    
    else:
        param_rng = np.random.default_rng(seed=param_seed)
        # generate and save random circuit parameters
        """ We are storing a sufficiently large array of size (500,50) though we only need size (depth,L). To get the actual parameters we slice the big array."""
        param_array = 4*np.pi*param_rng.uniform(0,1,(PARAMS_PER_GATE,500,50))
        with open(param_file,'wb') as f:
            pickle.dump(param_array,f)
    _,T_LARGE, L_LARGE = param_array.shape
    for L in L_list:
        param_list = [list(param_array[:,i,L_LARGE//2 - L//2:L_LARGE//2+L//2 - i%2]) for i in range(L*depth_ratio)]
        for p in p_list:
            start = time.time()
            get_trajectories_unitaries_fixed(L,L*depth_ratio,L//2,p,samples,1,param_list,param_seed,scrambled=scrambled)
            get_trajectories_unitaries_fixed(L,L*depth_ratio,L//2-1,p,samples,1,param_list,param_seed,scrambled=scrambled)
            print(L,p,time.time()-start)

collect_nothing_fixed_data([8],[1],samples=10,depth_ratio=1,scrambled=False)