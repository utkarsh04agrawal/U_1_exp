import time
import numpy as np
import qiskit as qk
from qiskit.providers.aer.noise import NoiseModel, pauli_error, depolarizing_error, ReadoutError
from qiskit.providers.aer import AerSimulator
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
def scrambled_state(depth,L,Q,debug=False,BC='PBC'):
    """
    L is system size. Depth is the depth of scrambling circuit. Q is total charge and is less than equal to int(L/2)
    """
    assert Q <= int(L/2)
    T_scram = depth
    filename = 'Weak measurements/data/scrambled_states/L='+str(L)+'_T='+str(T_scram)+'_Q='+str(Q)
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            state = pickle.load(f)
        return state

    q = qk.QuantumRegister(L+1, 'q') # There are L+1 qubits as the last qubit belongs to ancilla
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
            circ.append(U,[q[0],q[-2]])
            
            
    backend = qk.Aer.get_backend('statevector_simulator')
    job = qk.execute(circ,backend=backend)
    state = np.asarray(job.result().data()['statevector'])
    with open(filename,'wb') as f:
        pickle.dump(state,f)
    return state


def noisy_scrambling(circ,qreg,scr_param):
    ## scr_param is a list with alternate parameters about a even layer of u1 gates and random swaps.
    for t in range(len(scr_param)//2):
        u1_param = scr_param[2*t]
        for j in range(len(u1_param)):
            circ = u1gate(circ,u1_param[j],qreg[2*j],qreg[2*j+1])
        swap_param = scr_param[2*t+1]
        for j in range(0,len(swap_param),2):
            circ.swap(swap_param[j],swap_param[j+1])
    

def weak_measurement(circ,q,q_a,theta):
    """
    circ: quantum circuit
    q: the systen qubit to be measured
    q_a: ancilla qubit
    theta: strength of measurement; theta=\pi is projective measurement
    To implement exp{-i*theta/2 [1-Z_q]X_qa} = exp{-i*theta X_qa/2} exp{i*theta/2 Z_q*X_qa}
    """
    # temp = qk.QuantumCircuit(1)
    # temp.rx(2*theta,0)
    # crx = temp.to_gate().control(1)
    # circ.append(crx,[q,q_a])
    # return circ

    # Doing exp{-i*theta X_qa/2}
    circ.rx(theta,q_a)

    ##### Doing exp{i*theta/2 Z_q*X_qa} ########
    circ.h(q_a) # Rotating X_qa to Z_qa

    # apply exp{i*theta/2 Z_q*Z_qa}
    circ.cnot(q,q_a)
    circ.rz(-theta,q_a)
    circ.cnot(q,q_a)

    circ.h(q_a)

    return circ


def noise_model(p_depo_1,p_depo_2,basis_gates_1_site,basis_gates_2_site):
    """
    p_depo_1: probability of applying depolatization channel on single qubit gates
    p_depo_2: probability of applying depolatization channel on two qubit gates
    What are the basis states? Are there two-site gate errors?
    """
   
    noise = NoiseModel(basis_gates_1_site+basis_gates_2_site)
    basis_gates_1_site.remove('id')
    if 'swap' in basis_gates_2_site: basis_gates_2_site.remove('swap')
    error_depo_1 = depolarizing_error(p_depo_1,1)
    error_depo_2 = depolarizing_error(p_depo_2,2)
    noise.add_all_qubit_quantum_error(error_depo_1,basis_gates_1_site)
    noise.add_all_qubit_quantum_error(error_depo_2,basis_gates_2_site)
    return noise


# Function to generate a random circuit
def generate_u1mrc(L,depth,m_locs,params,Q,theta,scrambling_type,scrambling_param=[],debug=False):
    """
    inputs:
        - L, int, system size
        - depth, int, number of circuit layers (one layer = even or odd bond gates, not both)
        - m_locs, np.array of bools, m_locs[i,j]=1 => add measurement after layer i and on qubit j
        - params, nested list of circuit parameters, 
            params[i][j] is an np.array of PARAMS_PER_GATE=6 floats
            specifying the circuit parameters for the jth gate in layer i (counting from the left of the circuit)
        - Q is the total charge in the system. Q has to be less than equal to L//2
        - theta is the strength of weak measurement. theta=0 is no measurement, theta=\pi is projective measurement.          
        - debug, bool, if True replaces u1 gates with cz gates and adds barriers so that you can visualize more easily
    outputs:
        - qiskit circuit
    """
    assert Q <= L//2
    qreg = qk.QuantumRegister(L+1,'q') #L+1 qubit is for ancilla to perform weak measurement
    creg_list = [qk.ClassicalRegister(L,'c'+str(j)) for j in range(depth)] # for measurement outcomes

    # add the registers to the circuit
    circ = qk.QuantumCircuit(qreg)
    for reg in creg_list:
        circ.add_register(reg)
    
    if scrambling_type == 'Normal':
        initial_state = scrambled_state(2*L,L,Q)
        circ.initialize(initial_state)

    elif scrambling_type == 'Special': #Ideal is for the case where we have special scrambling but there are no errors. This is to compare the Noisy data against that without noise.
        t_scram = len(scrambling_param)//2
        if t_scram == 0:
            print("No scrambling paramters provided for noisy scrambling")
        noisy_scrambling(circ,qreg,scrambling_param)

    elif scrambled is None:
        for i in range(0,Q,1):
            circ.x(qreg[2*i])
    else:
        print("Scrambled input argument not recognized. It should be either \'Normal\', \'Special\' or None")

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
                    circ = weak_measurement(circ,qreg[j],qreg[L],theta)
                    circ.measure(qreg[L],creg_list[i][j])
                    circ.reset(qreg[L])
                    # circ.x(qreg[j]).c_if(creg_list[i][j], 1)
        if debug: circ.barrier()
        # circ.save_statevector(str(i))

    # final measurements. These are projective measurement.
    circ.measure(qreg[:L],creg_list[i])
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


def quantum_trajectories(L,depth,Q,theta,shots,m_locs,p_depo_1,p_depo_2,basis_gate_1_site,basis_gate_2_site,scrambling_type,is_noisy,circ=None,scrambling_param = [],param_list=[],seed=None):
    # Create a circuit geometry and execute to get # of trajectories=shots

    noise = noise_model(p_depo_1=p_depo_1,p_depo_2=p_depo_2,basis_gates_1_site=basis_gate_1_site.copy(),basis_gates_2_site=basis_gate_2_site.copy())
    if is_noisy:
        simulator = AerSimulator(noise_model=noise) 
    else:
        simulator = AerSimulator() 
    """
    I tried using simulator = qk.Aer.get_backend('aer_simulator') but was not able to change basis gates. The above line with noise_model argument allows the basis gates of the simulator to be equal to that of the noise_model.
    """ 

    if circ is None:
        _,_,circ = generate_u1mrc(L,depth,m_locs,param_list-param_list,Q=Q,theta=theta,seed=seed,debug=False,scrambling_type=scrambling_type,scrambling_param=scrambling_param)
        # circ.draw(output='mpl',scale=1)
        circ = qk.transpile(circ, simulator)

    if shots > 0:
        result = simulator.run(circ,shots=shots).result()
        if np.sum(m_locs) == 0:
            measurement_array = [(np.zeros((depth,L)),shots)]
        else:
            measurement_array = outcome_history(result, L, depth, m_locs)

        return measurement_array,circ
    
    else:
        return [],circ


# L=6
# depth=6
# Q=3
# p=0.677
# seed=1
# m_locs = np.array([[1,1]*(L//2) for t in range(depth-1)])

# circ_file_dir = 'Weak measurements/circ_data/scrambling/basis_gate_set='+str(1) + '/'
# circ_file = circ_file_dir + 'L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p='+str(p)+'_seed='+str(seed)+'.imdat'
# if os.path.isfile(circ_file):
#     with open(circ_file,'rb') as f:
#         circ_data = pickle.load(f)
#         circ = circ_data['circuit']
#         basis_gate_1_site = circ_data['basis_gate_1_site']
#         basis_gate_2_site = circ_data['basis_gate_2_site']

# a,b = quantum_trajectories(L=L,depth=depth,Q=Q,theta=p,shots=2,m_locs=m_locs,p_depo_2=0.003,p_depo_1=0.0001,circ=circ,basis_gate_1_site = ['id','sx','u1','u2','u3','rz'], basis_gate_2_site = ['cx'])
# a

def get_trajectories(L,depth,Q,theta,m_locs,shots,seed,filedir,p_depo_1,p_depo_2,
t_scram,scrambling_type,is_noisy):

    filename = filedir+'L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p='+str(theta)+'_seed='+str(seed)
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            measurement_array, m_locs, param_list = pickle.load(f)
    else:
        measurement_array = []

    rng = np.random.default_rng(seed=seed)
    # generate random circuit parameters
    # each layer has L//2
    #params = 4*np.pi*np.random.rand(depth,L//2,PARAMS_PER_GATE)
    param_list = [[4*np.pi*rng.uniform(0,1,PARAMS_PER_GATE) 
                for j in range(L//2-(i%2))] # there are either L//2 or L//2-1 gates for even/odd layers resp.
                    for i in range(depth)]

    ## generate pre-scrambling parameters; This is generated after the monitored circuit parameters so that the monitored dynamics can be reproduced independent of the scrambling protocols.
    scr_param = []
    if scrambling_type == 'Ideal':
        indices = list(range(0,L,1))
        for t in range(t_scram):
            scr_param.append([4*np.pi*rng.uniform(0,1,PARAMS_PER_GATE) # unitary layer
                for j in range(L//2)])
            rng.shuffle(indices)
            scr_param.append(indices) # this tells how to act the SWAP layer, indices[0],indices[1] are swapped and so on
    
    basis_gate_set = 1
    if basis_gate_set == 1:
        basis_gate_1_site = ['id','sx','u1','u2','u3','rz']
        basis_gate_2_site = ['cx','swap']


    if scrambling_type == 'Ideal':
        circ_file_dir = 'Weak measurements/circ_data/special_scrambling/basis_gate_set='+str(basis_gate_set) + '/'
    elif scrambling_type == 'Normal':
        circ_file_dir = 'Weak measurements/circ_data/normal_scrambling/'
    else:
        circ_file_dir = 'Weak measurements/circ_data/no_scrambling/'

    if not os.path.isdir(circ_file_dir):
        os.makedirs(circ_file_dir)

    circ_file = circ_file_dir + 'L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p='+str(theta)+'_seed='+str(seed)+'.imdat'
    if os.path.isfile(circ_file):
        with open(circ_file,'rb') as f:
            circ_data = pickle.load(f)
            circ = circ_data['circuit']
            basis_gate_1_site = circ_data['basis_gate_1_site']
            basis_gate_2_site = circ_data['basis_gate_2_site']
    else:
        circ_data = {}
        circ=None
        circ_data['basis_gate_1_site'] = basis_gate_1_site
        circ_data['basis_gate_2_site'] = basis_gate_2_site


    new_trajectories,circ = quantum_trajectories(L=L,depth=depth,Q=Q,theta=theta,shots=shots,m_locs=m_locs,param_list=param_list,seed=seed,
    p_depo_1=p_depo_1,p_depo_2=p_depo_2,
    basis_gate_1_site = basis_gate_1_site,
    basis_gate_2_site = basis_gate_2_site,
    circ=circ,scrambling_type=scrambling_type,is_noisy=is_noisy,scrambling_param=scr_param)

    # Saving transpiled circuit
    circ_data['circuit'] = circ
    if not os.path.isfile(circ_file):
            with open(circ_file,'wb') as f:
                pickle.dump(circ_data,f)

    if shots>0:
        measurement_array.extend(new_trajectories)
        with open(filename,'wb') as f:
            pickle.dump([measurement_array,m_locs,param_list],f)


# This collects data where unitary AND locations are FIXED.
def collect_fixed_data(L_list,p_list,seed,samples,p_depo_1,p_depo_2,t_scram,scrambling_type,is_noisy,depth_ratio=1):
    for L in L_list:
        for p in p_list:
            start = time.time()
            T = L*depth_ratio
            m_locs = np.array([[1,1]*(L//2) for t in range(T-1)])
         
            if depth_ratio != 1:
                depth_label= "_depth_ratio="+str(depth_ratio)
            else:
                depth_label = ""

            if is_noisy:
                noisy_label = '_noisy'
            else:
                noisy_label = ''

            if scrambling_type is None:
                scrambling_label = ''
            elif scrambling_type == 'Normal':
                scrambling_label = '_normal'
            elif scrambled == 'Special':
                scrambling_label = '_special'
            else:
                print("Scrambled input argument not recognized. It should be either \'Normal\', \'Special\' or None")
                return

            filedir = 'Weak measurements/Noisy/data/measurement_data_all_qubits'+ scrambling_label + noisy_label + depth_label+'/'

            if not os.path.isdir(filedir):
                os.makedirs(filedir)

            get_trajectories(L=L,depth=T,Q=L//2,theta=p,m_locs=m_locs,seed=seed,shots=samples,filedir=filedir,p_depo_1=p_depo_1,p_depo_2=p_depo_2,t_scram=t_scram,scrambling_type=scrambling_type,is_noisy=is_noisy)
            get_trajectories(L=L,depth=T,Q=L//2-1,theta=p,m_locs=m_locs,seed=seed,shots=samples,filedir=filedir,p_depo_1=p_depo_1,p_depo_2=p_depo_2,t_scram=t_scram,scrambling_type=scrambling_type,is_noisy=is_noisy)
            print(L,p,time.time()-start)


p_list = np.round(np.linspace(0.2,0.6*np.pi/2,15),3)[:10]
L_list = [6,8,10,12,14][:1]
p_depo_1 = 0.0001
p_depo_2 = 0.003
scrambled = 'Noisy'
t_scram = 5
collect_fixed_data(L_list,p_list,1,samples=0,depth_ratio=1,p_depo_1=p_depo_1,p_depo_2=p_depo_2,scrambled=scrambled,t_scram=t_scram)


# basis_gate_1_site = ['u1','u2','u3','rz','sx','id','x']
# basis_gate_2_site = ['cx']
# circ_trial = qk.QuantumCircuit(2,2)
# circ_trial.h(0)
# # circ_trial.z(0)
# # circ_trial.cnot(0,1)
# # circ_trial.x(0)
# circ_trial.save_statevector()
# circ_trial.measure(0,0)
# circ_trial.measure(1,1)
# noise = noise_model(1,1,basis_gate_1_site,basis_gates_2_site=basis_gate_2_site)
# print(noise)
# simulator = AerSimulator(noise_model=noise) 
# """
# How to change basis gates of the noise model
# """

# circ_trial = qk.transpile(circ_trial, simulator)
# circ_trial.draw()
# result = simulator.run(circ_trial,simulator,shots=10000).result()
# print(result.get_counts())

# import numpy as np
# aa = np.arange(0,16,1).reshape((4,4))
# print(aa[(0,1),:])
# print(aa.flatten())