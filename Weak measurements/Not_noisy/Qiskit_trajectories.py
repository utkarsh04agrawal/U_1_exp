import time
import numpy as np
import qiskit as qk
from qiskit.providers.aer.noise import NoiseModel, pauli_error, depolarizing_error, ReadoutError
from qiskit.providers.aer import AerSimulator
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

aa = ['id', 'h', 'rx', 'rz', 'cz']
print(aa.copy().remove('id'))

def noise_model(p_depo_1,p_depo_2,basis_gates_1_site,basis_gates_2_site):
    """
    p_depo_1: probability of applying depolatization channel on single qubit gates
    p_depo_2: probability of applying depolatization channel on two qubit gates
    What are the basis states? Are there two-site gate errors?
    """
   
    noise = NoiseModel(basis_gates_1_site+basis_gates_2_site)
    basis_gates_1_site.remove('id')
    error_depo_1 = depolarizing_error(p_depo_1,1)
    error_depo_2 = depolarizing_error(p_depo_2,2)
    noise.add_all_qubit_quantum_error(error_depo_1,basis_gates_1_site)
    noise.add_all_qubit_quantum_error(error_depo_2,basis_gates_2_site)
    return noise


# Function to generate a random circuit
def generate_u1mrc(L,depth,m_locs,params,Q,theta,debug=False,scrambled=False):
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
        - qiskit circuit of appropriate
    """
    assert Q <= L//2
    qreg = qk.QuantumRegister(L+1,'q') #L+1 qubit is for ancilla to perform weak measurement
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


def quantum_trajectories(L,depth,Q,theta,shots,m_locs, param_list,scrambled=False):
    # Create a circuit instance
    # Circuit geometry

    # Draw circuit
    # print('Remember to set debug=FALSE to generate actual circuits...')

    _,_,circ = generate_u1mrc(L,depth,m_locs,param_list,Q,theta,debug=False,scrambled=scrambled)
    # circ.draw(output='mpl',scale=1)
    noise = noise_model(0.5,0.5)

    simulator = AerSimulator(noise_model=noise) 
    """
    I tried using simulator = qk.Aer.get_backend('aer_simulator') but was not able to change basis gates. The above line with noise_model argument allows the basis gates of the simulator to be equal to that of the noise_model.
    """    

    circ = qk.transpile(circ, simulator,noise_model=noise)

    result = simulator.run(circ,shots=shots).result()
    if np.sum(m_locs) == 0:
        measurement_array = [(np.zeros((depth,L)),shots)]
    else:
        measurement_array = outcome_history(result, L, depth, m_locs)

    return measurement_array


# print(measurement_array[2])

aa = [1,2,3]
aa.extend([[1,2]])
aa

def get_trajectories(L,depth,Q,theta,m_locs,shots,seed,filedir,scrambled=False):
    

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

    

    new_trajectories = quantum_trajectories(L=L,depth=depth,Q=Q,theta=theta,shots=shots,m_locs=m_locs,param_list=param_list,scrambled=scrambled)
    measurement_array.extend(new_trajectories)

    with open(filename,'wb') as f:
        pickle.dump([measurement_array,m_locs,param_list],f)


# This collects data where unitary AND locations are FIXED.
def collect_fixed_data(L_list,p_list,seed,samples,depth_ratio=1,scrambled=False):
    for L in L_list:
        for p in p_list:
            start = time.time()
            T = L*depth_ratio
            m_locs = np.array([[1,1]*(L//2) for t in range(T-1)])
         
            if depth_ratio != 1:
                depth_label= "_depth_ratio="+str(depth_ratio)
            else:
                depth_label = ""
            if not scrambled:
                filedir = 'Weak measurements/data/measurement_data_all_qubits'+depth_label+'/'
            else:
                filedir = 'Weak measurements/data/measurement_data_all_qubits_scrambled'+depth_label+'/'

            if not os.path.isdir(filedir):
                os.makedirs(filedir)

            get_trajectories(L=L,depth=T,Q=L//2,theta=p,m_locs=m_locs,seed=seed,shots=samples,filedir=filedir,scrambled=scrambled)
            get_trajectories(L=L,depth=T,Q=L//2-1,theta=p,m_locs=m_locs,seed=seed,shots=samples,filedir=filedir,scrambled=scrambled)
            print(L,p,time.time()-start)

p_list = np.round(np.linspace(0.2,0.6*np.pi/2,15),3)[:]
L_list = [6,8,10,12,14][:4]
collect_fixed_data(L_list,p_list,1,300,depth_ratio=1,scrambled=True)


basis_gate_1_site = ['u1','u2','u3','rz','sx','id','x']
basis_gate_2_site = ['cx']
circ_trial = qk.QuantumCircuit(2,2)
circ_trial.h(0)
# circ_trial.z(0)
# circ_trial.cnot(0,1)
# circ_trial.x(0)
circ_trial.save_statevector()
circ_trial.measure(0,0)
circ_trial.measure(1,1)
noise = noise_model(1,1,basis_gate_1_site,basis_gates_2_site=basis_gate_2_site)
print(noise)
simulator = AerSimulator(noise_model=noise) 
"""
How to change basis gates of the noise model
"""

circ_trial = qk.transpile(circ_trial, simulator)
circ_trial.draw()
result = qk.execute(circ_trial,simulator,shots=10000).result()
print(result.get_counts())

import numpy as np
aa = np.arange(0,16,1).reshape((4,4))
print(aa[(0,1),:])
print(aa.flatten())