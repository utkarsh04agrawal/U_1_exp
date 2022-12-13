import os
import numpy as np
import qiskit as qk
from qiskit.providers.aer.noise import NoiseModel, pauli_error, depolarizing_error, ReadoutError
import pickle

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
    basis_gates_1_site and basis_gates_2_site are the basis gates in which the circuit is transpiled to and which have some errors associated with them. For Ion-traps there are no error for 'id' and 'swap' gate.
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
        - scrambling_type specifies the type of pre-scrambling
            - None: no scrambling
            - 'Normal': linear depth scrambling with nearest neighbor interactions
            - 'Special': constant depth scrambling using random 2-body interactions and random SWAPs
        - scrambling_param: list of parameters for 'Special' constant depth scrambling.
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
    

    if scrambling_type is None:
        for i in range(0,Q,1):
            circ.x(qreg[2*i])

    elif scrambling_type == 'Normal':
        initial_state = scrambled_state(2*L,L,Q)
        circ.initialize(initial_state)

    elif scrambling_type == 'Special': #This is for the case where we have special scrambling for Ion trap setup.
        for i in range(0,Q,1):
            circ.x(qreg[2*i])
        t_scram = len(scrambling_param)//2
        if t_scram == 0:
            print("No scrambling paramters provided for noisy scrambling")
        noisy_scrambling(circ,qreg,scrambling_param)        
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
