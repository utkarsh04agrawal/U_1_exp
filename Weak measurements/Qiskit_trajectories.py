import time
import numpy as np
import qiskit as qk
from qiskit.providers.aer.noise import NoiseModel, pauli_error, depolarizing_error, ReadoutError
from qiskit.providers.aer import AerSimulator
import pickle
import os
import circuit_generation

## Global parameters
PARAMS_PER_GATE = 6 # number parameters for general U1 2q gate


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

    noise = circuit_generation.noise_model(p_depo_1=p_depo_1,p_depo_2=p_depo_2,basis_gates_1_site=basis_gate_1_site.copy(),basis_gates_2_site=basis_gate_2_site.copy())
    
    simulator = AerSimulator(basis_gates = basis_gate_1_site+basis_gate_2_site) 
  
    """
    I tried using simulator = qk.Aer.get_backend('aer_simulator') but was not able to change basis gates. The above line with noise_model argument allows the basis gates of the simulator to be equal to that of the noise_model.
    """ 

    if circ is None:
        _,_,circ = circuit_generation.generate_u1mrc(L,depth,m_locs,params=param_list,Q=Q,theta=theta,debug=False,scrambling_type=scrambling_type,scrambling_param=scrambling_param)
        # circ.draw(output='mpl',scale=1)
        circ = qk.transpile(circ, simulator)

    if shots > 0:
        if is_noisy: 
            result = simulator.run(circ,shots=shots,noise_model=noise).result()
        else:
            result = simulator.run(circ,shots=shots).result()
        if np.sum(m_locs) == 0:
            measurement_array = [(np.zeros((depth,L)),shots)]
        else:
            measurement_array = outcome_history(result, L, depth, m_locs)

        return measurement_array,circ
    
    else:
        return [],circ

def minimal_working_example():
    L=6
    depth=2
    Q=3
    p=0.677
    seed=1
    m_locs = np.array([[1,1]*(L//2) for t in range(depth-1)])

    circ_file_dir = 'circ_data/special_scrambling/basis_gate_set='+str(1) + '/'
    circ_file = circ_file_dir + 'L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p='+str(p)+'_seed='+str(seed)+'.imdat'
    if os.path.isfile(circ_file):
        print('hurray')
        with open(circ_file,'rb') as f:
            circ_data = pickle.load(f)
            circ = circ_data['circuit']
            basis_gate_1_site = circ_data['basis_gate_1_site']
            basis_gate_2_site = circ_data['basis_gate_2_site']

    a,b = quantum_trajectories(L=L,depth=depth,Q=Q,theta=p,shots=20,m_locs=m_locs,p_depo_2=0.003,p_depo_1=0.0001,circ=circ,basis_gate_1_site = ['id','sx','u1','u2','u3','rz'], basis_gate_2_site = ['cx','swap'],is_noisy=False,scrambling_type='Special')
    return a,b

# a,b = minimal_working_example()

def get_trajectories(L,depth,Q,theta,m_locs,shots,seed,filedir,p_depo_1,p_depo_2,
t_scram,scrambling_type,is_noisy):

#----------------------------------------------------------------------------------------------------#
    def set_circuit_variables():
        basis_gate_set = 1
        if basis_gate_set == 1:
            basis_gate_1_site = ['id','sx','u1','u2','u3','rz']
            basis_gate_2_site = ['cx','swap']

        root_dir = 'Weak measurements/data/circ_data/'
        if scrambling_type == 'Special':
            circ_file_dir = root_dir + 'special_scrambling/basis_gate_set='+str(basis_gate_set) + '/'
        elif scrambling_type == 'Normal':
            circ_file_dir = root_dir + 'normal_scrambling/'
        else:
            circ_file_dir = root_dir + 'no_scrambling/'
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
            circ = None
            circ_data['basis_gate_1_site'] = basis_gate_1_site
            circ_data['basis_gate_2_site'] = basis_gate_2_site

        return circ, circ_file, circ_data, basis_gate_1_site, basis_gate_2_site
#----------------------------------------------------------------------------------------------------#

    filename = filedir+'L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p='+str(theta)+'_seed='+str(seed)
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            measurement_array, m_locs, param_list = pickle.load(f)
    else:
        measurement_array = []

    scr_param, param_list = circuit_generation.get_circuit_parameters(seed=seed,t_scram=t_scram,L=L,depth=depth,scrambling_type=scrambling_type)
    
    circ, circ_file, circ_data, basis_gate_1_site, basis_gate_2_site = set_circuit_variables()

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

    def get_filedir():
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

        filedir = 'Weak measurements/data/qiskit_data/measurement_data_all_qubits'+ scrambling_label + noisy_label + depth_label+'/'

        if not os.path.isdir(filedir):
            os.makedirs(filedir)

        return filedir

    for L in L_list:

        if scrambling_type == 'Normal':
            t_scram = 2*L

        for p in p_list:

            start = time.time()
            T = int(L*depth_ratio)
            m_locs = np.array([[1,1]*(L//2) for t in range(T-1)])
         
            filedir = get_filedir()

            get_trajectories(L=L,depth=T,Q=L//2,theta=p,m_locs=m_locs,seed=seed,shots=samples,filedir=filedir,p_depo_1=p_depo_1,p_depo_2=p_depo_2,t_scram=t_scram,scrambling_type=scrambling_type,is_noisy=is_noisy)

            get_trajectories(L=L,depth=T,Q=L//2-1,theta=p,m_locs=m_locs,seed=seed,shots=samples,filedir=filedir,p_depo_1=p_depo_1,p_depo_2=p_depo_2,t_scram=t_scram,scrambling_type=scrambling_type,is_noisy=is_noisy)
            print(L,p,time.time()-start)


p_list = np.round(np.linspace(0.2,0.6*np.pi/2,15),3)[:]
L_list = [6,8,10,12,14][:4]
scrambling_type = 'Special'
is_noisy = False
if is_noisy:
    p_depo_1 = 0.0001
    p_depo_2 = 0.003
else:
    p_depo_1 = 0
    p_depo_2 = 0

t_scram = 5
collect_fixed_data(L_list,p_list,1,samples=1,depth_ratio=1,p_depo_1=p_depo_1,p_depo_2=p_depo_2,scrambling_type=scrambling_type,is_noisy=is_noisy,t_scram=t_scram)


# basis_gate_1_site = ['u1','u2','u3','rz','sx','id','x']
# basis_gate_2_site = ['cx','swap]
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