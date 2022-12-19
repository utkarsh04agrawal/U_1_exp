import numpy as np
from circuit_generation import u1gate, get_circuit_parameters
import qiskit as qk

def unitary_gate_from_params(params):
    """
    generate U_1 unitary matrix for a given set of parameters
    """
    circ = qk.QuantumCircuit(2)
    circ = u1gate(circ,params,0,1)
    simulator = qk.Aer.get_backend('unitary_simulator')
    job = qk.execute(circ,simulator)
    return job.result().get_unitary(circ,10)
PARAMS_PER_GATE = 6


def get_circuit(L,depth,scrambling_type,seed=1,t_scram=5):
    """
    Generate unitary matrices of the quantum circuit.
    L: length of the system
    depth: depth of the circuit (does not include the pre-scrambling step)
    scrambling_type: Specify the type of pre-scrambling
    seed: seed for the random numbers
    t_scram: time steps for the 
    output:
        - a nested list. Each element contains list of the gates applied at that time step. Each gate is stored in the format (x,y,U), where x,y are the two legs on which unitary U is applied.
    """
    if scrambling_type == 'Normal':
        t_scram = 2*L
    scr_param,param_list = get_circuit_parameters(seed,t_scram=t_scram,L=L,depth=depth,scrambling_type=scrambling_type)

    circuit_layer = []

    if scr_param:
        for t in range(len(scr_param)//2):
            circuit_layer.append([])
            u1_param = scr_param[2*t]
            swap_indices = scr_param[2*t+1]
            for j in range(len(u1_param)):
                circuit_layer[-1].append((swap_indices[2*j],swap_indices[2*j+1],unitary_gate_from_params(u1_param[j])))

    for t in range(depth):
        params = param_list[t]
        circuit_layer.append([])
        if t%2 == 0:
            for x in range(L//2):
                circuit_layer[-1].append((2*x,2*x+1,unitary_gate_from_params(params[x])))
        else:
            for x in range(1,L//2):
                circuit_layer[-1].append((2*x-1,2*x,unitary_gate_from_params(params[x-1])))

    return circuit_layer



def get_diffusion_constants(depth,circuit_layer):
    Diff = []
    for t in range(depth):
        Diff.append([])
        for gate in circuit_layer[t-depth]:
            x = gate[0]
            y = gate[1]
            D = np.abs(np.asarray(gate[2])[1,2])**2
            Diff[-1].append((x,y,D))
    return Diff



