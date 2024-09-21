from pytket.circuit import Circuit, fresh_symbol
from pytket.circuit.display import render_circuit_jupyter
from pytket.extensions.quantinuum import QuantinuumBackend


import os
print(os.getcwd())

from pytket.extensions.qiskit import qiskit_to_tk, AerBackend

import qiskit as qk

import pickle

file_loc = "Weak measurements/Sharpening/data/circ_data/normal_scrambling/L=6_depth=24_Q=3_p=0.2_seed=1.imdat"
with open (file_loc,'rb') as f:
    circ_q = pickle.load(f)


def remove_save_statevector(circ_q):
    position = 0
    count = 0
    save_positions = []
    for inst in circ_q.data:
        if inst.operation.name == 'save_statevector':
            count+=1
            save_positions.append(position)
        position+=1

    new_circ = circ_q.copy()
    for pos in reversed(save_positions):
        new_circ.data.pop(pos)

    print("Before remvoing: # of save instructions = ",count,'\n save positions = ',save_positions,'\n save positions = ',position)

    position = 0
    count = 0
    save_positions = []
    for inst in new_circ.data:
        if inst.operation.name == 'save_statevector':
            count+=1
            save_positions.append(position)
        position+=1
    print("After remvoing: # of save instructions = ",count,'\n save positions = ',save_positions,'\n save positions = ',position)
    return new_circ


new_circ = remove_save_statevector(circ_q['circuit'])
circ_p = qiskit_to_tk(new_circ)
# render_circuit_jupyter(circ_p)

machine = 'H1-1E'
backend = QuantinuumBackend(machine)
# backend.login()

print(machine, "status:", backend.device_state(device_name=machine))

compiled_circ = backend.get_compiled_circuit(circ_p)
# compiled_circ

n_shots = 1
HQC = backend.cost(compiled_circ, n_shots=n_shots, syntax_checker='H1-1SC')
print(HQC)