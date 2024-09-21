## Structure

U_1_exp
- Projective measurements (code for doing the exp with proj. measurements. Structure not up to date)  
- Weak measurements (code implementing weak measurements)
    - Sharpening (study sharpening transition for weak measurements)
    - Purifiction (study entanglement transition for weak measurements)
    - [Obsolete] -- Noisy, Not_noisy
- Quantinuum (code for running the circuit on Quantinuum hardware)
    - There is a README file inside that explains the technical points regarding this part of the project.


## Data struture
- data 
    - qiskit_data (trajectory data using Qiskits)
    - circ_data (data for transpiled qiskit circuit)
    - sep_data (data after running SEP decoder on the qiskit_data)
        - post_selection protocols (P_1 = prob. for the correct charge Q1, P_2 = prob. of incorrect charge Q2)
            - 0 -- either P_1 or P_2 remain non-zero
            - 1 -- P_1 remain non-zero all the time. P_suc is proportional to P_1. So we are postselcting on P_suc being non zero
            - 2 -- Final charge is equal to the initial charge (projective measurements are done on the final charge)
            - 3 -- Combination of 1 and 2
    - biased_data (data after running biased decoder on the qiskit_data)

- data_quantum_decoder (data directory for quantum decoder)

- qiskit_data (label information):
    - "all_qubits" labels mean that all qubits are weakly measured
    - "no_scrambling" label mean that there was no scrambling before measurements
    - "normal" means normal nearest-neighbor scrambling
    - "special" means special all-to-all scrmabling condusive to Ion-trap experiments
    - "noisy" means the data is with noise
    - "depth_ratio" is the ratio T/L, where T is the number of odd AND even layers
    - seed=1 means that the unitary circuit was generated using seed=1 for the random numbers

- Common used code files - 
    - circuit_generation.py - contains relevant functions to generate and help in generating qiskit cirucit
    - Qiskit_trajectories.py - make the qiskit circuit and collect various trajectories based on the parameters one want to run (e.g normal vs special scrmabling, noisy vs not noisy)



- multiple_ancilla label and directory is associated with the case where the multiple ancilla qubits are used to perform weak measurements. This saves runtime on quantinuum hardware.



## Few observations:
1. It was taking very long time for compiling the circuit when I was doing random pre-scrambling (scrambling prior to doing measurements). I was doing this scrambling via Haar random unitary which were not decomposed into basis gates. But doing scrambling by decomposing them from the start is much faster. This is because the qiskit compiler doesn't need to transpile the scrambling steps (if it needs to, it is not much).

2. I was using "simulator = qk.Aer.get_backend('aer_simulator')" to initiate the simulator. But I couldn't figure out how to change the basis gates of the simulator to something custom. I found the solution here: https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html#Executing-a-noisy-simulation-with-a-noise-model. Now I am using "simulator = AerSimulator(noise_model=noise)"

3. Circuit barrier does not allow qiskit transpiler to optimize gates on the either side of the barrier.

4. Long range interactions on Quantinumm compiler: The Quantinumm compiler will see long range SWAP as just another interaction and will decompose it in CNOTs. So to implement long range interactions, instead of decomposing it in SWAP and nearest neighbor gate, we should apply long range gate in the Qiskit circuit; the Quantinumm compiler will automatically decomposed it in SWAP and gates.