### 14 Nov 22 Mon
Qiskit traj for nothing fixed for 8,10,12, samples=2000: job id 465598
Qiskit traj for nothing fixed for L=14, samples=1000: job id 465611
Qiskit traj for nothing fixed for 8,10,12 + scrambling, samples=1000: job id 465612

Have not updated sep_decoder.py file on cluster. The updated file has modification to deal with unitaries_fixed case
Updated Qiskit_traj file on cluster (latest being introduction of scrambled boolean)

To_do: run Qiskit_traj for unitary_fixed case.

### 15 Nov 22 Tues
1. Running Qiskit data for nothing fixed with Neel initial state. L=8,10,12. dir=data/measurement_data. There already exist L=14 in there. Job id 466099
2. Running Qiskit data for unitary fixed with Neel initial state. L=8,10,12. shots=1500. dir=data/measurement_data_unitaries_fixed. Job id 466104


### 16 Nov 22 Wed
1. Running Qiskit data for nothing fixed with scrambled state but DEPTH=2*L. L=8,10,12. dir=data/measurement_data_depth_ratio=2. Job id 466726