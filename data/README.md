This is a directory which contains data generated using Qiskit. The data are not tracked due to potentially large sizes.

Data when the measurement locations AND unitaries are FIXED to some random realizations are stored in folder "measurement_data_fixed".

Data when the measurement locations AND unitaries are UNFIXED and vary sample to sample are stored in folder "measurement_data". The "seed" label is useless and a relic of past code.

Data when the unitaries are FIXED to some random realizations but the measurement locations are varied sampled to sampled are stored in folder "measurement_data_unitaries_fixed". The format is ./measurement_data_unitaries_fixed/param_seed=%param_seed%/ where %param_seed% is the seed value of numpy random generator used to generate random circuit parameters. This folder contains trajectory data