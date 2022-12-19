## 6 Dec
1. Make scrambling all-to-all and scrambling.
2. Probably the bottleneck for Qiskit simulations is the transpiling step. Should save the transpiled circuit.

## 7 Dec
Above tasks Done!
Submitting noisy scrambling data on cluster
1. The noisy data has been collected. The new code is much faster as the scrambling step is decomposed from the start and the transpiler has to work less. To run SEP on it.
2. Get unitary gates to run quantum decoder and biased decoder.
3. Forgot to save the scrambling_parameter. Should figure out an efficeint way to do so.

## 8 Dec
1. Ran SEP on the noisy data. It is stored in Noisy/sep_data/seed=1_all_qubits_noisy_scrambled.
2. Still need to get unitary gates and run other decoders. 
3. Need to save scrambling_parameter.
4. The non-noisy scrambling case I have so far is with normal scrambling scheme. I should also run the modified scrambling scheme without noise as well to compare with the latest data.


### 15 Dec Thurs
1. ~~Remove SWAPs and do long range U1 gates.~~ (16 Dec)
2. ~~Biased decoder.~~ (15 Dec)
3. Quantum decoder.
4. ~~Sharpening transition with weak measurements.~~ (16 Dec)
5. ~~Update the file structure on the cluster~~ (16 Dec)
6. Run biased decoder
7. ~~run all data again after doing point 1 above~~ (16 Dec)
8. 

### 16 Dec Fri
1. ~~Save entropy of the sharpening ancilla at intermediate times.~~ (16 Dec)
2. Run sharpening data.