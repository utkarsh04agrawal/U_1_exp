## Few observations:
1. It was taking very long time for compiling the circuit when I was doing random pre-scrambling (scrambling prior to doing measurements). I was doing this scrambling via Haar random unitary which were not decomposed into basis gates. But doing scrambling by decomposing them from the start is much faster. This is because the qiskit compiler doesn't need to transpile the scrambling steps (if it needs to, it is not much).

2. I was using "simulator = qk.Aer.get_backend('aer_simulator')" to initiate the simulator. But I couldn't figure out how to change the basis gates of the simulator to something custom. I found the solution here: https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html#Executing-a-noisy-simulation-with-a-noise-model. Now I am using "simulator = AerSimulator(noise_model=noise)"

3. Circuit barrier does not allow qiskit transpiler to optimize gates on the either side of the barrier.

4. Long range interactions on Quantinumm compiler: The Quantinumm compiler will see long range SWAP as just another interaction and will decompose it in CNOTs. So to implement long range interactions, instead of decomposing it in SWAP and nearest neighbor gate, we should apply long range gate in the Qiskit circuit; the Quantinumm compiler will automatically decomposed it in SWAP and gates.