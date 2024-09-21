# Problems encountered
- [x] Qiskit circuits I compiled have save_statevector commad in them. Pytket does not recognize this and throws an error.
    - Solved it by removing those instructions from the circuit.
    ```python
    position = 0
    count = 0
    save_positions = []
    # Find the location of 'save_statevector' in qiskit circuit circ_q
    for inst in circ_q.data:
        if inst.operation.name == 'save_statevector':
            count+=1
            save_positions.append(position)
        position+=1

    # create a new copy of the circuit
    new_circ = circ_q.copy()
    for pos in reversed(save_positions):
        ew_circ.data.pop(pos) # removes the 'save_statevector' instruction from the new qiskit circuit
    ```

- [x] The tket simulations were returning wrong answers when compared to Qiskit answer. The reason was that the "get_bitlist()" and "get_counts()" method have opposite order of classical registers. To be on the safe side it is better to define our own bitlist order
```python
    def get_bitlist():
        c_bits = [pytket.Bit('') for _ in range(L*depth)]
        for i in range(L*depth):
            x = i%L
            t = i//L
            temp = ['c'+str(t),[x]]
            c_bits[i] = c_bits[i].from_list(temp)
        return c_bits

    bitlist = get_bitlist()
    count_dic = result.get_counts(cbits=bitlist)
```


# Notes on using Quantinuum hardware
These are the main points I gathered from various documentation (and personal experience) that were useful for me.
- There are two ways to access the Quantinuum backend, 1) through QASM, 2) through pytket. 
## Pytket
The backends can be imported through 
```python
from pytket.extensions.quantinuum import QuantinuumBackend
 ```
 There are 3 kinds of backend:
 1. 'H1-1SC','H1-2SC'
 2. 'H1-1E', 'H1-2E'
 3. 'H1-1','H1-2': actual quantinuum hardware. 'H1' backend picks any of these two whichever is available sooner.

 - Basics of submitting a circuit to Quantinuum hardware
 1. Define a pytket circuit
    - this could be done by converting qiskit circuit to pytket circuit 
    ```python
    from pytket.extensions.qiskit import qiskit_to_tk
    circ_pktket = qiskit_to_tk(circ_qiskit)
    ``'
2. Compile the circuit to quantinuum backend 
    ```python
        machine = 'H1-1E'
        backend = QuantinuumBackend(machine)
        compiled_circ = backend.get_compiled_circuit(circ_p,optimisation_level=2)
    ```
    - optimisation_level tells the amount of optimization the compiler will do. optimisation_level=2 means that result is the most optimized circuit. optimisation_level=0 means no optimisation. Note that even if there is no optimisation requested, the actual hardware run on 'H1-1' or 'H1-2' backend will do the optimisation (unless specified not to do so when submitting the circuit through 'process_circuit' command.

    - Here is what the quantinuum documentation says:
        > Since the TKET compilation passes have been integrated into the H-Series stack, performing circuit optimizations is redundant before submitting to hardware, unless the user would like to see the optimizations applied before submitting. Given this, users may take 1 of 3 approaches when submitting jobs:
        > 1. Use `optimisation_level=0` when running `get_compiled_circuit`, then submit the circuit using `process_circuits` knowing that the corresponding optimization level actually run will be 2. 
        > 2. Use the `get_compiled_circuit` function with the desired optimization level to observe the transformed circuit before submitting and then specify `tket-opt-level=None` in the `process_circuits` function when submitting, in order for the optimizations to be applied as desired.
        >3. If the user desires to have no optimizations applied, use `optimisation_level=0` in `get_compiled_circuit` and `tket-opt-level=None` in `process_circuits`. This should be specified in both functions.

3. Calculate cost using 
    ```python
    HQC = backend.cost(compiled_circ, n_shots=n_shots, syntax_checker='H1-1SC')
    ```
    >HQC between 500 and 1200 can be exceuted in an hour


4. Submitting job to the hardware (backend = 'H1-1' or 'H1-2' or 'H1')
    - The circuit is submitted via following command
        ```python
        handle = backend.process_circuit(compiled_circuit, n_shots=n_shots, options={'tket-opt-level':2})
        ```
        (tket-opt-level=2 is the default option; it does maximum optimization. tket-opt-level':None means no optimization is performed.)

    - The status can be checked as follows:
        ```python
        status = backend.circuit_status(handle)
        print(status)
        ```

    - Getting results ...

5. Batching
