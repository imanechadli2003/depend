from qiskit import Aer, QuantumCircuit
from qiskit.algorithms import Grover
import numpy as np

def grover_select(similarities, th=0.65):
    good = [i for i,s in enumerate(similarities) if s >= th]
    if not good: return 0                       # fallback : premier

    n = int(np.ceil(np.log2(len(similarities))))  # nb qubits (256→8)
    oracle = QuantumCircuit(n)
    for g in good:
        bits = format(g, f"0{n}b")[::-1]
        for j,b in enumerate(bits):
            if b == '0': oracle.x(j)
        oracle.mcx(list(range(n-1)), n-1)        # multi-CNOT
        for j,b in enumerate(bits):
            if b == '0': oracle.x(j)

    grover  = Grover(iterations=1)
    result  = grover.solve(oracle=oracle,
                           backend=Aer.get_backend("aer_simulator"))
    return int(result.top_measurement, 2)
