import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import random_statevector, random_unitary, Statevector, Operator, average_gate_fidelity

class QuantumObjective:
    """
    Universal Wrapper for Phase 2 Benchmarking.
    Replaces 'benchmarks.py' from Phase 1.
    """
    def __init__(self, num_qubits=3, num_layers=1, problem_type='QML', n_shots=1024):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.problem_type = problem_type
        self.n_shots = n_shots
        self.history = []
        
        # --- 1. Define the Target (The "Teacher") ---
        if self.problem_type == 'QML':
            # Target is a random state vector (Data)
            self.target = random_statevector(2**num_qubits)
        elif self.problem_type == 'COMPILER':
            # Target is a random Unitary Gate (Algorithm)
            self.target = random_unitary(2**num_qubits)
            
        # --- 2. Define the Model (The "Student") ---
        # Hardware Efficient Ansatz (standard for Barren Plateau research)
        self.ansatz = EfficientSU2(num_qubits, reps=num_layers, entanglement='linear')

    def func(self, params):
        """
        The Cost Function: 1.0 - Fidelity.
        """
        # Bind parameters to the circuit
        # FIX: bind_parameters -> assign_parameters for Qiskit 1.0+
        qc = self.ansatz.assign_parameters(params)
        
        # --- Calculate Fidelity (Mathematically Exact) ---
        if self.problem_type == 'QML':
            # State Fidelity: |<target|predicted>|^2
            pred_state = Statevector(qc)
            fidelity = abs(pred_state.inner(self.target))**2
            
        elif self.problem_type == 'COMPILER':
            # Process Fidelity: How close is the Unitary U_pred to U_target?
            U_pred = Operator(qc)
            fidelity = average_gate_fidelity(U_pred, self.target)

        # --- Inject "Real" Shot Noise ---
        # We simulate the uncertainty of measuring this on a real chip.
        if self.n_shots is not None:
            sigma = 1.0 / np.sqrt(self.n_shots)
            noise = np.random.normal(0, sigma)
            fidelity = np.clip(fidelity + noise, 0.0, 1.0)
            
        return 1.0 - fidelity

    def get_adaptive_gradient(self, params):
        """
        YOUR RESEARCH CONTRIBUTION: QuGStep (Adaptive Finite Difference).
        Formula: epsilon ~ (Noise / Shots)^0.25
        """
        # 1. Determine Adaptive Step Size
        if self.n_shots is None:
            epsilon = 1e-6 
        else:
            # Heuristic derived from the QuGStep paper logic
            noise_est = 1.0 / np.sqrt(self.n_shots)
            # The '8' and power '0.25' come from balancing truncation vs variance error
            epsilon = (8 * noise_est**2)**0.25 
            epsilon = np.clip(epsilon, 1e-3, 0.2) # Safety bounds

        # 2. Compute Gradient
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            
            # Apply Adaptive Step Size
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            # Note: We use the noisy func() here!
            c_plus = self.func(params_plus)
            c_minus = self.func(params_minus)
            
            grad[i] = (c_plus - c_minus) / (2 * epsilon)
            
        return grad

    def get_num_params(self):
        return self.ansatz.num_parameters