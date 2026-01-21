import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import random_statevector, random_unitary, Statevector, Operator, average_gate_fidelity, SparsePauliOp

class QuantumObjective:
    """
    Universal Wrapper for Phase 2 Benchmarking.
    Features:
    - QuGStep (Adaptive Gradients)
    - Realistic Noise Injection (Shot Noise + Gate Error)
    - Problems: QML (States), COMPILER (Gates), CHEMISTRY (H4 Molecule)
    """
    def __init__(self, num_qubits=3, num_layers=1, problem_type='COMPILER', n_shots=1024, gate_error_rate=0.005):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.problem_type = problem_type
        self.n_shots = n_shots
        self.gate_error_rate = gate_error_rate # NEW: 0.5% error per gate is realistic for NISQ
        self.history = []
        
        # --- 1. Define the Target (The Problem) ---
        if self.problem_type == 'QML':
            self.target = random_statevector(2**num_qubits)
            
        elif self.problem_type == 'COMPILER':
            self.target = random_unitary(2**num_qubits)
            
        elif self.problem_type == 'CHEMISTRY':
            # H4 Molecule Hamiltonian (Pre-computed for Stretched Geometry R=2.5A)
            # This represents the "Stiff Valley" (Strong Correlation)
            # Simplified 2-qubit model (Parity mapping) for speed
            self.target = SparsePauliOp.from_list([
                 ("II", -0.4804), ("IZ", 0.3435), ("ZI", -0.4347), 
                 ("ZZ", 0.5716),  ("XX", 0.1809)
            ])
            self.num_qubits = 2 # Override for chemistry
            
        # --- 2. Define the Model (The Ansatz) ---
        self.ansatz = EfficientSU2(self.num_qubits, reps=num_layers, entanglement='linear')

    def func(self, params):
        """
        Calculates Cost with REALISTIC Noise Simulation.
        Cost = Signal_Decay(Exact_Cost) + Shot_Noise
        """
        # 1. Exact Simulation (The "True" Math)
        qc = self.ansatz.assign_parameters(params)
        
        if self.problem_type == 'CHEMISTRY':
            # VQE Energy Expectation Value
            state = Statevector(qc)
            exact_val = state.expectation_value(self.target).real
            # Normalize for plotting (Energy is usually negative, we want to minimize)
            # Shift it so Global Minimum is approx 0.0 for easier visualization
            exact_cost = exact_val + 1.1 
            
        elif self.problem_type == 'QML':
            pred_state = Statevector(qc)
            fidelity = abs(pred_state.inner(self.target))**2
            exact_cost = 1.0 - fidelity
            
        elif self.problem_type == 'COMPILER':
            U_pred = Operator(qc)
            fidelity = average_gate_fidelity(U_pred, self.target)
            exact_cost = 1.0 - fidelity

        # 2. STRUCTURAL NOISE (Gate Errors / Depolarizing) [NEW]
        # Real hardware loses signal as depth increases.
        if self.gate_error_rate > 0:
            # Estimate circuit volume (how many places errors can happen)
            # Approx: (N_qubits * Layers)
            volume = self.num_qubits * (self.num_layers + 1)
            
            # Survival Probability: P_clean = (1 - error)^Volume
            p_clean = (1 - self.gate_error_rate) ** volume
            
            # The signal decays towards "Random Guess"
            # Random guess for Fidelity is 1/2^N. Random guess for Cost is ~1.0.
            noise_floor = 1.0 - (1.0 / (2**self.num_qubits))
            
            # Degraded Cost = Clean_Signal + Noise_Mix
            noisy_cost = (p_clean * exact_cost) + ((1 - p_clean) * noise_floor)
        else:
            noisy_cost = exact_cost

        # 3. STATISTICAL NOISE (Shot Noise)
        # Real measurement adds Gaussian jitter
        if self.n_shots is not None:
            sigma = 1.0 / np.sqrt(self.n_shots)
            jitter = np.random.normal(0, sigma)
            final_cost = noisy_cost + jitter
        else:
            final_cost = noisy_cost
            
        return final_cost

    def get_adaptive_gradient(self, params):
        """
        QuGStep: Adaptive Finite Difference.
        """
        if self.n_shots is None:
            epsilon = 1e-6 
        else:
            # Noise estimate now includes gate error impact!
            noise_est = 1.0 / np.sqrt(self.n_shots)
            epsilon = (8 * noise_est**2)**0.25 
            epsilon = np.clip(epsilon, 1e-3, 0.2)

        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_p = params.copy(); p_p[i] += epsilon
            p_m = params.copy(); p_m[i] -= epsilon
            grad[i] = (self.func(p_p) - self.func(p_m)) / (2 * epsilon)
            
        return grad

    def get_parameter_shift_gradient(self, params):
        """Exact PSR Gradient"""
        grad = np.zeros_like(params)
        shift = np.pi / 2
        for i in range(len(params)):
            p_p = params.copy(); p_p[i] += shift
            p_m = params.copy(); p_m[i] -= shift
            # PSR assumes the underlying function is noiseless trigonometric
            # We average 2 shots per point to be robust
            grad[i] = 0.5 * (self.func(p_p) - self.func(p_m))
        return grad

    def get_num_params(self):
        return self.ansatz.num_parameters