# src/problems.py
import pennylane as qml
from pennylane import numpy as np

class MaxCutTask:
    """
    Physics Benchmark: Finding Ground State of Ising Hamiltonian.
    Backend: PennyLane 'default.qubit' with finite shots.
    """
    def __init__(self, n_qubits, depth, shots=1000):
        self.n = n_qubits
        self.depth = depth
        self.shots = shots
        
        # Define Hamiltonian (Ring Topology: Z_i Z_{i+1})
        # Minimizing this Hamiltonian = Maximizing the Cut
        coeffs = [1.0] * n_qubits
        obs = [qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits) for i in range(n_qubits)]
        self.H = qml.Hamiltonian(coeffs, obs)
        
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    def get_cost_fn(self):
        @qml.qnode(self.dev)
        def circuit(params):
            # Reshape for Hardware Efficient Ansatz
            # params shape: (depth, n, 2)
            p_shaped = params.reshape(self.depth, self.n, 2)
            
            for d in range(self.depth):
                for i in range(self.n):
                    qml.RY(p_shaped[d, i, 0], wires=i)
                    qml.RZ(p_shaped[d, i, 1], wires=i)
                for i in range(self.n):
                    qml.CNOT(wires=[i, (i + 1) % self.n])
                    
            return qml.expval(self.H)

        def cost_wrapper(flat_params):
            # Ensure proper numpy typing for PennyLane
            return circuit(np.array(flat_params))
            
        n_params = self.depth * self.n * 2
        return cost_wrapper, n_params


class MakeMoonsTask:
    """
    QML Benchmark: Binary Classification (VQC).
    Backend: PennyLane with Data Re-uploading/Angle Embedding.
    """
    def __init__(self, n_samples=100, noise=0.1):
        # Generate Data Manually to avoid sklearn dependency in strict environments
        t = np.linspace(0, np.pi, n_samples // 2)
        outer_x = np.cos(t); outer_y = np.sin(t)
        inner_x = 1 - np.cos(t); inner_y = 1 - np.sin(t) - 0.5
        
        X = np.vstack([np.column_stack([outer_x, outer_y]), 
                       np.column_stack([inner_x, inner_y])])
        
        # Normalization for Angle Embedding [0, pi]
        X = (X - X.min()) / (X.max() - X.min()) * np.pi
        y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
        
        # Shuffle
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        self.X, self.y = X[indices], y[indices]

    def get_loss_fn(self, n_qubits, depth, shots=1000):
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(dev)
        def circuit(params, x_in):
            # 1. Embedding
            qml.AngleEmbedding(x_in, wires=range(n_qubits), rotation='X')
            
            # 2. Strong Entangling Ansatz
            # PennyLane's standard template for QML
            shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
            p_shaped = params.reshape(shape)
            qml.StronglyEntanglingLayers(p_shaped, wires=range(n_qubits))
            
            # 3. Measure Z on first qubit (Class probability)
            return qml.expval(qml.PauliZ(0))

        def loss_wrapper(flat_params, X_batch=None, y_batch=None):
            if X_batch is None: X_batch, y_batch = self.X, self.y
            
            loss = 0.0
            p = np.array(flat_params)
            
            for x, y_true in zip(X_batch, y_batch):
                exp_val = circuit(p, x_in=x)
                # Map Expectation [-1, 1] to Label [1, 0]
                # If y=0 (Class A), we want Z=+1. If y=1 (Class B), we want Z=-1.
                target = 1.0 if y_true == 0 else -1.0
                loss += (exp_val - target) ** 2
                
            return loss / len(X_batch)

        shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
        n_params = np.prod(shape)
        return loss_wrapper, n_params