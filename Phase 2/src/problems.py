import pennylane as qml
from pennylane import numpy as np
import networkx as nx
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- UTILITY METER ---
class CostTracker:
    def __init__(self, func):
        self.func = func
        self.calls = 0  # <--- The "NFE" Counter
    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.func(*args, **kwargs)

# --- 1. PHYSICS TASK (MaxCut on 3-Regular Graphs) ---
class MaxCutRegularTask:
    def __init__(self, n_qubits, depth, shots=1000, seed=42):
        self.n = n_qubits
        self.depth = depth
        self.shots = shots
        # Seed control for graph generation ensures "Frustrated" structure is consistent
        self.graph = nx.random_regular_graph(d=3, n=n_qubits, seed=seed)
        
        coeffs, obs = [], []
        for (u, v) in self.graph.edges():
            coeffs.append(0.5) # Shifted Hamiltonian (standard for 0-centered energy)
            obs.append(qml.PauliZ(u) @ qml.PauliZ(v))
            
        self.H = qml.Hamiltonian(coeffs, obs)
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    def get_cost_fn(self):
        @qml.qnode(self.dev)
        def circuit(params):
            # Hardware Efficient Ansatz
            p_shaped = params.reshape(self.depth, self.n, 2)
            for d in range(self.depth):
                for i in range(self.n):
                    qml.RY(p_shaped[d, i, 0], wires=i)
                    qml.RZ(p_shaped[d, i, 1], wires=i)
                # Entangling Ring
                for i in range(self.n - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n - 1, 0])
            return qml.expval(self.H)

        def cost_wrapper(flat_params):
            return circuit(np.array(flat_params))
        
        n_params = self.depth * self.n * 2
        tracked_cost = CostTracker(cost_wrapper)
        return tracked_cost, n_params

# --- 2. QML TASK (Make Moons) ---
class MakeMoonsTask:
    """
    Benchmarks optimizer on Non-Linear Geometric Classification.
    Matches Paper Section 3.4.
    """
    def __init__(self, n_qubits=2, n_samples=100, noise=0.1, seed=42):
        # 1. Generate Data
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        
        # 2. Rescale Features to [0, pi] for Angle Embedding
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.X = scaler.fit_transform(X)
        
        # 3. Labels: 0 -> -1 (State |1>), 1 -> +1 (State |0>) for PauliZ measurement
        self.y = np.where(y == 0, -1, 1)
        
        # 4. Train/Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=seed
        )

    def _get_circuit(self, n_qubits, depth, shots):
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        
        @qml.qnode(dev)
        def circuit(params, x_in):
            # Encoding: Angle Embedding (Preserves Distance Metrics)
            qml.AngleEmbedding(x_in, wires=range(n_qubits), rotation='Y')
            
            # Variational: Strongly Entangling Layers (High Expressivity)
            shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
            qml.StronglyEntanglingLayers(params.reshape(shape), wires=range(n_qubits))
            
            # Measurement: Expectation of Z on first qubit
            return qml.expval(qml.PauliZ(0))
            
        return circuit

    def get_loss_fn(self, n_qubits, depth, shots=1000):
        circuit = self._get_circuit(n_qubits, depth, shots)

        def loss_wrapper(flat_params, X_batch=None, y_batch=None):
            if X_batch is None: X_batch, y_batch = self.X_train, self.y_train
            
            p = np.array(flat_params)
            total_loss = 0.0
            
            # Mean Squared Error Loss
            # Prediction range [-1, 1], Target {-1, 1}
            for x, target in zip(X_batch, y_batch):
                prediction = circuit(p, x_in=x)
                total_loss += (prediction - target) ** 2
            
            return total_loss / len(X_batch)

        shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
        tracked_loss = CostTracker(loss_wrapper)
        return tracked_loss, np.prod(shape)

    def get_test_accuracy(self, params, n_qubits, depth, shots=1000):
        circuit = self._get_circuit(n_qubits, depth, shots)
        p = np.array(params)
        correct = 0
        
        for x, y_true in zip(self.X_test, self.y_test):
            pred_val = circuit(p, x_in=x)
            # Threshold sign: pred > 0 -> Class +1, pred < 0 -> Class -1
            pred_label = 1 if pred_val > 0 else -1
            
            if pred_label == y_true:
                correct += 1
                
        return correct / len(self.y_test)