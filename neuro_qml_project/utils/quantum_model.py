# utils/quantum_model.py
import pennylane as qml
from pennylane import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import numpy as np_c

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    for i in range(n_qubits):
        qml.RX(x[i % len(x)], wires=i)

def variational_circuit(params):
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev)
def circuit(x, params):
    feature_map(x)
    variational_circuit(params)
    return qml.expval(qml.PauliZ(0))

class QuantumClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, steps=100, lr=0.1):
        self.steps = steps
        self.lr = lr
        self.params = np.random.uniform(0, np.pi, size=(n_qubits,), requires_grad=True)

    def fit(self, X, y):
        opt = qml.GradientDescentOptimizer(stepsize=self.lr)

        for _ in range(self.steps):
            for xi, yi in zip(X, y):
                self.params = opt.step(self.cost_fn, self.params, xi, yi)
        return self

    def cost_fn(self, params, x, y):
        pred = circuit(x, params)
        return (pred - (2 * y - 1)) ** 2

    def predict(self, X):
        preds = [np.sign(circuit(x, self.params)) for x in X]
        preds_binary = [1 if p > 0 else 0 for p in preds]
        return np_c.array(preds_binary)
