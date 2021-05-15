import numpy as np


class RFF:
    def __init__(self, n_components=100, gamma=1):
        self.n_components = n_components
        self.gamma = gamma
        self.weight = None

    def generate(self, X):
        # Generates random weights according to RBF RFF
        self.weight = 1 / self.gamma * np.random.normal(size=(X.shape[1], self.n_components))
        # Generates random offsets according to RBF RDD
        self.b = np.random.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        # Calculates the transformation for the approx. kernel
        # in multiple steps. Could probably done in one line as well.
        proj = np.dot(X, self.weight)
        proj += self.b
        np.cos(proj, proj)  # Some sklearn magic
        proj *= np.sqrt(2 / self.n_components)
        return proj
