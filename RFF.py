import numpy as np


class RFF:
    def __init__(self, n_components=100, sigma=1, delta=None):
        self.n_components = n_components
        self.sigma = sigma
        self.delta = delta
        self.weight = None

    def RBF(self):
        return np.exp(-np.abs(self.delta) ** 2 / (2 * self.sigma ** 2))


    def test_approx(self):

        for _ in range(self.n_components):
            w, b = self.generate(np.zeros_like(self.n_components))
            self.k_approx = np.zeros_like(self.delta)

            for i, Y in zip(range(len(self.k_approx)), self.delta):
                x = 0
                y = -Y
                zx = np.sqrt(2) / np.sqrt(self.n_components) * np.cos(w*x + b)
                zy = np.sqrt(2) / np.sqrt(self.n_components) * np.cos(w*y + b)
                self.k_approx[i] = np.dot(zx, zy.T)


    def generate(self, X):
        # Generates random weights according to RBF RFF
        try:
            self.weight = 1 / self.sigma * np.random.normal(size=(X.shape[1], self.n_components))
        
        except:
            self.weight = 1 / self.sigma * np.random.normal(size=(1, self.n_components))
        # Generates random offsets according to RBF RDD
        self.b = np.random.uniform(0, 2 * np.pi, size=self.n_components)

        return self.weight, self.b

    def transform(self, X):
        # Calculates the transformation for the approx. kernel
        # in multiple steps. Could probably done in one line as well.
        proj = np.dot(X, self.weight)
        proj += self.b
        np.cos(proj, out = proj) # ensures that in dimension = out dimension
        proj *= np.sqrt(2 / self.n_components)

        return proj
