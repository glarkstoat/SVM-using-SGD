import numpy as np

class RFF():
    def __init__(self, n_components=100, sigma=1):

        """
        Class to generate random fourier features
        params:
        n_components = 100
        sigma = 1
        """

        self.n_components = n_components
        self.sigma = sigma
        self.weight = None
        self.fitted = False


    def fit(self, X, y = None):

        """
        Generates random weights and offsets according to the RBF
        """

        self.weight = 1 / self.sigma * np.random.normal(size=(X.shape[1], self.n_components))
        self.b = np.random.uniform(0, 2 * np.pi, size = self.n_components)

        self.fitted = True

        return self


    def transform(self, X):

        """
        Calculates the projection of the data
        """

        if self.fitted == False:
            print('Needs to be fitted first.')
            return None

        proj = np.dot(X, self.weight)
        proj += self.b
        proj = np.cos(proj)
        proj *= np.sqrt(2 / self.n_components)

        return proj


    def kernel(self, X):

        """
        Calculates the kernel matrix to compare with the true RBF kernel
        """

        if self.fitted == False:
            print('Needs to be fitted first.')
            return None

        Z = self.transform(X)
        K = Z.dot(Z.T)

        return K
