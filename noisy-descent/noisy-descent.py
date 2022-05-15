import matplotlib.pyplot as plt
import numpy as np

class NoisyDescent:
    def __init__(self, f, x_dim, max_iter=1000, alpha=0.1, h=10e-2, diff_method='complex'):
        self.f = f
        self.x_dim = x_dim
        self.initial_point = np.random.randn(x_dim)
        self.max_iter = max_iter
        self.alpha = alpha
        self.h = h
        self.diff_method = diff_method
        
        # Iteration count
        self.k = 0
        
        self.current_x = self.initial_point
        
        # Values of f at each iteration
        self.values = [self.f(self.current_x)]
        
        # Values of x at each iteration
        self.steps = [self.current_x]
        
        self.optimized = False
        self.optimized_x = None
        
    def optimize(self):
        for _ in range(self.max_iter):
            self.k += 1
            
            # Perform noisy descent step
            grad = self._diff()
            self.current_x = self.current_x - alpha * grad + _noise()
            
            # Store values
            f_value = self.f(self.current_x)
            self.values.append(f_value)
            self.steps.append(self.current_x)
        
        self.optimized = True
        self.optimized_x = self.steps[-1]
        return self.optimized_x
    
    def _sigma(self):
        return 1/k
    
    def _noise():
        return np.random.normal(scale=self._sigma, size=(x_dim,))
    
    def _diff(self):
        if self.diff_method == 'complex':
            return _diff_complex()
        elif self.diff_method == 'numeric':
            return _diff_numeric()
        return 0
    
    def _diff_complex(self):
        H = np.eye(self.x_dim) * complex(0, self.h)
        result = np.apply_along_axis(lambda x: self.f(x).imag / h, self.current_x + H)
        return result
    
    def _diff_numeric(self):
        H = np.eye(self.x_dim) * h/2
        left = np.apply_along_axis(self.f, 1, self.current_x - H)
        right = np.apply_along_axis(self.f, 1, self.current_x + H)
        return (right - left) / h

if __name__ == '__main__':
    f = lambda x: np.sum(x ** 2)
    x = np.array(
        [[1 + 1j],
         [3 + 2j]]
    )
    h = 10e-3

    
    print(f(x))
    