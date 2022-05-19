import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

def exponential_annealing(T, cooldown_coeff=0.85):
    """
    :param T: current temperature
    :type: float
    :param cooldown_coeff: cooling coefficient
    :type: float
    :return: new temperature
    """
    return T * cooldown_coeff

def fast_annealing(T0, k):
    """
    :param T0: initial temperature
    :type: float
    :param k: iteration number
    :type: int
    :return: new temperature
    """
    return T0 / (1 + k)

def logarithmic_annealing(T0, k):
    """
    :param T0: initial temperature
    :type: float
    :param k: iteration number
    :type: int
    :return: new temperature
    """
    return T0 * np.log(2) / np.log(k+2)

class SimulatedAnnealing():
    def __init__(self, f, x, T, t, k_max):
        """
        :param f: function to optimize
        :type: function
        :param x: initial point
        :type: np.ndarray
        :param T: initial temperature
        :type: float
        :param t: temperature function
        :type: function
        :param k_max: maximum number of iterations
        :type: int
        """
        self.f = f
        self.x = x
        self.y = f(x)
        self.T = T
        self.t = t
        self.k_max = k_max

        self.values = [self.y]
        self.steps = [self.x]

    
    def get_new_x(self, cur_x):
        """
        :param cur_x: current point
        :type: np.ndarray
        :return: new point
        """
        new_x = cur_x + np.random.normal(0,0.01)
        return new_x
    
    def optimize(self):
        """
        :return: optimal point
        """
        cur_x = self.x
        cur_y = self.y
        T = self.T

        for k in range(self.k_max):
            new_x = self.get_new_x(cur_x)
            new_y = self.f(new_x)
            delta_y = new_y - cur_y

            # metropolis acceptance
            if delta_y <= 0 or np.exp(-delta_y / T) > np.random.random():
                cur_x, cur_y = new_x, new_y
            if new_y < self.y:
                self.x, self.y = new_x, new_y
                self.values.append(self.y)
                self.steps.append(self.x)
        return self.x
    
    def get_distance_from_global_minima(self, global_minima):
        """
        :param global_minima: global minimum
        :type: np.ndarray
        :return: distance from global minima
        """
        return np.linalg.norm(self.x - global_minima)
        
func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
global_minima = -0.19506755


# params setup
# f = McCormick
X = 0.8
startingX = X
initial_temp = 1000
iters_per_temp = 10000
temperature_func = exponential_annealing

sa = SimulatedAnnealing(func, X, initial_temp, temperature_func, iters_per_temp)
sa.optimize()

# plot y values
plt.plot(sa.values)
plt.title(f'Fitness: initial X = {startingX}')
plt.savefig('y_values2.png')
plt.show()



# plot contour
Xs = np.linspace(-1, 1, 1000)
plt.plot(Xs, func(Xs))
plt.scatter(global_minima, func(global_minima), c='k', s=100)
plt.title(f'Simulated Annealing: initial X = {startingX}')



# plot optimization
steps = np.array(sa.steps)
plt.plot(steps, func(steps), marker='x', c='red')
plt.show()


# Create GIF
steps = np.array(sa.steps)
frames = []
for i in range(steps.shape[0]):
    # Plot
    plt.plot(Xs, func(Xs))
    plt.scatter(global_minima, func(global_minima), c='k', s=100)
    plt.plot(steps[:i+1], func(steps[:i+1]), marker='x', color='r');
    plt.title(f'Simulated Annealing: initial X = {startingX}')
    # Save GIF frame to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    
    # Load and store frame using PIL
    frame = Image.open(buffer)
    frames.append(frame)
# Save frame
frames[0].save(fp='simulated_annealing2.gif', format='GIF',
               append_images=frames, save_all=True, duration=100, loop=0)

# Distance from global minima
print(f'Optimal point: {sa.x}')
print(f'Global minima: {global_minima}')
print(f'Distance from global minima: {sa.get_distance_from_global_minima(global_minima)}')
