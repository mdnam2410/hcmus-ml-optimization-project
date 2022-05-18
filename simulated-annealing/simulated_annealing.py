import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

class SimulatedAnnealing():
    def __init__(self, f, X, initial_temp, final_temp, iters_per_temp, cooldown_coeff):
        self.f = f
        self.X = X
        self.y = f(X)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.iters_per_temp = iters_per_temp
        self.cooldown_coeff = cooldown_coeff
    
        self.values = [self.y]
        self.steps = [X]

    def get_new_X(self, cur_X):
        new_X = cur_X + np.random.normal(0, 0.05, cur_X.shape)
        return new_X

    def optimize(self):
        cur_X = self.X
        cur_y = self.y
        T = self.initial_temp

        while T > self.final_temp:
            for _ in range(self.iters_per_temp):
                new_X = self.get_new_X(cur_X)
                new_y = self.f(new_X)

                # metropolis acceptance
                diff_y = new_y - cur_y
                if diff_y < 0 or np.exp(diff_y / T) > np.random.random():
                    cur_X = new_X
                    cur_y = new_y
                    if new_y < self.y:
                        self.y = new_y
                        self.X = new_X
                        self.values.append(cur_y)
                        self.steps.append(cur_X)
                        
            # exponential cooling
            T = T * self.cooldown_coeff

def McCormick(X):
    """
    McCormick function
    search space: x in [-1.5, 4], y in [-3, 4]
    """
    x, y = X
    return np.sin(x+y) + (x-y)**2 - 1.5*x + 2.5*y + 1
global_minima = np.array([-0.54719, -1.54719])


# params setup
f = McCormick
# X = np.array([0, 0])
X = np.array([1, -2])

startingX = X

initial_temp = 100
final_temp = 0.1
iters_per_temp = 100
cooldown_coeff = 0.7

sa = SimulatedAnnealing(f, X, initial_temp, final_temp, iters_per_temp, cooldown_coeff)
sa.optimize()

# plot y values
plt.plot(sa.values)
plt.axhline(y=McCormick(global_minima), color='r', linestyle='-')
plt.title('y_values')
plt.savefig('y_values2.png')
plt.show()



# plot contour
Xs = np.linspace(-1.5, 4, 1000)
Ys = np.linspace(-3, 4, 1000)
X, Y = np.meshgrid(Xs, Ys)
Z = McCormick((X, Y))
plt.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 10))
plt.scatter(global_minima[0], global_minima[1], c='w')


# plot optimization
steps = np.array(sa.steps)
plt.plot(steps[:,0], steps[:,1], marker='x', c='red')
plt.show()


# Create GIF
steps = np.array(sa.steps)
frames = []
for i in range(steps.shape[0]):
    # Plot
    plt.contourf(Xs, Ys, Z)
    plt.plot(steps[:i+1, 0], steps[:i+1, 1], marker='x', color='r');
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