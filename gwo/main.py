import numpy as np
import matplotlib.pyplot as plt

def gwo(max_iterations, population_size,
        init_pupulation, calculate_fitness, # functions
        dim):
    plot_data = {}
    plot_data['mean_fitness'] = []
    plot_data['max_fitness'] = []
    plot_data['min_fitness'] = []
    plot_data['alpha_fitness'] = []
    plot_data['alpha_pos'] = []
    plot_data['beta_fitness'] = []
    plot_data['beta_pos'] = []
    plot_data['delta_fitness'] = []
    plot_data['delta_pos'] = []

    alpha_pos = np.zeros(dim)
    alpha_fitness = np.inf
    beta_pos = np.zeros(dim)
    beta_fitness = np.inf
    delta_pos = np.zeros(dim)
    delta_fitness = np.inf

    positions = init_population(dim, population_size)

    for _ in range(max_iterations):
        fitnesses = [calculate_fitness(positions[i,:]) for i in range(positions.shape[0])]

        for i in range(positions.shape[0]):
            fitness = fitnesses[i]
            if fitness<alpha_fitness:
                alpha_fitness=fitness
                alpha_pos=positions[i,:]

            if fitness>alpha_fitness and fitness<beta_fitness:
                beta_fitness=fitness
                beta_pos=positions[i,:]
            
            if fitness>alpha_fitness and fitness>beta_fitness and fitness<delta_fitness:
                delta_fitness=fitness
                delta_pos=positions[i,:]

        plot_data['mean_fitness'].append(np.mean(fitnesses))
        plot_data['max_fitness'].append(np.max(fitnesses))
        plot_data['min_fitness'].append(np.min(fitnesses))
        plot_data['alpha_fitness'].append(alpha_fitness)
        plot_data['alpha_pos'].append(alpha_pos)
        plot_data['beta_fitness'].append(beta_fitness)
        plot_data['beta_pos'].append(beta_pos)
        plot_data['delta_fitness'].append(delta_fitness)
        plot_data['delta_pos'].append(delta_pos)
        
        a = 2-1*(2/max_iterations)
        for i in range (positions.shape[0]):
            for j in range (positions.shape[1]):
                r1=np.random.random() # [0,1]
                r2=np.random.random() # [0,1]

                A1 = 2*a*r1-a
                C1 = 2 * r2

                d_alpha = abs(C1*alpha_pos[j] - positions[i,j])
                X1 = alpha_pos[j]-A1*d_alpha
                
                r1=np.random.random()
                r2=np.random.random()

                A2 = 2*a*r1-a
                C2 = 2*r2

                d_beta = abs(C2*beta_pos[j]-positions[i,j])
                X2 = beta_pos[j]-A2*d_beta

                r1 = np.random.random()
                r2 = np.random.random()

                A3 = 2*a*r1-a
                C3 = 2*r2

                d_delta = abs(C3 * delta_pos[j] - positions[i,j])
                X3 = delta_pos[j] - A3 * d_delta

                positions[i,j] = (X1 + X2 + X3) / 3
    return alpha_fitness, alpha_pos, plot_data


# Algorithm params
max_iterations = 20
population_size = 100

def calculate_fitness(x):
    result = np.sum(x ** 2)
    return result

def init_population(dim, population_size):
    positions = np.random.rand(population_size,dim)*(100-(-100))-100
    return positions

best_fitness, best_solution, plot_data = gwo(max_iterations,
                                             population_size,
                                             init_population,
                                             calculate_fitness,
                                             2)
print(best_fitness, best_solution)

x = [i for i in range(max_iterations)]
plt.plot(x, plot_data['alpha_fitness'], label='Alpha Fitness', linewidth=1)
plt.plot(x, plot_data['beta_fitness'], label='Beta Fitness', linewidth=1)
plt.plot(x, plot_data['delta_fitness'], label='Delta Fitness', linewidth=1)
plt.plot(x, plot_data['mean_fitness'], label='Mean Fitness', linewidth=1)
plt.xticks(x)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.legend()
plt.show()
