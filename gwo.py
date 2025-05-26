import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(0)

def gwo(max_iterations, population_size,
        init_population_func, fitness_func, # functions
        limit_pos_func):
    plot_data = {}
    plot_data['mean_fitness']  = []
    plot_data['max_fitness']   = []
    plot_data['min_fitness']   = []

    plot_data['alpha_fitness'] = []
    plot_data['alpha_pos']     = []
    plot_data['beta_fitness']  = []
    plot_data['beta_pos']      = []
    plot_data['delta_fitness'] = []
    plot_data['delta_pos']     = []

    plot_data['alpha_time'] = []
    plot_data['alpha_cost'] = []

    positions = np.array(init_population_func(population_size))
    fitnesses = []

    alpha_pos     = positions[0]
    alpha_fitness = np.inf
    beta_pos      = positions[0]
    beta_fitness  = np.inf
    delta_pos     = positions[0]
    delta_fitness = np.inf

    for _ in range(max_iterations):
        fitnesses = [fitness_func(positions[i,:]) for i in range(positions.shape[0])]

        # NOTE(Pawel Hermansdorfer): Withouth copy, operation on positions overwrites alpha, beta, delta wolfs and best solution may be lost
        for i in range(positions.shape[0]):
            fitness = fitnesses[i]
            if fitness < alpha_fitness:
                alpha_fitness = fitness
                alpha_pos = copy.deepcopy(positions[i,:])

            if fitness > alpha_fitness and fitness < beta_fitness:
                beta_fitness = fitness
                beta_pos = copy.deepcopy(positions[i,:])
            
            if fitness > alpha_fitness and fitness > beta_fitness and fitness < delta_fitness:
                delta_fitness = fitness
                delta_pos = copy.deepcopy(positions[i,:])

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
                r1 = np.random.random() # [0,1]
                r2 = np.random.random() # [0,1]

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                d_alpha = abs(C1*alpha_pos[j] - positions[i,j])
                X1 = alpha_pos[j]-A1*d_alpha
                
                r1=np.random.random()
                r2=np.random.random()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                d_beta = abs(C2*beta_pos[j]-positions[i,j])
                X2 = beta_pos[j]-A2*d_beta

                r1 = np.random.random()
                r2 = np.random.random()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                d_delta = abs(C3 * delta_pos[j] - positions[i,j])
                X3 = delta_pos[j] - A3 * d_delta

                pos_real = (X1 + X2 + X3) / 3
                positions[i,j] = limit_pos_func(pos_real, j)

    # NOTE(Pawel Hermansdorfer): Check for better solutions after last iteration
    fitnesses = np.array([fitness_func(positions[i,:]) for i in range(positions.shape[0])])
    for position, fitness in zip(positions, fitnesses):
        if fitness < alpha_fitness:
            # NOTE(Pawel Hermansdorfer): Deepcopy isn't necessary since we wont modify positions
            alpha_fitness = fitness
            alpha_pos = position

    return alpha_fitness, alpha_pos, plot_data


def gwo_plot_result(plot_data):
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 2, width_ratios=[1, 1])


    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(plot_data['alpha_fitness'], label='Alpha Fitness', linewidth=1)
    ax1.plot(plot_data['beta_fitness'], label='Beta Fitness', linewidth=1)
    ax1.plot(plot_data['delta_fitness'], label='Delta Fitness', linewidth=1)
    ax1.plot(plot_data['mean_fitness'], label='Mean Fitness', linewidth=1)
    ax1.set_title('Fitness')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(plot_data['alpha_time'])
    ax2.set_title('Alpha time')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(plot_data['alpha_cost'])
    ax3.set_title('Alpha cost')

    plt.tight_layout()
    plt.show()

    # plt.plot(plot_data['alpha_fitness'], label='Alpha Fitness', linewidth=1)
    # plt.plot(plot_data['beta_fitness'], label='Beta Fitness', linewidth=1)
    # plt.plot(plot_data['delta_fitness'], label='Delta Fitness', linewidth=1)
    # plt.plot(plot_data['mean_fitness'], label='Mean Fitness', linewidth=1)
    # plt.ylabel('Fitness')
    # plt.xlabel('Iteration')
    # plt.legend()
    # plt.show()
