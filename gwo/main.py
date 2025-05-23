import numpy as np

def calculate_fitness(x):
    result = np.sum(x ** 2)
    return result


def init_population(dim, population_size):
    positions = np.random.rand(population_size,dim)*(100-(-100))-100
    return positions


def gwo(max_iterations, population_size, tasks, procs, channs):

    dim = len(tasks)

    alpha_pos = np.zeros(dim)
    alpha_fitness = np.inf
    beta_pos = np.zeros(dim)
    beta_fitness = np.inf
    delta_pos = np.zeros(dim)
    delta_fitness = np.inf

    positions = init_population(dim, population_size)

    l = 0
    for _ in range(max_iterations):
        for i in range (positions.shape[0]):
            Fitness = calculate_fitness(positions[i,:])

            if Fitness<alpha_fitness:
                alpha_fitness=Fitness
                alpha_pos=positions[i,:]

            if Fitness>alpha_fitness and Fitness<beta_fitness:
                beta_fitness=Fitness
                beta_pos=positions[i,:]
            
            if Fitness>alpha_fitness and Fitness>beta_fitness and Fitness<delta_fitness:
                delta_fitness=Fitness
                delta_pos=positions[i,:]
        
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
    return alpha_fitness, alpha_pos


# Algorithm params
max_iterations = 100
population_size = 100

best_fitness, best_solution = gwo(max_iterations,
                                  population_size,
                                  tasks,
                                  procs,
                                  channs)
