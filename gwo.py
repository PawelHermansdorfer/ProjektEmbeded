import numpy as np

POPULATION_SIZE = 100

def CalculateFitness(x):
    return np.sum(x ** 2)


def initialization (Dim,LB,UB):
    SS_Boundary = len(LB) if isinstance(UB,(list,np.ndarray)) else 1
    if SS_Boundary ==1:
        positions = np.random.rand(POPULATION_SIZE,Dim)*(UB-LB)+LB
    else:
        positions = np.zeros((POPULATION_SIZE,Dim))
        for i in range(Dim):
            positions[:,i]=np.random.rand(POPULATION_SIZE)*(UB[i]-LB[i])+LB[i]
    return positions

def GWO(MaxT,LB,UB,Dim):
    alpha_pos = np.zeros(Dim)
    alpha_fitness = np.inf
    beta_pos = np.zeros(Dim)
    beta_fitness = np.inf
    delta_pos = np.zeros(Dim)
    delta_fitness = np.inf

    positions = initialization(Dim,UB,LB)

    l = 0
    while l<MaxT:
        for i in range (positions.shape[0]):
            BB_UB = positions[i,:]>UB 
            BB_LB = positions[i,:]<LB
            positions[i,:] = (positions[i,:]*(~(BB_UB+BB_LB)))+UB*BB_UB+LB*BB_LB
            Fitness = CalculateFitness(positions[i,:])

            if Fitness<alpha_fitness:
                alpha_fitness=Fitness
                alpha_pos=positions[i,:]

            if Fitness>alpha_fitness and Fitness<beta_fitness:
                beta_fitness=Fitness
                beta_pos=positions[i,:]
            
            if Fitness>alpha_fitness and Fitness>beta_fitness and Fitness<delta_fitness:
                delta_fitness=Fitness
                delta_pos=positions[i,:]
        
        a = 2-1*(2/MaxT)
        for i in range (positions.shape[0]):
            for j in range (positions.shape[1]):
                r1=np.random.random()
                r2=np.random.random()

                A1 = 2*a*r1-a
                C1 = 2 * r2

                D_Alpha = abs(C1*alpha_pos[j]-positions[i,j])
                X1 = alpha_pos[j]-A1*D_Alpha
                
                r1=np.random.random()
                r2=np.random.random()

                A2 = 2*a*r1-a
                C2=2*r2

                D_Beta = abs(C2*beta_pos[j]-positions[i,j])
                X2 = beta_pos[j]-A2*D_Beta

                r1 = np.random.random()
                r2 = np.random.random()

                A3 = 2*a*r1-a
                C3 = 2*r2

                D_Delta = abs(C3 * delta_pos[j] - positions[i,j])
                X3 = delta_pos[j] - A3 * D_Delta

                positions[i,j] = (X1 + X2 + X3) / 3
        l += 1
    return alpha_fitness, alpha_pos

if __name__ == "__main__":
    LB = -100
    UB = 100
    Dim = 30
    MaxT = 100

    bestfit, bestsol = GWO(MaxT,LB,UB,Dim)
    print("Best Fitness =", bestfit)
    print("Best Solution = ",bestsol)
