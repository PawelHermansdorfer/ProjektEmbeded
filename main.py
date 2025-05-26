import numpy as np

from pprint import pprint

from base import *
from gwo import *


TASKS_TO_SPLIT_COUNT = 3
MIN_SUBTASK_TO_MAIN_TASK_RATIO = 0.15
MIN_SUBTASK_COUNT = 3
MAX_SUBTASK_COUNT = 6
# NOTE(Pawel Hermansdorfer): MAX_SUBTASK_COUNT * MIN_SUBTASK_TO_MAIN_TASK_RATIO must be <= 1

SUBTASK_CONFIGURATION_SIZE = 3
CONFIGURATIONS_PER_UT = 4

np.random.seed(0)


(task_count, tasks), (proc_count, procs), (chann_count, channs) = read_graph_file('./data/test.txt')
read_architecture_file(tasks, procs, channs, './data/architektura.txt')

pp_procs = [proc for proc in procs if proc.is_multipurpose]
pp_proc_count = len(pp_procs)

subtask_count, subtasks = create_subtasks(tasks,
                                          TASKS_TO_SPLIT_COUNT, MIN_SUBTASK_TO_MAIN_TASK_RATIO,
                                          MIN_SUBTASK_COUNT, MAX_SUBTASK_COUNT) # list with references to all subtasks
ut_task_count, ut_tasks = select_configurations_for_unpredicted_tasks(tasks, subtasks, CONFIGURATIONS_PER_UT, SUBTASK_CONFIGURATION_SIZE) # list with references to all UT tasks


# solution = [
#               subtask configuration idx for UT_0,
#               subtask configuration idx for UT_1,
#               ...,
#               subtask configuration idx for UT_N,
#               PP proc idx for 1'st subtask
#               PP proc idx for 2'nd subtask
#               ...,
#               PP proc idx for N'th subtask
#            ]
# len(solution) = ut_count + subtask_count


def apply_solution(solution):
    configuration_idxs = solution[:ut_task_count]
    subtask_procs      = solution[ut_task_count:]

    for configuration_idx, task in zip(configuration_idxs, ut_tasks):
        task.unpredicted_subtasks = task.subtask_configurations[configuration_idx]

    for proc_idx, subtask in zip(subtask_procs, subtasks):
        subtask.proc_idx = pp_procs[proc_idx].idx


def init_population(population_size):
    result = []
    for _ in range(population_size):
        solution = []

        for task in ut_tasks:
            solution.append(np.random.randint(0, task.subtask_configuration_count))

        for subtask in subtasks:
            solution.append(np.random.randint(0, pp_proc_count))

        result.append(solution)

    return result

def fitness(solution):
    apply_solution(solution)
    result = get_time(tasks, procs, channs) * get_cost(tasks, procs, channs) 
    return result

def limit_pos(x, idx):
    result = 0
    if idx < ut_task_count:
        result = max(min(round(x), SUBTASK_CONFIGURATION_SIZE), 0)
    else:
        result = max(min(round(x), pp_proc_count-1), 0)
    return result 

max_iterations = 100
population_size = 10
best_fitness, best_solution, plot_data = gwo(max_iterations, population_size, init_population, fitness, limit_pos)

for alpha_pos in plot_data['alpha_pos']:
    apply_solution(alpha_pos)
    plot_data['alpha_time'].append(get_time(tasks, procs, channs))
    plot_data['alpha_cost'].append(get_cost(tasks, procs, channs))

apply_solution(best_solution)
print(f'Best solution: {best_solution} Best fitness: {best_fitness}')
print(f'Time:  {get_time(tasks, procs, channs)}')
print(f'Cost:  {get_cost(tasks, procs, channs)}')
gwo_plot_result(plot_data)
