from collections import defaultdict
import numpy as np

from base import *
from gwo import *


TASKS_TO_SPLIT_COUNT = 6
MIN_SUBTASK_TO_MAIN_TASK_RATIO = 0.15
MIN_SUBTASK_COUNT = 3
MAX_SUBTASK_COUNT = 6
# NOTE(Pawel Hermansdorfer): MAX_SUBTASK_COUNT * MIN_SUBTASK_TO_MAIN_TASK_RATIO must be <= 1

SUBTASK_CONFIGURATION_SIZE = 3
CONFIGURATIONS_PER_UT = 4

# np.random.seed(100)
np.random.seed()


(task_count, tasks), (proc_count, procs), (chann_count, channs) = read_graph_file('./data/graf_2.txt')
# draw_graph(tasks)
read_architecture_file(tasks, procs, channs, './data/architektura_2.txt')

pp_procs = [proc for proc in procs if proc.is_multipurpose]
pp_proc_count = len(pp_procs)

subtask_count, subtasks = create_subtasks(tasks,
                                          TASKS_TO_SPLIT_COUNT, MIN_SUBTASK_TO_MAIN_TASK_RATIO,
                                          MIN_SUBTASK_COUNT, MAX_SUBTASK_COUNT)
ut_task_count, ut_tasks = select_configurations_for_unpredicted_tasks(tasks, subtasks, CONFIGURATIONS_PER_UT, SUBTASK_CONFIGURATION_SIZE)


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
stop_after_stagnation = 15
best_fitness, best_solution, plot_data, iterations = gwo(max_iterations, population_size, init_population, fitness, limit_pos, stop_after_stagnation)
apply_solution(best_solution)

print(f'Best solution: {best_solution} Best fitness: {round(best_fitness, 1)} Itrations: {iterations}')
print(f'Time:  {get_time(tasks, procs, channs): .1f}')
print(f'Cost:  {get_cost(tasks, procs, channs): .1f}')

print("\n=== ARCHITEKTURA KOŃCOWA ===")
proc_to_tasks = defaultdict(list)
for task in tasks:
    if task.is_unpredicted:
        continue

    if task.subtasks:
        for sub in task.subtasks:
            label = f"T{task.idx}_{sub.idx}"
            proc = procs[sub.proc_idx]
            proc_label = f"PP{proc.idx}" if proc.is_multipurpose else f"HC{proc.idx}"
            proc_to_tasks[proc_label].append(label)
    else:
        label = f"T{task.idx}"
        proc = procs[task.proc_idx]
        proc_label = f"PP{proc.idx}" if proc.is_multipurpose else f"HC{proc.idx}"
        proc_to_tasks[proc_label].append(label)

print("\n== Przypisanie zadań ==")
for proc_label, tasks_list in proc_to_tasks.items():
    print(f"{proc_label}: {', '.join(tasks_list)}")


unpredicted_to_subtask = defaultdict(list)

for task in tasks:
    if task.is_unpredicted:
        label = f"UT{task.idx}"
        for sub in task.unpredicted_subtasks:
            unpredicted_to_subtask[label].append(f"T{sub.main_task_idx}_{sub.minor_idx}")

print("\n== Rozwiązanie zadań nieprzewidzianych ==")
for unpredicted_list, sub_list in unpredicted_to_subtask.items():
    print(f"{unpredicted_list}: {', '.join(sub_list)}")

tast_to_substract = defaultdict(list)

for task in tasks:
    label = f"T{task.idx}"
    for sub in task.subtasks:
        tast_to_substract[label].append(f"T{sub.main_task_idx}_{sub.minor_idx} ratio: {sub.ratio}")
    if not task.subtasks:
        tast_to_substract[label].append(label)

print("\n== Podział zadań ==")
for task_list, sub_list in tast_to_substract.items():
    print(f"{task_list}: {', '.join(sub_list)}")


print("\n== Przydział kanałów ==")
used_proc_idxs = {int(label[2:]) for label in proc_to_tasks.keys()}
for chann in channs:
    connected_procs = []
    for proc in procs:
        if proc.idx in used_proc_idxs and proc.chann_connected[chann.idx]:
            label = f"PP{proc.idx}" if proc.is_multipurpose else f"HC{proc.idx}"
            connected_procs.append(label)
    if connected_procs:
        print(f"CHANN_{chann.idx}: {', '.join(connected_procs)}")


# Calculate times
get_time(tasks, procs, channs)
for task in tasks:
    if task.is_unpredicted:
        task.time = 0
        for subtask in task.unpredicted_subtasks:
            task.time += tasks[subtask.main_task_idx].times[subtask.proc_idx] * subtask.ratio

    elif task.subtasks:
        task.time = 0
        for subtask in task.subtasks:
            task.time += task.times[subtask.proc_idx] * subtask.ratio

    else:
        task.time = task.times[task.proc_idx]

    task.begin_time = task.finish_time - task.time

for task in tasks:
    print(f'Task{task.idx} begin time {task.begin_time} | time: {task.time} | finish time: {task.finish_time}')


for alpha_pos in plot_data['alpha_pos']:
    apply_solution(alpha_pos)
    plot_data['alpha_time'].append(get_time(tasks, procs, channs))
    plot_data['alpha_cost'].append(get_cost(tasks, procs, channs))
apply_solution(best_solution)
gwo_plot_result(plot_data)
