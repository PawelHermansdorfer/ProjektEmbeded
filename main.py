from dataclasses import dataclass
from typing import List
from dataclasses import dataclass, field

import random

from pprint import pprint


@dataclass
class Task():
    idx: int = None

    children_count: int = None
    children_idxs: List[int] = field(default_factory=list) 
    children_costs: List[int] = field(default_factory=list) 
    parents_idxs:  List[int] = field(default_factory=list) 

    costs: List[int] = field(default_factory=list) 
    times: List[int] = field(default_factory=list) 

    proc_idx: int = None

@dataclass
class Proc():
    pass

@dataclass
class Chann():
    pass

CHUNK_NONE = 0
CHUNK_TASKS = 1
CHUNK_PROC = 2
CHUNK_TIMES = 3
CHUNK_COSTS = 4
CHUNK_COMM = 5
parsing_chunk = CHUNK_NONE

GRAPH_FILE = './graf_6.txt'

task_count = 0
proc_count = 0
chann_count = 0

tasks = []
procs = []
channs = []

times_idx = 0
costs_idx = 0

file = open(GRAPH_FILE)
line = file.readline().strip().replace('\t', ' ')
while line:
    if line[0] == '@':
        chunk_title = line[1:]
        parsing_chunk += 1
        if chunk_title.startswith('tasks'):
            task_count = int(line.split(' ')[1])
            tasks = [Task() for _ in range(task_count)]
        if chunk_title.startswith('proc'):
            proc_count = int(line.split(' ')[1])
        if chunk_title.startswith('times'):
            pass
        if chunk_title.startswith('cost'):
            pass
        if chunk_title.startswith('comm'):
            chann_count = int(line.split(' ')[1])

    elif parsing_chunk == CHUNK_TASKS:
        split = line.split(' ')
        task_idx = int(split[0][1])
        task = tasks[task_idx]

        task.idx = task_idx
        task.children_count = int(split[1])

        for child in split[2:]:
            offset = 0 if not child[0].isalpha() else 1
            child_idx = int(child[0 + offset])

            task.children_idxs.append(child_idx)
            task.children_costs.append(int(child[2 + offset: -1]))
            tasks[child_idx].parents_idxs.append(task_idx)

    # elif parsing_chunk == CHUNK_PROC:
    #     cost, _, is_multipurpose = line.strip().split(' ')
    #     procs.append({'cost': int(cost), 'is_multipurpose': int(is_multipurpose)})

    elif parsing_chunk == CHUNK_TIMES:
        times = line.split(' ')
        tasks[times_idx].times = list(map(int, times))
        times_idx += 1

    elif parsing_chunk == CHUNK_COSTS:
        costs = line.split(' ')
        tasks[costs_idx].costs = list(map(int, costs))
        costs_idx += 1

    # elif parsing_chunk == CHUNK_COMM:
    #     split = line.split(' ')
    #     name, cost, throughput = split[0], int(split[1]), int(split[2])
    #     allows_procs = []
    #     for b in split[3:-1]:
    #         allows_procs.append(int(b))
    #     chann = {
    #             'cost': cost,
    #             'throughput': throughput,
    #             'allows_procs': allows_procs
    #     }
    #     channs.append(chann)

    line = file.readline().strip().replace('\t', ' ')

pprint(tasks)








# # Calculate cost
# def get_cost():
#     cost = 0
#     tasks_per_proc = [0 for _ in range(proc_count)]
#     for task in tasks:
#         proc_idx = task['proc_idx']
#         tasks_per_proc[proc_idx] += 1
#         cost += task['costs'][proc_idx]

#     for proc_idx in range(proc_count):
#         if tasks_per_proc[proc_idx] > 0 and procs[proc_idx]['is_multipurpose']:
#             cost += procs[proc_idx]['cost']

#     for proc_idx, proc in enumerate(procs):
#         for chann_idx in range(chann_count):
#             if proc['chann_used'][chann_idx] == 1 and tasks_per_proc[proc_idx]>0:
#                 if proc['is_multipurpose'] == 1: 
#                     cost += channs[chann_idx]['cost']
#                 else:
#                     cost += channs[chann_idx]['cost'] * tasks_per_proc[proc_idx]
#     return cost

# # # Calculate time
# def get_time():
#     time = 0

#     proc_free_time = [0 for _ in range(proc_count)]
#     tasks_to_be_done= [(task, task_idx) for task_idx, task in enumerate(tasks)]
#     for task in tasks: task['finish_time'] = None

#     while tasks_to_be_done:
#         task_task_idx = tasks_to_be_done[0]
#         tasks_to_be_done.remove(task_task_idx)
#         task, task_idx = task_task_idx

#         parents_finished = True
#         latest_parent_finish_idx = 0
#         latest_parent_finish_time = 0
#         for parent_idx in task['parents']:
#             if tasks[parent_idx]['finish_time'] == None: 
#                 tasks_to_be_done.append(task)
#                 continue
#             else:
#                 if tasks[parent_idx]['finish_time'] > latest_parent_finish_time:
#                     latest_parent_finish_idx = parent_idx
#                     latest_parent_finish_time = tasks[parent_idx]['finish_time']

#         task['finish_time'] = max(proc_free_time[task['proc_idx']], latest_parent_finish_time)
#         task['finish_time'] += task['times'][task['proc_idx']]
#         if task['proc_idx'] != tasks[latest_parent_finish_idx]['proc_idx']:
#             B = tasks[latest_parent_finish_idx]['children_costs'][task_idx] / channs[0]['throughput']
#             task['finish_time'] += B

#         proc_free_time[task['proc_idx']] = task['finish_time']
#         time = max(time, task['finish_time'])
#     return time


# # Zaczynamy od rozwiązania najszybszego
# for task in tasks:
#     fastest_proc_idx = 0
#     fastest_proc_time = task['times'][fastest_proc_idx]
#     for proc_idx in range(proc_count):
#         if task['times'][proc_idx] < fastest_proc_time:
#             fastest_proc_time = task['times'][proc_idx]
#             fastest_proc_idx = proc_idx

#     task['proc_idx'] = fastest_proc_idx


# for proc in procs:
#     proc['chann_used'] = [0 for _ in range(chann_count)]
#     proc['chann_used'][0] = 1

# # Szukamy najdroższego zadania/szyny komunikacyjnej

# # Zamieniamy zadanie lub szynę na średnie rozwiązanie między najszybszym a najwolniejszym używając kryteriów:
# #    - Maksymalnego czasu/kosztu systemu

# # . Algorytm kontynuujemy aż do momentu, gdy minimalny zysk jest jeszcze osiągany (jeśli jakaś z
# #   decyzji nie powoduje co najmniej minimalnego zysku, to kończymy rafinowanie systemu lub gdy któraś z
# #   decyzji musiała by spowodować przekroczenie czasu maksymalnego)

# # Funkcja "greedy" ma prawdopodobieństwo wyboru 0.1 dla najtańszego zasobu, opcja minimalna
# # czas*koszt: 0.3, opcja średni czas: 0.2, opcja najrzadziej używany: 0.2, opcja najdłużej bezczynny: 0.2

# MIN_MEAN_GAIN = 50
# MAX_TIME = 1500

# iteration = 0
# should_stop = False
# system_value = get_cost()
# five_last_gains = []
# while not should_stop:
#     most_expensive_task_idx = 0
#     task = tasks[most_expensive_task_idx]
#     most_expensive_task_cost = task['costs'][task['proc_idx']]
#     for task_idx, task in enumerate(tasks):
#         cost = task['costs'][task['proc_idx']]
#         if cost > most_expensive_task_cost:
#             most_expensive_task_cost = cost
#             most_expensive_task_idx = task_idx
#     most_expensive_task = tasks[most_expensive_task_idx]

#     CHAEPEST = 0
#     MIN_TIME_TIMES_COST = 1
#     MEAN_TIME = 2
#     LEAST_USED = 3
#     LONGEST_IDLE = 4
#     options = [
#                 CHAEPEST, # 0.1
#                 MIN_TIME_TIMES_COST, MIN_TIME_TIMES_COST, MIN_TIME_TIMES_COST, # 0.3
#                 MEAN_TIME, MEAN_TIME, # 0.2
#                 LEAST_USED, LEAST_USED, # 0.2
#                 LONGEST_IDLE, LONGEST_IDLE # 0.2
#             ]
#     choice = random.choice(options)
    
#     print(f'Iteration {iteration} t={get_time():.2f}, c={get_cost():.2f} selected option: ', end='')

#     # CHEAPEST
#     if choice == CHAEPEST:
#         print(f'"cheapest resource" for task {most_expensive_task_idx}', end ='')
#         cheapest_proc_idx = 0
#         cheapest_proc_cost = most_expensive_task['costs'][cheapest_proc_idx]
#         for proc_idx in range(proc_count):
#             cost = most_expensive_task['costs'][proc_idx]
#             if cost < cheapest_proc_cost:
#                 cheapest_proc_cost = cost
#                 cheapest_proc_idx = proc_idx

#         most_expensive_task['proc_idx'] = cheapest_proc_idx

#     # MEAN TIME
#     elif choice == MIN_TIME_TIMES_COST:
#         print(f'"min time*cost" for task {most_expensive_task_idx}', end ='')
#         best_proc_idx = 0
#         best_proc_value = most_expensive_task['costs'][best_proc_idx] * most_expensive_task['times'][best_proc_idx]
#         for proc_idx in range(proc_count):
#             value = most_expensive_task['costs'][proc_idx] * most_expensive_task['times'][proc_idx]
#             if value < best_proc_value:
#                 best_proc_value = cost
#                 best_proc_idx = proc_idx

#         most_expensive_task['proc_idx'] = best_proc_idx

#     elif choice == MEAN_TIME:
#         print(f'"mean time" for task {most_expensive_task_idx}', end ='')

#         mean_time = sum(most_expensive_task['times']) / len(most_expensive_task['times'])
#         best_proc_idx = 0
#         best_proc_value = abs(mean_time - most_expensive_task['times'][best_proc_idx])
#         for proc_idx in range(proc_count):
#             value = abs(most_expensive_task['times'][proc_idx] - mean_time)
#             if value < best_proc_value:
#                 best_proc_value = cost
#                 best_proc_idx = proc_idx

#     elif choice == LEAST_USED:
#         print(f'"least used proc" for task {most_expensive_task_idx}', end ='')
#         tasks_per_proc = [0 for _ in range(proc_count)]
#         for task in tasks: tasks_per_proc[task['proc_idx']] += 1
#         least_used_idx = 0
#         least_used_idx_uses = tasks_per_proc[0]
#         for proc_idx in range(proc_count):
#             if tasks_per_proc[proc_idx] < least_used_idx_uses:
#                 least_used_idx = proc_idx
#                 least_used_idx_uses = tasks_per_proc[proc_idx]

#         most_expensive_task['proc_idx'] = least_used_idx

#     elif choice == LONGEST_IDLE:
#         print(f'"longest idle proc" for task {most_expensive_task_idx}', end ='')

#         time = 0
#         proc_free_time = [0 for _ in range(proc_count)]
#         tasks_to_be_done= [(task, task_idx) for task_idx, task in enumerate(tasks)]
#         for task in tasks: task['finish_time'] = None

#         best_proc_idx = None
#         while tasks_to_be_done:
#             task_task_idx = tasks_to_be_done[0]
#             tasks_to_be_done.remove(task_task_idx)
#             task, task_idx = task_task_idx

#             if task_idx == most_expensive_task_idx:
#                 best_proc_idx = 0
#                 best_proc_free_time = proc_free_time[0]
#                 for proc_idx, free_time in enumerate(proc_free_time):
#                     if free_time < best_proc_free_time:
#                         best_proc_free_time = free_time
#                         best_pro_idx = proc_idx
#                 break


#             parents_finished = True
#             latest_parent_finish_idx = 0
#             latest_parent_finish_time = 0
#             for parent_idx in task['parents']:
#                 if tasks[parent_idx]['finish_time'] == None: 
#                     tasks_to_be_done.append(task)
#                     continue
#                 else:
#                     if tasks[parent_idx]['finish_time'] > latest_parent_finish_time:
#                         latest_parent_finish_idx = parent_idx
#                         latest_parent_finish_time = tasks[parent_idx]['finish_time']

#             task['finish_time'] = max(proc_free_time[task['proc_idx']], latest_parent_finish_time)
#             task['finish_time'] += task['times'][task['proc_idx']]
#             if task['proc_idx'] != tasks[latest_parent_finish_idx]['proc_idx']:
#                 B = tasks[latest_parent_finish_idx]['children_costs'][task_idx] / channs[0]['throughput']
#                 task['finish_time'] += B

#             proc_free_time[task['proc_idx']] = task['finish_time']
#             time = max(time, task['finish_time'])

#         most_expensive_task['proc_idx'] = best_proc_idx

#     system_time = get_time()
#     new_system_value = get_cost()
#     gain = system_value - new_system_value
#     five_last_gains.append(gain)
#     mean_gain = sum(five_last_gains) / len(five_last_gains)
#     print(f'| COST GAIN={gain:.2f} | LAST 5 COST GAINS MEAN={mean_gain:.2f}')
#     system_value = new_system_value
#     iteration += 1

#     if system_time > MAX_TIME:
#         print('MAX TIME REACHED')
#         should_stop = True

#     if len(five_last_gains) == 5:
#         mean_gain = sum(five_last_gains) / len(five_last_gains)
#         five_last_gains.pop(0)
#         if mean_gain < MIN_MEAN_GAIN:
#             print('MIN GAIN REACHED')
#             should_stop = True


# # print(f'Time: {time}')
# # print(f'Cost: {cost}')
