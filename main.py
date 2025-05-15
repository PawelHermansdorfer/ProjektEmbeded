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


# # print(f'Time: {time}')
# # print(f'Cost: {cost}')
