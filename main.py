
from dataclasses import dataclass
from typing import List
from dataclasses import dataclass, field

import random
random.seed()

from pprint import pprint # pip install pprint

########################################
# CONSTANTS
MIN_SUBTASK = 3
MAX_SUBTASK = 6


########################################
@dataclass
class Subtask():
    idx:            int = None # Index of subtask
    main_task_idx:  int = None # Index of main task
    proc_idx:       int = None # Index of processor
    ratio:          float = None # Percent (0.0;1.0)

@dataclass
class Task():
    idx:      int = None # Index of task
    proc_idx: int = None # Index of processor assinged to this task

    # TODO(Pawel Hermansdorfer): Read from file
    is_unpredicted: bool = None

    children_count: int       = None
    parents_idxs:   List[int] = field(default_factory=list) 
    children_idxs:  List[int] = field(default_factory=list) 
    children_costs: List[int] = field(default_factory=list) # Cost for each children from children_idxs list

    costs: List[int] = field(default_factory=list) # Costs of this task per each processor
    times: List[int] = field(default_factory=list) # Times of this task per each processor

    subtasks: List[Subtask] = field(default_factory=list) # List of subtasks
    unpredicted_subtasks: List[Subtask] = field(default_factory=list) # List for 3 subtasks for unpredicted tasks


@dataclass
class Proc():
    idx:             int        = None
    cost:            int        = None
    is_multipurpose: bool       = None
    chann_connected: List[bool] = field(default_factory=list) # Bool for each chann

@dataclass
class Chann():
    name:         str        = None
    idx:          int        = None
    cost:         int        = None
    throughput:   int        = None
    allows_procs: List[bool] = field(default_factory=list) # Bool for each processor

########################################
GRAPH_FILE = './data/test.txt'

task_count = 0
tasks = []

proc_count = 0
procs = []

chann_count = 0
channs = []

# Helpers for parsing graph file
CHUNK_NONE  = 0
CHUNK_TASKS = 1
CHUNK_PROC  = 2
CHUNK_TIMES = 3
CHUNK_COSTS = 4
CHUNK_COMM  = 5
parsing_chunk = CHUNK_NONE

times_idx = 0
costs_idx = 0
proc_idx  = 0
chann_idx = 0

########################################
# Read file
file = open(GRAPH_FILE)
line = file.readline().strip().replace('\t', ' ')
while line:
    # DETERMINE NEXT CHUNK
    if line[0] == '@':
        chunk_title = line[1:]
        parsing_chunk += 1
        if chunk_title.startswith('tasks'):
            task_count = int(line.split(' ')[1])
            tasks = [Task() for _ in range(task_count)]
        elif chunk_title.startswith('proc'):
            proc_count = int(line.split(' ')[1])
            procs = [Proc() for _ in range(proc_count)]
        elif chunk_title.startswith('times'):
            pass
        elif chunk_title.startswith('cost'):
            pass
        elif chunk_title.startswith('comm'):
            chann_count = int(line.split(' ')[1])
            channs = [Chann() for _ in range(chann_count)]
            for proc in procs:
                proc.chann_connected = [False for _ in range(chann_count)]

    # TASKS
    elif parsing_chunk == CHUNK_TASKS:
        split = line.split(' ')
        task_label = split[0]
        is_unpredicted = task_label.startswith("UT")

        task_idx = int(''.join(filter(str.isdigit,task_label)))
        task = tasks[task_idx]

        task.idx = task_idx
        task.is_unpredicted = is_unpredicted
        task.children_count = int(split[1])

        for child in split[2:]:
            offset = 0 if not child[0].isalpha() else 1
            child_idx = int(child[0 + offset])

            task.children_idxs.append(child_idx)
            task.children_costs.append(int(child[2 + offset: -1]))
            tasks[child_idx].parents_idxs.append(task_idx)

    # PROCESSORS
    elif parsing_chunk == CHUNK_PROC:
        cost, _, is_multipurpose = line.strip().split(' ')
        procs[proc_idx].idx             = proc_idx
        procs[proc_idx].cost            = int(cost)
        procs[proc_idx].is_multipurpose = int(is_multipurpose)==1
        proc_idx += 1

    # TIMES
    elif parsing_chunk == CHUNK_TIMES:
        times = line.split(' ')
        tasks[times_idx].times = list(map(int, times))
        times_idx += 1

    # COSTS
    elif parsing_chunk == CHUNK_COSTS:
        costs = line.split(' ')
        tasks[costs_idx].costs = list(map(int, costs))
        costs_idx += 1

    # CHANNELS 
    elif parsing_chunk == CHUNK_COMM:
        split = line.split(' ')
        name, cost, throughput = split[0], int(split[1]), int(split[2])
        channs[chann_idx].idx          = chann_idx
        channs[chann_idx].name         = name
        channs[chann_idx].cost         = cost
        channs[chann_idx].throughput   = throughput
        channs[chann_idx].allows_procs = [None for _ in range(proc_count)]
        for allows_idx, allows in enumerate(split[3:]):
            channs[chann_idx].allows_procs[allows_idx] = int(allows)==1
        chann_idx += 1

    line = file.readline().strip().replace('\t', ' ')

########################################
# Read architecture
file = open('./data/architektura.txt')
for line in file:
    parts = line.strip().split()
    task_idx = int(parts[0])
    proc_idx = int(parts[1])
    chan_idx = int(parts[2])

    tasks[task_idx].proc_idx = proc_idx
    procs[proc_idx].chann_connected[chan_idx] = True
file.close()

########################################
########## TEMP ########
# These fields need to be non None to calculate cost and time
#for task in tasks:
#    task.proc_idx = 0
#
#for proc in procs:
#    proc.chann_connected[0] = True

########################################
# Create subtasks
def create_subtasks():
    for task in tasks:
        if task.is_unpredicted or task.subtasks: # nieprzewidzane albo już ma podzadania
            continue
        num_subtasks = random.randint(MIN_SUBTASK,MAX_SUBTASK)
        random_values = [random.random() for _ in range(num_subtasks)] # losuje proporcje
        total = sum(random_values)
        ratios = [v / total for v in random_values] # normalizuje - suma bedzie rowna 1
        if sum(ratios) != 1: #TODO: test
            print("ERROR - create_subtasks() - powinno byc 1")
            print(sum(ratios))


        for i, ratio in enumerate(ratios):
            task.subtasks.append(Subtask(
                idx=i,
                main_task_idx=task.idx,
                proc_idx=task.proc_idx,
                ratio=ratio
            ))

########################################

def choose_subtasks():
    subtasks_available=[]
    
    for i in tasks:
        if i.is_unpredicted:
            continue
        subtasks_available.append(i.subtasks)

    for i in range(len(tasks)):
        if tasks[i].is_unpredicted:
            unexpected_solution=[]
            deleted=[]
            #TODO: #wiecej niz jedno rozw?
            for j in range(3):
                task_id=random.randint(0, len(subtasks_available)-1)
                subtask_id=random.randint(0, len(subtasks_available[task_id])-1)
                choose_sub = subtasks_available[task_id][subtask_id]
                unexpected_solution.append(choose_sub)
                deleted.append(choose_sub)
                del subtasks_available[task_id][subtask_id]

            tasks[i].unpredicted_subtasks=unexpected_solution #unpredicted
            for sub in deleted:
                subtasks_available[sub.main_task_idx].append(sub)

# Calculate cost
def get_cost():
    result = 0
    tasks_per_proc = [0 for _ in range(proc_count)]

    # Cost of tasks
    for task in tasks:
        if not task.is_unpredicted:
            tasks_per_proc[task.proc_idx] += 1
            result += task.costs[task.proc_idx]

    # Cost of procs
    for proc in procs:
        if tasks_per_proc[proc.idx] > 0 and proc.is_multipurpose:
            result += proc.cost

    # Cost of channs
    for proc in procs:
        for chann in channs:
            if proc.chann_connected[chann.idx] == 1 and tasks_per_proc[proc.idx] > 0:
                if proc.is_multipurpose == 1: 
                    result += chann.cost
                else:
                    result += chann.cost * tasks_per_proc[proc_idx]
    return result

########################################

def get_subtask_cost(subtask: Subtask) -> int:
    base_cost = tasks[subtask.main_task_idx].costs[substask.proc_idx]
    return round(base_cost * subtask.ratio)

def get_subtask_time(subtask: Subtask) -> int:
    base_time = tasks[subtask.main_task_idx].times[subtask.proc_idx]
    return round(base_time * subtask.ratio)

########################################
# Calculate time
def get_time():
    result = 0

    proc_free_time   = [0 for _ in range(proc_count)]
    tasks_to_be_done = [task for task in tasks if not task.is_unpredicted]
    finish_times     = [None for _ in tasks]

    while tasks_to_be_done:
        task = tasks_to_be_done[0]
        tasks_to_be_done.remove(task)
        parents_finished = True
        latest_parent_finish_idx = 0
        latest_parent_finish_time = 0
        for parent_idx in task.parents_idxs:
            if finish_times[parent_idx] == None: 
                tasks_to_be_done.append(task)
                continue
            else:
                if finish_times[parent_idx] > latest_parent_finish_time:
                    latest_parent_finish_idx = parent_idx
                    latest_parent_finish_time = finish_times[parent_idx]

        finish_times[task.idx] = max(proc_free_time[task.proc_idx], latest_parent_finish_time)
        finish_times[task.idx] += task.times[task.proc_idx]
        if task.proc_idx != tasks[latest_parent_finish_idx].proc_idx:
            # TODO(Pawel Hermansdorfer): Replace chann0 with correct one. What if they are connected by two channels????
            B = tasks[latest_parent_finish_idx].children_costs[task_idx] / channs[0].throughput
            finish_time[task.idx] += B

        proc_free_time[task.proc_idx] = finish_times[task.idx]
        result = max(result, finish_times[task.idx])
    # + time of unpredicted tasks
    for task in tasks:
        if task.is_unpredicted:
            u_start_time = result
            for sub in task.unpredicted_subtasks:
                u_start_time = max(u_start_time, proc_free_time[sub.proc_idx])
                u_finish_time = u_start_time + get_subtask_time(sub)
                proc_free_time[sub.proc_idx] = u_finish_time
                u_start_time = u_finish_time
            finish_times[task.idx] = u_finish_time
    return max([t for t in finish_times if t is not None])

create_subtasks()
choose_subtasks()

# tasks[6].unpredicted_subtasks = [
#     Subtask(0, 3, 0, 0.04),#10
#     Subtask(1, 2, 1, 0.5),#180
#     Subtask(2, 5, 1, 0.2)#11
# ]
#
# tasks[7].unpredicted_subtasks = [
#     Subtask(0, 4, 0, 0.5),#75
#     Subtask(1, 3, 1, 0.5),#110
#     Subtask(2, 1, 2, 0.5)#12
# ]

print(f'Time: {get_time()}')
print(f'Cost: {get_cost()}')

pprint(procs)
pprint(channs)
pprint(tasks)


########################################
# Draw graph
# Helper, may be helpful
# pip install networkx matplotlib
'''
import networkx as nx
import matplotlib.pyplot as plt

def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    if pos is None:
        pos = {}
    if root is None:
        root = list(nx.topological_sort(G))[0]
    children = list(G.successors(root))
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Only works with DiGraph")
    if parent:
        children = [c for c in children if c != parent]
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
    pos[root] = (xcenter, vert_loc)
    return pos

# Draw original graph
G = nx.DiGraph()
edges = []
for task in tasks:
    for child_idx in task.children_idxs:
        edges.append((task.idx, child_idx))
G.add_edges_from(edges)
pos = hierarchy_pos(G, root=0)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray',
        node_size=2000, font_size=16, arrows=True, arrowsize=20)
plt.show()
'''
