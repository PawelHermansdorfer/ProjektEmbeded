from dataclasses import dataclass
from typing import List
from dataclasses import dataclass, field

import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
import re

@dataclass
class Subtask():
    idx:            int = None # Index of subtask
    minor_idx:      int = None
    main_task_idx:  int = None # Index of main task
    proc_idx:       int = None # Index of processor
    ratio:          float = None # Percent (0.0;1.0)


@dataclass
class Task():
    idx:      int = None # Index of task
    proc_idx: int = None # Index of processor assinged to this task

    is_unpredicted: bool = None

    children_count: int       = None
    parents_idxs:   List[int] = field(default_factory=list) 
    children_idxs:  List[int] = field(default_factory=list) 
    children_costs: List[int] = field(default_factory=list) # Cost for each children from children_idxs list

    costs: List[int] = field(default_factory=list) # Costs of this task per each processor
    times: List[int] = field(default_factory=list) # Times of this task per each processor

    subtasks:               List[Subtask] = field(default_factory=list) # List of subtasks
    unpredicted_subtasks:   List[Subtask] = field(default_factory=list) # List for 3 subtasks for unpredicted tasks
    subtask_configurations: List[List[Subtask]] = field(default_factory=list)
    subtask_configuration_count: int = None

    time: float = None
    begin_time: float = None 
    finish_time: float = None


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
def create_subtasks(tasks, tasks_to_split_count, min_ratio, min_subtask_count, max_subtask_count):
    options = [task for task in tasks if not task.is_unpredicted]
    tasks_to_split_count = min(tasks_to_split_count, len(options))

    subtask_count = 0
    subtasks = []

    while tasks_to_split_count > 0:
        task = np.random.choice(options)
        task.proc_idx = None
        options.remove(task)

        num_subtasks = np.random.randint(min_subtask_count, max_subtask_count)

        remaining = 100 - num_subtasks * min_ratio*100
        cut_points = np.sort(np.random.randint(0, remaining + 1, num_subtasks - 1))
        cut_points = np.concatenate(([0], cut_points, [remaining]))
        increments = np.diff(cut_points)
        random_integers = increments + min_ratio*100
        ratios = random_integers / 100

        task.subtasks = []
        for idx, ratio in enumerate(ratios):
            subtask = Subtask(
                        idx=subtask_count,
                        minor_idx=idx,
                        main_task_idx=task.idx,
                        proc_idx=None,
                        ratio=ratio
                    )
            task.subtasks.append(subtask)
            subtasks.append(subtask)
            subtask_count += 1
        tasks_to_split_count -= 1
    return subtask_count, subtasks

def select_configurations_for_unpredicted_tasks(tasks, subtasks, configuration_count, configuration_size):
    ut_tasks = [task for task in tasks if task.is_unpredicted] 
    for task in ut_tasks:
        task.subtask_configurations = []
        task.subtask_configuration_count = configuration_count
        for _ in range(configuration_count):
            configuration = [np.random.choice(subtasks) for __ in range(configuration_size)]
            task.subtask_configurations.append(configuration)

    return len(ut_tasks), ut_tasks

def get_subtask_cost(tasks, subtask: Subtask) -> int:
    base_cost = tasks[subtask.main_task_idx].costs[subtask.proc_idx]
    return base_cost * subtask.ratio

def get_subtask_time(tasks, subtask: Subtask) -> int:
    base_time = tasks[subtask.main_task_idx].times[subtask.proc_idx]
    return base_time * subtask.ratio

########################################
def get_cost(tasks, procs, channs):
    result = 0
    tasks_per_proc = [0 for _ in procs]

    # Cost of tasks
    for task in tasks:
        if not task.is_unpredicted:
            if task.subtasks:
                for sub in task.subtasks:
                    tasks_per_proc[sub.proc_idx] += 1
                    result += task.costs[sub.proc_idx] * sub.ratio
            else:
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
                    result += chann.cost * tasks_per_proc[proc.idx]
    return result


def get_time(tasks, procs, channs):
    result = 0

    proc_free_time   = [0 for _ in procs]
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

        if task.subtasks:
            current_time = latest_parent_finish_time
            for sub in task.subtasks:
                start_time = max(proc_free_time[sub.proc_idx], current_time)
                finish_time = start_time + get_subtask_time(tasks, sub)
                proc_free_time[sub.proc_idx] = finish_time
                current_time = finish_time
            finish_times[task.idx] = current_time
        else:
            finish_times[task.idx] = max(proc_free_time[task.proc_idx], latest_parent_finish_time)
            finish_times[task.idx] += task.times[task.proc_idx]
        if task.idx != latest_parent_finish_idx:
            if task.subtasks:
                proc_a = procs[task.subtasks[0].proc_idx]
            else:
                proc_a = procs[task.proc_idx]
        
            if tasks[latest_parent_finish_idx].subtasks:
                proc_b = procs[tasks[latest_parent_finish_idx].subtasks[-1].proc_idx]
            else:
                proc_b = procs[tasks[latest_parent_finish_idx].proc_idx]
       
            if proc_a != proc_b:
                shared_channs = [i for i in range(len(channs)) if proc_a.chann_connected[i] and proc_b.chann_connected[i]]
                best_chann_idx = min(shared_channs, key=lambda idx: channs[idx].throughput)
                best_chann = channs[best_chann_idx]
                child_cost_idx = tasks[latest_parent_finish_idx].children_idxs.index(task.idx)
                B = tasks[latest_parent_finish_idx].children_costs[child_cost_idx] / best_chann.throughput
                finish_times[task.idx] += B

        if task.subtasks:
            last_proc_idx = task.subtasks[-1].proc_idx
            proc_free_time[last_proc_idx] = finish_times[task.idx]
        else:
            proc_free_time[task.proc_idx] = finish_times[task.idx]
        result = max(result, finish_times[task.idx])

    # + time of unpredicted tasks
    for task in tasks:
        if task.is_unpredicted:
            u_start_time = result
            for sub in task.unpredicted_subtasks:
                u_start_time = max(u_start_time, proc_free_time[sub.proc_idx])
                u_finish_time = u_start_time + get_subtask_time(tasks, sub)
                proc_free_time[sub.proc_idx] = u_finish_time
                u_start_time = u_finish_time
            finish_times[task.idx] = u_finish_time

    for task in tasks:
        task.finish_time = finish_times[task.idx]
    return max(finish_times)

########################################
def read_graph_file(file_path):
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

    # Read file
    file = open(file_path)
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
                for proc in procs:
                    proc.chann_connected[0] = True

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
                idx_part = (offset, child.find('('))
                child_idx = int(child[idx_part[0]: idx_part[1]])
                task.children_idxs.append(child_idx)

                child_cost = int(child[child.find('(') + 1: -1])
                task.children_costs.append(child_cost)
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
    
    return (task_count, tasks), (proc_count, procs), (chann_count, channs)


def read_architecture_file(tasks, procs, channs, file_path):
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if line.startswith("PP") or line.startswith("HC"):
                proc_name, task_list_str = line.split(":")
                proc_idx = int(re.findall(r'\d+', proc_name)[0])
                task_names = [t.strip() for t in task_list_str.split(",") if t.strip()]
                for task_name in task_names:
                    task_idx = int(re.findall(r'\d+', task_name)[0])
                    tasks[task_idx].proc_idx = proc_idx
            elif line.startswith("CHANN_"):
                chann_str, proc_names_str = line.split(":")
                chann_idx = int(re.findall(r'\d+', chann_str)[0])
                proc_names = [name.strip() for name in proc_names_str.split(",") if name.strip()]
                for proc_name in proc_names:
                    proc_idx = int(re.findall(r'\d+', proc_name)[0])
                    procs[proc_idx].chann_connected[chann_idx] = True



def draw_graph(tasks):
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

    G = nx.DiGraph()
    G.add_edges_from(edges)
    pos = hierarchy_pos(G, root=0)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray',
            node_size=2000, font_size=16, arrows=True, arrowsize=20)
    plt.show()
