import json
import tracemalloc

import networkx as nx
import sys
from collections import defaultdict
import time


def get_R_u(file_path, user):       # Read the JSONL file and populate R_u
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            user_id = str(data['user_id'])  # Ensure user_id is a string
            if user_id == user:
                recommended_items_ids = list(map(str, data['recommended_items_ids']))  # Ensure item IDs are strings
                R_u[user_id] = recommended_items_ids
                return dict(R_u)

def compute_user_new_weights_new(filepath, initial_graph, R_u, lambda_param, user):
    new_weights = {}
    with open(filepath, 'r') as file:
        for line in file:
            data = json.loads(line)
            user_id = str(data['user_id'])  # Ensure user_id is a string
            if user_id == user:
                top_k_paths = data['top_k_paths']
                pairs = []
                for path in top_k_paths:
                    for i in range(len(path) - 1):
                        node1 = path[i][-1]
                        node2 = path[i + 1][-1]
                        pairs.append((node1, node2))
        for (u, v) in pairs:
            indicator_sum = 0
            if initial_graph.has_edge(str(u), str(v)):
                w0 = initial_graph[str(u)][str(v)]['weight']
                indicator_sum += 1
            elif initial_graph.has_edge(str(v), str(u)):
                w0 = initial_graph[str(v)][str(u)]['weight']
                indicator_sum += 1
            else:
                print(f"Edge ({u}, {v}) or ({v}, {u}) does not exist in the graph")
            Ru_size = len(R_u[user])
            if Ru_size > 0:
                if w0 != 0:
                    new_weight = w0 * (1 + lambda_param * indicator_sum / Ru_size)
                else:
                    new_weight = lambda_param * indicator_sum / Ru_size
            else:
                new_weight = w0
            new_weights[(str(u), str(v))] = new_weight
        return new_weights


def update_weights_new(initial_graph, new_weights):
    updated_graph = initial_graph.copy()
    for edge, weight in new_weights.items():
        u, v = edge
        if updated_graph.has_edge(u, v):
            updated_graph[u][v]['weight'] = weight
        elif updated_graph.has_edge(v, u):  # Check for reverse direction
            updated_graph[v][u]['weight'] = weight
    return updated_graph

def shift_weights(graph):
    min_weight = min(data['weight'] for u, v, data in graph.edges(data=True))
    if min_weight < 0:
        shift_value = abs(min_weight) + 1
    else:
        shift_value = 0
    for u, v, data in graph.edges(data=True):
        data['weight'] += shift_value
    return graph, shift_value


def shift_weights_back(graph, shift_value):
    for u, v, data in graph.edges(data=True):
        data['weight'] -= shift_value
    return graph

def find_terminal_nodes(R_u, user):
    terminal_nodes = set(R_u[user])
    terminal_nodes.add(user)
    return list(terminal_nodes)

def steiner_tree(graph, terminal_nodes,weight='weight'):

    complete_graph = nx.Graph()
    for i in range(len(terminal_nodes)):
        for j in range(i + 1, len(terminal_nodes)):
            length, path = nx.single_source_dijkstra(graph, terminal_nodes[i], terminal_nodes[j],weight=weight)
            complete_graph.add_edge(terminal_nodes[i], terminal_nodes[j], weight=length)

    mst = nx.minimum_spanning_tree(complete_graph, weight=weight)

    steiner_tree = nx.Graph()
    for u, v, data in mst.edges(data=True):
        length, path = nx.single_source_dijkstra(graph, u, v)
        nx.add_path(steiner_tree, path, weight=length)

    return steiner_tree



def print_specific_edges(graph, specific_edges):
    specific_edge_set = {(u, v) if u <= v else (v, u) for u, v in specific_edges}
    for u, v in graph.edges(data=True):
        if (u, v) in specific_edge_set or (v, u) in specific_edge_set:
            print((u, v, data))

file_path = 'pgpr_recommendations.jsonl' #change pgpr to cafe for the other baseline
graph_filename = 'kg_static.graphml'
initial_graph = nx.read_graphml(graph_filename)

#adjust lambda value
lambda_param = 1
users_list = ['u94','u821','u1474','u5937','u5497','u1031','u1472','u5193','u1854','u5096','u3887','u3125','u2796','u1726','u2231','u58','u2895','u3261','u1501','u5747','u3475','u1764','u1884','u4732','u1451','u5312','u3705','u1737','u2777','u4979','u4647','u4312','u4725','u4425','u1010','u4543','u1285','u5367','u424','u1449','u1015','u1980','u3618','u2063','u889','u1181','u1941','u4277','u1680','u4169','u3370','u1682','u3661','u25','u5224','u5132','u198','u1472','u1209','u5026','u3712','u2144','u4600','u2824','u1909','u3010','u2898','u5150','u4126','u5319','u4374','u5924','u5261','u1380','u5896','u4721','u160','u1990','u1752','u2611','u2272','u1686','u1261','u4635','u3072','u5462','u4667','u2880','u4282','u4094','u3492','u450','u3605','u1231','u1923','u5432','u650','u3353','u1777','u525','u2482','u1535','u5830','u2036','u1239','u4674','u5785','u918','u2968','u1407','u2684','u59','u3529','u518','u5663','u4726','u5990','u2624','u3217','u2913','u601','u4611','u2188','u3900','u5744','u3967','u5268','u4771','u4472','u854','u1701','u4408','u5759','u1125','u1920','u5996','u1988','u411','u3562','u721','u1812','u5433','u1899','u5605','u4482','u2529','u5333','u4085','u2907','u531','u1051','u3292','u2106','u1605','u752','u5643','u1448','u3539','u3224','u1088','u1150','u2128','u4905','u3933','u383','u1633','u5304','u1766','u2259','u6004','u3325','u1475','u2512','u5904','u3589','u4870','u5905','u4195','u1212','u2267','u75','u4610','u2851','u1525','u4231','u5906','u982','u5672','u601','u1236','u4347','u887','u4912','u704','u1323','u1609','u1350','u5579','u4336','u2226']


output_data = []
performance_metrics=[]

for user in users_list:
    print("Write user:")
    print(user)

    R_u = defaultdict(list)
    R_u_in= defaultdict(list)

    R_u_in = get_R_u(file_path, user)
    for k in range(1, 11):
        tracemalloc.start()
        start_time = time.time()
        snapshot_before = tracemalloc.take_snapshot()
        R_u = {user: R_u_in[user][:k]}
        print("k", k)

        new_weights = compute_user_new_weights_new(file_path, initial_graph, R_u, lambda_param, user)

        updated_graph = update_weights_new(initial_graph, new_weights)
        for u, v, data in updated_graph.edges(data=True):
            data['weight'] = -data['weight']

        shifted_graph, shift_value = shift_weights(updated_graph)

        terminal_nodes = find_terminal_nodes(R_u, user)

        steiner_tree_result = steiner_tree(shifted_graph, terminal_nodes)
        sum_weight=0
        sum_w0=0
        for u, v, data in steiner_tree_result.edges(data=True):
            if updated_graph.has_edge(u, v):
                data['weight']= updated_graph[str(u)][str(v)]['weight']
                sum_weight = sum_weight + data['weight']
                edge_data = initial_graph.get_edge_data(u, v)
                w0_temp= initial_graph[u][v]['weight']
                sum_w0= sum_w0 + w0_temp
            else:
                continue

        new_user_id = user
        steiner_summary = list(steiner_tree_result.edges())
        sum_length = len(steiner_tree_result.edges())

        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        memory_diff = sum(stat.size_diff for stat in top_stats)
        print(f"Memory usage at k={k}: {memory_diff / 1024:.2f} KB")
        tracemalloc.stop()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Time usage at k={k}: {execution_time} sec")
        user_data = {
            'user_id': new_user_id,
            'k':k,
            'lambda': lambda_param,
            'terminal_nodes': terminal_nodes,
            'steiner_summary': steiner_summary,
            'sum_metrics': {
                'sum_lengths': sum_length,
                'sum_weight': sum_weight,
                'original_weight': sum_w0
            },
            'performance': {
            'execution_time': execution_time,
            'memory_usage': memory_diff/(1024**2) #MB
            }
        }
        output_data.append(user_data)
        print(output_data)

with open("performance-final-steiner-user-pgpr.jsonl", 'w') as jsonl_file: #change pgpr to cafe for the other baseline
    for user_data in output_data:
        jsonl_file.write(json.dumps(user_data) + '\n')