import json
import networkx as nx
from collections import defaultdict
import pcst_fast
import numpy as np
import time
import pandas as pd
import tracemalloc


def get_R_u(file_path, user):
    R_u = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            user_id = str(data['user_id'])
            if user_id == user:
                recommended_items_ids = list(map(str, data['recommended_items_ids']))
                R_u[user_id] = recommended_items_ids
                return dict(R_u)

def find_terminal_nodes(R_u, user, k):
    terminal_nodes = set(R_u[user][:k])
    terminal_nodes.add(user)
    return list(terminal_nodes)

def get_group_terminal_nodes(users_list, file_path, k):
    group_terminal_nodes = set()
    for user in users_list:
        R_u = get_R_u(file_path, user)
        if R_u:
            terminal_nodes = find_terminal_nodes(R_u, user, k)
            group_terminal_nodes.update(terminal_nodes)
    print(f"Combined terminal nodes for group at k={k}: {group_terminal_nodes}")
    return list(group_terminal_nodes)

def verify_graph_attributes(graph, terminal_nodes):
    for node in graph.nodes():
        graph.nodes[node]['prize'] = 1 if node in terminal_nodes else 0
    for u, v in graph.edges():
        graph.edges[u, v]['weight'] = 0

def run_pcst(graph, terminal_nodes):
    node_indices = {node: idx for idx, node in enumerate(graph.nodes())}
    reverse_node_indices = {idx: node for node, idx in node_indices.items()}
    prizes = np.array([graph.nodes[node]['prize'] for node in graph.nodes()], dtype=np.float64)
    edges = np.array([[node_indices[u], node_indices[v]] for u, v in graph.edges()], dtype=np.int64)
    edge_weights = np.array([graph.edges[u, v].get('weight', 1.0) for u, v in graph.edges()], dtype=np.float64)

    print(f"Number of nodes: {len(node_indices)}")
    print(f"Number of edges: {len(edges)}")
    print(f"Terminal nodes indices: {[node_indices[node] for node in terminal_nodes]}")
    print(f"Prizes: {prizes[:10]}")
    print(f"Edge weights: {edge_weights[:10]}")

    vertices, edges_result = pcst_fast.pcst_fast(
        edges, prizes, edge_weights, -1, 1, "strong", 0
    )

    # Create the resulting Steiner tree
    steiner_tree = nx.Graph()
    for u in vertices:
        steiner_tree.add_node(reverse_node_indices[u])
    for e in edges_result:
        u, v = edges[e]
        steiner_tree.add_edge(reverse_node_indices[u], reverse_node_indices[v],
                              weight=graph.edges[reverse_node_indices[u], reverse_node_indices[v]]['weight'])

    return steiner_tree

def filter_users_by_gender(user_list, csv_file):
    df = pd.read_csv(csv_file)
    filtered_df = df[df['user_id'].isin(user_list)]
    users_male = filtered_df[filtered_df['gender'] == 'M']['user_id'].tolist()
    users_female = filtered_df[filtered_df['gender'] == 'F']['user_id'].tolist()
    return users_male, users_female

# baseline explanations and knowledge graph
file_path = 'pgpr_recommendations.jsonl' #change pgpr to cafe for the other baseline
graph_filename = 'kg_static.graphml'
initial_graph = nx.read_graphml(graph_filename)
lambda_param = 1

user_list = ['u94','u821','u1474','u5937','u5497','u1031','u1472','u5193','u1854','u5096','u3887','u3125','u2796','u1726','u2231','u58','u2895','u3261','u1501','u5747','u3475','u1764','u1884','u4732','u1451','u5312','u3705','u1737','u2777','u4979','u4647','u4312','u4725','u4425','u1010','u4543','u1285','u5367','u424','u1449','u1015','u1980','u3618','u2063','u889','u1181','u1941','u4277','u1680','u4169','u3370','u1682','u3661','u25','u5224','u5132','u198','u1472','u1209','u5026','u3712','u2144','u4600','u2824','u1909','u3010','u2898','u5150','u4126','u5319','u4374','u5924','u5261','u1380','u5896','u4721','u160','u1990','u1752','u2611','u2272','u1686','u1261','u4635','u3072','u5462','u4667','u2880','u4282','u4094','u3492','u450','u3605','u1231','u1923','u5432','u650','u3353','u1777','u525','u2482','u1535','u5830','u2036','u1239','u4674','u5785','u918','u2968','u1407','u2684','u59','u3529','u518','u5663','u4726','u5990','u2624','u3217','u2913','u601','u4611','u2188','u3900','u5744','u3967','u5268','u4771','u4472','u854','u1701','u4408','u5759','u1125','u1920','u5996','u1988','u411','u3562','u721','u1812','u5433','u1899','u5605','u4482','u2529','u5333','u4085','u2907','u531','u1051','u3292','u2106','u1605','u752','u5643','u1448','u3539','u3224','u1088','u1150','u2128','u4905','u3933','u383','u1633','u5304','u1766','u2259','u6004','u3325','u1475','u2512','u5904','u3589','u4870','u5905','u4195','u1212','u2267','u75','u4610','u2851','u1525','u4231','u5906','u982','u5672','u601','u1236','u4347','u887','u4912','u704','u1323','u1609','u1350','u5579','u4336','u2226']
csv_file = 'sampled_users.csv'

users_male, users_female = filter_users_by_gender(user_list, csv_file)
print("Male Users:", users_male)
print("Female Users:", users_female)
groups = {
    "male": users_male,
    "female": users_female
}

output_data = []
steiner_tree_count = 0

for group_id, users_list in groups.items():
    print(f"Processing group: {group_id}")
    for k in range(1, 11):
        tracemalloc.start()
        start_time = time.time()
        snapshot_before = tracemalloc.take_snapshot()
        print(f"Processing k={k} for group {group_id}")
        group_terminal_nodes = get_group_terminal_nodes(users_list, file_path, k)

        missing_nodes = [node for node in group_terminal_nodes if node not in initial_graph]
        if missing_nodes:
            print(f"Warning: Terminal nodes missing in the graph: {missing_nodes}")

        verify_graph_attributes(initial_graph, group_terminal_nodes)

        steiner_tree_result = run_pcst(initial_graph, group_terminal_nodes)

        if len(steiner_tree_result.edges()) == 0:
            print(f"Warning: Empty Steiner tree for group {group_id} with k={k}")

        sum_weight = sum(data['weight'] for u, v, data in steiner_tree_result.edges(data=True))
        sum_w0 = sum(initial_graph[u][v]['weight'] for u, v in steiner_tree_result.edges())
        steiner_summary = list(steiner_tree_result.edges(data=True))
        sum_length = len(steiner_tree_result.edges())
        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        memory_diff = sum(stat.size_diff for stat in top_stats)
        print(f"Memory usage at k={k}: {memory_diff / 1024:.2f} KB")
        tracemalloc.stop()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Time usage at k={k}: {execution_time} sec")
        group_data = {
            'group_id': group_id,
            'k': k,
            'lambda': lambda_param,
            'terminal_nodes': group_terminal_nodes,
            'steiner_summary': steiner_summary,
            'sum_metrics': {
                'sum_lengths': sum_length,
                'sum_weight': sum_weight,
                'original_weight': sum_w0
            },
            'performance': {
                'execution_time': execution_time,
                'memory_usage': memory_diff / (1024 ** 2)  # MB
            }
        }
        output_data.append(group_data)
        steiner_tree_count += 1  # Increment the counter
        print(f"Generated Steiner tree {steiner_tree_count} for group {group_id} with k={k}")

with open("performance-final-pcst-user_group-pgpr.jsonl", 'w') as jsonl_file: #change pgpr to cafe for the other baseline
    for group_data in output_data:
        jsonl_file.write(json.dumps(group_data) + '\n')
