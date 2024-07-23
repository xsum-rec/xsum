import json
import networkx as nx
from collections import defaultdict
import pcst_fast
import numpy as np
import time
import tracemalloc

def get_R_u(file_path, user):
    R_u = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            user_id = str(data['item_id'])
            if user_id == user:
                recommended_items_ids = list(map(str, data['recommended_users_ids']))
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
        graph.edges[u, v]['weight'] = 0  # Default weight

def run_pcst(graph, terminal_nodes):
    node_indices = {node: idx for idx, node in enumerate(graph.nodes())}
    reverse_node_indices = {idx: node for node, idx in node_indices.items()}
    prizes = np.array([graph.nodes[node]['prize'] for node in graph.nodes()], dtype=np.float64)
    edges = np.array([[node_indices[u], node_indices[v]] for u, v in graph.edges()], dtype=np.int64)
    edge_weights = np.array([graph.edges[u, v].get('weight', 1.0) for u, v in graph.edges()], dtype=np.float64)

    # Run pcst_fast
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

def filter_users_by_gender(): # Ids are item ids
    users_male = ['763', '781', '2156', '2934', '2636', '1250', '1003', '2412', '1524', '1480', '2933', '455', '2189', '2244', '1087', '1060', '1924', '1451', '1183', '2499', '1461', '852', '1376', '1265', '1552', '782', '1691', '1833', '328', '1749', '1880', '2083', '1280', '2702', '1663', '1316', '1607', '1230', '1186', '484', '1834', '2733', '1573', '999', '2798', '2698', '2667', '789', '2407', '2101']
    users_female = ['2149', '491', '1824', '2552', '2973', '1577', '307', '2829', '2552', '2153', '517', '926', '2721', '2782', '959', '780', '1946', '403', '2033', '2321', '2697', '2911', '2949', '371', '410', '2523', '2279', '784', '2535', '2409', '2431', '2676', '536', '1097', '2545', '152', '2823', '82', '1859', '2287', '2333', '1182', '492', '334', '2125', '1057', '445', '2744', '1417', '1859']

    return users_male, users_female

# Baseline paths and knowledge graph
file_path = ('pgpr_item_paths.jsonl') #change pgpr to cafe for the other baseline
graph_filename = 'kg_static.graphml'
initial_graph = nx.read_graphml(graph_filename)
lambda_param = 1 #adjust for different lambda values
#items
user_list = ['763', '781', '2156', '2934', '2636', '1250', '1003', '2412', '1524', '1480', '2933', '455', '2189', '2244', '1087', '1060', '1924', '1451', '1183', '2499', '1461', '852', '1376', '1265', '1552', '782', '1691', '1833', '328', '1749', '1880', '2083', '1280', '2702', '1663', '1316', '1607', '1230', '1186', '484', '1834', '2733', '1573', '999', '2798', '2698', '2667', '789', '2407', '2101', '2149', '491', '1824', '2552', '2973', '1577', '307', '2829', '2552', '2153', '517', '926', '2721', '2782', '959', '780', '1946', '403', '2033', '2321', '2697', '2911', '2949', '371', '410', '2523', '2279', '784', '2535', '2409', '2431', '2676', '536', '1097', '2545', '152', '2823', '82', '1859', '2287', '2333', '1182', '492', '334', '2125', '1057', '445', '2744', '1417', '1859']

users_male, users_female = filter_users_by_gender()
print("Most popular items:", users_male) #popular items
print("Less popular items:", users_female) #unpopular items
groups = {
    "pop": users_male,
    "nopop": users_female
}

output_data = []
steiner_tree_count = 0

for group_id, users_list in groups.items():
    print(f"Processing group: {group_id}")
    for k in range(1, 11):
        tracemalloc.start()
        start_time = time.time()
        snapshot_before = tracemalloc.take_snapshot()
        try:
            print(f"Processing k={k} for group {group_id}")
            group_terminal_nodes = get_group_terminal_nodes(users_list, file_path, k)

            # Verify terminal nodes exist in the graph
            missing_nodes = [node for node in group_terminal_nodes if node not in initial_graph]
            if missing_nodes:
                print(f"Warning: Terminal nodes missing in the graph: {missing_nodes}")

            # adjust pcst penalties
            verify_graph_attributes(initial_graph, group_terminal_nodes)

            # Run PCST
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
        except Exception as e:
            print(f"Error processing item {group_id} with k={k}")
            print(f"Exception: {str(e)}")
            continue

with open("performance-final-pcst-item_group-pgpr.jsonl", 'w') as jsonl_file: #change pgpr to cafe for the other baseline
    for group_data in output_data:
        jsonl_file.write(json.dumps(group_data) + '\n')