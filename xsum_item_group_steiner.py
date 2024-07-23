import json
import networkx as nx
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import tracemalloc
import time


def filter_users_by_gender():
#item ids
    users_male = ['763', '781', '2156', '2934', '2636', '1250', '1003', '2412', '1524', '1480', '2933', '455', '2189', '2244', '1087', '1060', '1924', '1451', '1183', '2499', '1461', '852', '1376', '1265', '1552', '782', '1691', '1833', '328', '1749', '1880', '2083', '1280', '2702', '1663', '1316', '1607', '1230', '1186', '484', '1834', '2733', '1573', '999', '2798', '2698', '2667', '789', '2407', '2101']
    users_female = ['2149', '491', '1824', '2552', '2973', '1577', '307', '2829', '2552', '2153', '517', '926', '2721', '2782', '959', '780', '1946', '403', '2033', '2321', '2697', '2911', '2949', '371', '410', '2523', '2279', '784', '2535', '2409', '2431', '2676', '536', '1097', '2545', '152', '2823', '82', '1859', '2287', '2333', '1182', '492', '334', '2125', '1057', '445', '2744', '1417', '1859']
    return users_male, users_female

def get_R_u(file_path, user):
    R_u = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            user_id = str(data['item_id'])  # Ensure user_id is a string
            if user_id == user:
                recommended_items_ids = list(map(str, data['recommended_users_ids']))  # Ensure item IDs are strings
                R_u[user_id] = recommended_items_ids
                return dict(R_u)

def compute_user_new_weights_new(filepath, initial_graph, R_u, lambda_param, user):
    new_weights = {}
    with open(filepath, 'r') as file:
        for line in file:
            data = json.loads(line)
            user_id = str(data['item_id'])
            if user_id == user:
                top_k_paths = data['top_k_paths']
                pairs = [(path[i][-1], path[i + 1][-1]) for path in top_k_paths for i in range(len(path) - 1)]
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
    print(f"Computed new weights for item {user}: {new_weights}")
    return new_weights

def update_weights_new(initial_graph, new_weights):
    updated_graph = initial_graph.copy()
    for edge, weight in new_weights.items():
        u, v = edge
        if updated_graph.has_edge(u, v):
            updated_graph[u][v]['weight'] = weight
        elif updated_graph.has_edge(v, u):
            updated_graph[v][u]['weight'] = weight
    print("Updated graph weights.")
    return updated_graph

def shift_weights(graph):
    min_weight = min(data['weight'] for u, v, data in graph.edges(data=True))
    shift_value = abs(min_weight) + 1 if min_weight < 0 else 0
    for u, v, data in graph.edges(data=True):
        data['weight'] += shift_value
    print(f"Shifted graph weights by {shift_value}.")
    return graph, shift_value

def find_terminal_nodes(R_u, user, k):
    terminal_nodes = set(R_u[user][:k])
    terminal_nodes.add(user)
    return list(terminal_nodes)

def precompute_shortest_paths(graph, terminal_nodes, weight='weight'):
    shortest_paths = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(nx.single_source_dijkstra, graph, node, weight=weight): node for node in
                   terminal_nodes}
        for future in as_completed(futures):
            node = futures[future]
            length_dict, path_dict = future.result()
            for target in terminal_nodes:
                if node != target:
                    shortest_paths[(node, target)] = (length_dict[target], path_dict[target])
    return shortest_paths

def steiner_tree(graph, terminal_nodes, shortest_paths, weight='weight'):
    complete_graph = nx.Graph()
    for i in range(len(terminal_nodes)):
        for j in range(i + 1, len(terminal_nodes)):
            length, path = shortest_paths[(terminal_nodes[i], terminal_nodes[j])]
            complete_graph.add_edge(terminal_nodes[i], terminal_nodes[j], weight=length)
    mst = nx.minimum_spanning_tree(complete_graph, weight=weight)
    steiner_tree = nx.Graph()
    for u, v, data in mst.edges(data=True):
        length, path = shortest_paths[(u, v)]
        nx.add_path(steiner_tree, path, weight=length)
    print(f"Generated Steiner tree with terminal nodes: {terminal_nodes}")
    return steiner_tree

def get_group_terminal_nodes(users_list, file_path, k):
    group_terminal_nodes = set()
    for user in users_list:
        R_u = get_R_u(file_path, user)
        if R_u:
            terminal_nodes = find_terminal_nodes(R_u, user, k)
            group_terminal_nodes.update(terminal_nodes)
    print(f"Combined terminal nodes for group at k={k}: {group_terminal_nodes}")
    return list(group_terminal_nodes)

#baselines and knowledge graph
file_path = 'pgpr_item_paths.jsonl' #change pgpr to cafe for the other baseline
graph_filename = 'kg_static.graphml'
initial_graph = nx.read_graphml(graph_filename)

lambda_param = 1 #adjust for different lambda values

# items
user_list = ['763', '781', '2156', '2934', '2636', '1250', '1003', '2412', '1524', '1480', '2933', '455', '2189', '2244', '1087', '1060', '1924', '1451', '1183', '2499', '1461', '852', '1376', '1265', '1552', '782', '1691', '1833', '328', '1749', '1880', '2083', '1280', '2702', '1663', '1316', '1607', '1230', '1186', '484', '1834', '2733', '1573', '999', '2798', '2698', '2667', '789', '2407', '2101', '2149', '491', '1824', '2552', '2973', '1577', '307', '2829', '2552', '2153', '517', '926', '2721', '2782', '959', '780', '1946', '403', '2033', '2321', '2697', '2911', '2949', '371', '410', '2523', '2279', '784', '2535', '2409', '2431', '2676', '536', '1097', '2545', '152', '2823', '82', '1859', '2287', '2333', '1182', '492', '334', '2125', '1057', '445', '2744', '1417', '1859']

users_male, users_female = filter_users_by_gender()
print("Most popular items:", users_male)
print("Less popular items:", users_female)
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
        print(f"Processing k={k} for group {group_id}")
        group_terminal_nodes = get_group_terminal_nodes(users_list, file_path, k)
        combined_R_u = defaultdict(list)
        for user in users_list:
            R_u = get_R_u(file_path, user)
            if R_u:
                combined_R_u[user] = R_u[user][:k]
        new_weights = {}
        for user in combined_R_u:
            new_weights.update(compute_user_new_weights_new(file_path, initial_graph, combined_R_u, lambda_param, user))
        updated_graph = update_weights_new(initial_graph, new_weights)
        for u, v, data in updated_graph.edges(data=True):
            data['weight'] = -data['weight']
        shifted_graph, shift_value = shift_weights(updated_graph)

        # Precompute shortest paths between terminal nodes
        shortest_paths = precompute_shortest_paths(shifted_graph, group_terminal_nodes)

        steiner_tree_result = steiner_tree(shifted_graph, group_terminal_nodes, shortest_paths)
        sum_weight = sum(data['weight'] for u, v, data in steiner_tree_result.edges(data=True))
        sum_w0 = sum(initial_graph[u][v]['weight'] for u, v in steiner_tree_result.edges())
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

with open("performance-final-steiner-item_group-pgpr.jsonl", 'w') as jsonl_file: #change pgpr to cafe for the other baseline
    for group_data in output_data:
        jsonl_file.write(json.dumps(group_data) + '\n')
