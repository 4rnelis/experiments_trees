'''
In this file, I will conduct graph analysis regarding selected metrics from the articles. 
'''

from itertools import combinations
import networkx as nx
import math
from collections import Counter, defaultdict
import numpy as np
import math

# Translate the fault tree to NX graph for stats
def get_nx_graph(G):
    FT = nx.DiGraph()
    lijst = list(G.graph.keys())
    for source in lijst:
        for target in G.graph[source]:
            FT.add_edge(target, source)

    for node in FT.nodes:
        FT.nodes[node]["type"] = G.gates.get(node, {})
    return FT

# Counts the number of nodes in the graph
def graph_size(G):
    return len(G.nodes())

# NX function for density (0-1)
def graph_density(G):
    return nx.density(G)

# Returns the average in-degree of a connector
def avg_connector_degree(G):
    connector_nodes = [n for n, d in G.nodes(data=True) if d.get('type') != {}]
    if not connector_nodes:
        return 0
    in_degrees = [G.in_degree(n) for n in connector_nodes]
    return sum(in_degrees) / len(connector_nodes)

# Returns the maximum in-degree of any connector
def max_connector_degree(G):
    connector_nodes = [n for n, d in G.nodes(data=True) if d.get('type') != {}]
    if not connector_nodes:
        return 0
    
    in_degrees = [G.in_degree(n) for n in connector_nodes]
    return max(in_degrees)


def sequentiality(G):
    # Check if tree
    if not nx.is_directed_acyclic_graph(G):
        return 0  
    
    topo_order = list(nx.topological_sort(G))
    node_pos = {node: i for i, node in enumerate(topo_order)}
    
    consecutive_edge_count = 0
    for u, v in G.edges():
        # Check if v immediately follows u in topo order
        if node_pos[v] - node_pos[u] == 1:
            consecutive_edge_count += 1
            
    return consecutive_edge_count / G.number_of_edges() if G.number_of_edges() > 0 else 0

# Difference/mismatch between in and out-degrees
def connector_mismatch(G):
    if not nx.is_directed(G):
        return 0
    connector_nodes = [n for n, d in G.nodes(data=True) if d.get('type') != {}]

    in_degrees = [G.in_degree(n) for n in connector_nodes]
    out_degrees = [G.out_degree(n) for n in connector_nodes]
    return np.mean(np.abs(np.array(in_degrees) - np.array(out_degrees)))

# Variation in connector degrees
def connector_heterogeneity(G):
    connector_nodes = [n for n, d in G.nodes(data=True) if d.get('type') != {}]
    degrees = [G.degree(n) for n in connector_nodes]
    return np.std(degrees)

# Depth of the tree from root to deepest leaf (in inverse DAG)
def number_of_levels(G):
    if nx.is_directed_acyclic_graph(G):
        return nx.dag_longest_path_length(G.reverse()) + 1  # levels = edges + 1
    return 0

# Average number of children per node, based on reversed graph
# Since every node that branches out, it is the same as the avg. connector degree
def branching_factor(G):
    if nx.is_directed_acyclic_graph(G):
        G_rev = G.reverse()
        degrees = [G_rev.out_degree(n) for n in G_rev.nodes()]
        internal_nodes = [d for d in degrees if d > 0]
        return sum(internal_nodes) / len(internal_nodes) if internal_nodes else 0
    return 0

# Maximum number of nodes at any depth level in the reversed graph
def tree_width(G):
    if nx.is_directed_acyclic_graph(G):
        G_rev = G.reverse()
        levels = {}
        for node in nx.topological_sort(G_rev):
            level = max([levels.get(p, 0) for p in G_rev.predecessors(node)], default=0) + 1
            levels[node] = level
        width = Counter(levels.values())
        return max(width.values()) if width else 0
    return 0

# # ======================
# # Visual Layout Metrics
# # ======================

# # utils for orthogonal geometry
# def is_horizontal(p1, p2):
#     return math.isclose(p1[1], p2[1], abs_tol=1e-3)

# def is_vertical(p1, p2):
#     return math.isclose(p1[0], p2[0], abs_tol=1e-3)

# def is_orthogonal(p1, p2):
#     return is_horizontal(p1, p2) or is_vertical(p1, p2)

# def orthogonal_coords(p11, p12, p21, p22):
#     return is_orthogonal(p11, p12) and is_orthogonal(p21, p22)

# def clean_point(p):
#     return (round(p[0], 2), round(p[1], 2))

# def segments_from_spline(points):
#     cleaned = [clean_point(p) for p in points]
#     return [(cleaned[i], cleaned[i + 1]) for i in range(len(cleaned) - 1)]


# def orthogonal_lines_intersect(p1, p2, q1, q2):
#     if is_horizontal(p1, p2) and is_horizontal(q1, q2):
#         if math.isclose(p1[1], q1[1], abs_tol=1e-3):
#             x1_min, x1_max = sorted([p1[0], p2[0]])
#             x2_min, x2_max = sorted([q1[0], q2[0]])
#             return x1_max >= x2_min and x2_max >= x1_min

#     elif is_vertical(p1, p2) and is_vertical(q1, q2):
#         if math.isclose(p1[0], q1[0], abs_tol=1e-3):
#             y1_min, y1_max = sorted([p1[1], p2[1]])
#             y2_min, y2_max = sorted([q1[1], q2[1]])
#             return y1_max >= y2_min and y2_max >= y1_min

#     elif is_horizontal(p1, p2) and is_vertical(q1, q2):
#         hx1, hx2 = sorted([p1[0], p2[0]])
#         hy = p1[1]
#         vy = sorted([q1[1], q2[1]])
#         vx = q1[0]
#         return hx1 <= vx <= hx2 and vy[0] <= hy <= vy[1]

#     elif is_vertical(p1, p2) and is_horizontal(q1, q2):
#         return orthogonal_lines_intersect(q1, q2, p1, p2)

#     return False


# def edge_crossings_from_splines(splines):
#     crossings = 0
#     edges = list(splines.items())

#     for i in range(len(edges)):
#         (u1, v1), pts1 = edges[i]
#         segs1 = segments_from_spline(pts1)

#         for j in range(i + 1, len(edges)):
#             (u2, v2), pts2 = edges[j]
#             if len({u1, v1, u2, v2}) < 4:
#                 continue

#             segs2 = segments_from_spline(pts2)

#             for s1 in segs1:
#                 if not is_orthogonal(*s1):
#                     continue
#                 for s2 in segs2:
#                     if not is_orthogonal(*s2):
#                         continue
#                     if orthogonal_lines_intersect(*s1, *s2):
#                         print(f's1: {s1}, s2: {s2}')
#                         crossings += 1
#                         break
#     return crossings


# def node_overlap(pos, threshold=1):
#     """Count overlapping nodes based on positions"""
#     positions = list(pos.values())
#     overlaps = 0
#     for i in range(len(positions)):
#         for j in range(i+1, len(positions)):
#             if np.linalg.norm(np.array(positions[i]) - np.array(positions[j])) < threshold:
#                 overlaps += 1
#     return overlaps

# def aspect_ratio(pos):
#     """Width/height ratio of the layout"""
#     if not pos:
#         return 1
#     x_vals = [p[0] for p in pos.values()]
#     y_vals = [p[1] for p in pos.values()]
#     width = max(x_vals) - min(x_vals)
#     height = max(y_vals) - min(y_vals)
#     return width / height if height != 0 else float('inf')

# ======================
# Cognitive Load Metrics
# ======================

def path_complexity(G):

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG")
    
    # Find root nodes (no predecessors)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    # Find leaf nodes (no successors)
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    paths = []
    for root in roots:
        # print(f"ROOT: {root}")
        for leaf in leaves:
            # print(f"LEAF: {leaf}")
            for path in nx.all_simple_paths(G, root, leaf):
                paths.append(len(path))
                # print(f"LEN: {len(path)} PATH: {path}")
    
    return np.mean(paths) if paths else 0

# Ratio of edges to possible edges
def information_density(G):
    return nx.density(G)

# Textual information per unit area
def label_density(G, avg_label_length=5):
    total_chars = sum(len(str(n)) for n in G.nodes())
    return total_chars / len(G.nodes()) if G.nodes() else 0

# Measure of different gate types (if annotated)
def gate_diversity(G):
    if 'type' in next(iter(G.nodes(data=True)))[1]:
        types = [data['type'] if isinstance(data.get('type'), str) else 'unknown' for _, data in G.nodes(data=True)]

        return len(set(types)) / len(types) if types else 0
    return 0

# Advanced Stuff

def subtree_signature_inverse(G, node):
    children = list(G.predecessors(node))  # children → parent edges
    if not children:
        return "()"
    child_sigs = sorted(subtree_signature_inverse(G, c) for c in children)
    return "(" + "".join(child_sigs) + ")"

def graph_symmetry_soft(G):
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG")
    
    # In inverse trees, root has no outgoing edges
    roots = [n for n in G.nodes() if G.out_degree(n) == 0]
    # if len(roots) != 1:
    #     print(roots)
    #     raise ValueError("Graph must have exactly one root")
    root = roots[0]
    
    scores = []
    
    def compute_symmetry(node):
        children = list(G.predecessors(node))
        if len(children) < 2:
            return  # no symmetry possible

        sigs = [subtree_signature_inverse(G, c) for c in children]
        total = 0
        matches = 0
        for a, b in combinations(sigs, 2):
            total += 1
            if a == b:
                matches += 1
        if total > 0:
            scores.append(matches / total)
        
        for c in children:
            compute_symmetry(c)

    compute_symmetry(root)
    return sum(scores) / len(scores) if scores else 1.0

# Information entropy of the degree distribution
def graph_entropy(G):
    degrees = [d for n, d in G.degree()]
    prob = np.array(degrees) / sum(degrees)
    return -np.sum(prob * np.log2(prob + 1e-10))

# Total label information / the size of the graph
def information_density(G):
    info_content = sum(len(str(n)) for n in G.nodes())
    return info_content / graph_size(G)

# Mother func

def analyze_graph(G, pos=None, splines=None):
    results = {
        'Number of Nodes': graph_size(G),
        'Graph Density': graph_density(G),
        'Avg. Connector Degree': avg_connector_degree(G),
        'Max. Connector Degree': max_connector_degree(G),
        'Sequentiality': sequentiality(G),
        'Connector Mismatch': connector_mismatch(G),
        'Connector Heterogeneity': connector_heterogeneity(G),
        'Number of Levels': number_of_levels(G),
        'Branching Factor': branching_factor(G),
        'Tree Width': tree_width(G),
        'Path Complexity': path_complexity(G),
        'Information Density': information_density(G),
        'Label Density': label_density(G),
        'Gate Diversity': gate_diversity(G),
        'Graph Entropy': graph_entropy(G),
        'Graph Symmetry': graph_symmetry_soft(G)
    }
    
    # if pos:
    #     results['Visual Layout'] = {
    #         'Edge Crossings': edge_crossings_from_splines(splines),
    #         'Node Overlap': node_overlap(pos),
    #         'Aspect Ratio': aspect_ratio(pos)
    #     }
    
    return results

