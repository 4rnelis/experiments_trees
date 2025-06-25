import pygraphviz as pgv
import networkx as nx
import matplotlib.pyplot as plt
import os

def visualize_ftg(G, save_path="fault_tree.png", show_plot=True):
    # Create PyGraphviz AGraph (directed)
    FT = pgv.AGraph(directed=True, strict=False)
    
    # Add edges (reversed for bottom-up hierarchy)
    for source, targets in G.graph.items():
        for target in targets:
            FT.add_edge(target, source)

    GATE_IMAGES = {
        "and": r"gates\and_gate.png",
        "or": r"gates\or_gate.png",
        "not": r"gates\not_gate.png",
        "xor": r"gates\xor_gate.png",
        "atleast": r"gates\atleast_gate.png",
        "fdep": r"gates\fdep_gate.png",
        "csp": r"gates\csp_gate.png",
        "unknown": r"gates\unknown_gate.png"
    }

    for node in FT.nodes():
        gate_info = G.gates.get(node.name, {})
        gate_type = gate_info.get('type', 'unknown') if isinstance(gate_info, dict) else str(gate_info)
        
        if gate_type == 'unknown':
            label = node.name
        elif gate_type.startswith("('vot'"):
            n = int(eval(gate_type)[1])
            label = f"<<FONT POINT-SIZE='12'><B>{node.name}</B></FONT><BR/><FONT POINT-SIZE='10'><B>K/N</B><BR/><B>{n}/{FT.in_degree(node)}</B></FONT>>"
        else:
            label = f"<<FONT POINT-SIZE='12'><B>{node.name}</B></FONT><BR/><FONT POINT-SIZE='10'><B>{gate_type}</B></FONT>>"
        
        # Assign image or fallback shape
        if gate_type.startswith("('vot'"):
            gate_type = 'atleast'
        # print(f"gt: {gate_type}; label: {label}")
        img_path = GATE_IMAGES.get(gate_type, GATE_IMAGES["unknown"])
        if os.path.exists(img_path):
            FT.get_node(node).attr.update(
                image=img_path,
                fontsize="12",
                shape="none",
                width=1.0,
                height=1.0,
                label=label,
                margin=0,
                penwidth=0
            )
        else:
            FT.get_node(node).attr.update(
                shape="circle",
                fillcolor="#c7c7c7",
                style="filled",
                label=label,
                width=1.0,
                height=1.0
            )

    # --- Graph Attributes (Orthogonal Layout) ---
    FT.graph_attr.update(
        rankdir="BT",        # Bottom-to-top hierarchy
        splines="ortho",     # Orthogonal edges
        nodesep="0.5",       # Horizontal spacing
        ranksep="0.75",      # Vertical spacing
        concentrate="true",  # Merge edge lines
    )
    FT.node_attr.update(
        fontname="Helvetica",
        fontcolor="#2e3440"
    )
    FT.edge_attr.update(
        arrowsize="0.8",
        color="#3b4252"
    )

    FT.layout(prog="dot")

    pos = {}
    for n in FT.nodes():
        p = n.attr['pos']
        if p:
            x, y = map(float, p.split(','))
            pos[n.name] = (x, y)

    # Extract edge spline points or smthng
    splines = {}

    for e in FT.edges():
        edge_key = (e[0], e[1])
        spline_raw = e.attr.get("pos")

        if spline_raw:
            spline_raw = spline_raw.translate(str.maketrans('', '', '\\'))
            # print(spline_raw)
            points = []
            for part in spline_raw.strip().split(' '):
                if ',' in part:
                    clean = part.strip('e,')
                    x, y = map(float, clean.split(','))
                    points.append((x, y))
            if points:
                splines[edge_key] = points


    FT.draw(save_path)

    if show_plot:
        img = plt.imread(save_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Fault Tree Viz", pad=20)
        plt.tight_layout()
        plt.show()

    return FT, pos, splines

# import networkx as nx
# import matplotlib.pyplot as plt
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.image as mpimg
# import os

# def visualize_ftg(G, save_path=False, show_plot=False):
#     FT = nx.DiGraph()
    
#     # Build the graph (edges reversed as in your original code)
#     for source, targets in G.graph.items():
#         for target in targets:
#             FT.add_edge(target, source)  # Note: Still reversed (adjust if needed)

#     # Define paths to gate PNGs (replace with your actual paths)
#     GATE_IMAGES = {
#         "and": r"gates\and_gate.png",
#         "or": r"gates\or_gate.png",
#         "not": r"gates\not_gate.png",
#         "xor": r"gates\xor_gate.png",
#         "atleast": r"gates\atleast_gate.png",
#         "fdep": r"gates\fdep_gate.png",
#         "csp": r"gates\csp_gate.png",
#         "unknown": r"gates\unknown_gate.png"  # Fallback image
#     }

#     # Create labels (same as original)
#     custom_labels = {}
#     for node in FT.nodes():
#         gate_info = G.gates.get(node, {})
#         gate_type = gate_info.get('type', 'unknown') if isinstance(gate_info, dict) else str(gate_info)

#         if gate_type == 'unknown':
#             custom_labels[node] = f"{node}"
#         elif gate_type.startswith("('vot'"):
#             n = int(eval(gate_type)[1])
#             custom_labels[node] = f"{node}\nK/N\n{n}/{FT.in_degree(node)}"
#         else: custom_labels[node] = f"{node}\n{gate_type}"

        
    
#     # Layout (hierarchical if possible)
#     try:
#         # Use dot with orthogonal edge routing
#         pos = nx.nx_agraph.graphviz_layout(
#             FT, prog='dot', args='-Grankdir=BT -Gsplines=ortho -Gnodesep=0.5 -Granksep=0.75'
#         )
#     except ImportError:
#         print("Note: Install pygraphviz for hierarchical layouts (pip install pygraphviz).")
#         pos = nx.spring_layout(FT, seed=42)



#     # Draw the graph
#     fig, ax = plt.subplots(figsize=(12, 8))
    
#     # Draw edges first
#     nx.draw_networkx_edges(
#         FT, pos, ax=ax,
#         edge_color="gray",
#         arrows=True,
#         arrowsize=15,
#         connectionstyle='arc3,rad=0'  # Prevent curved arcs
#     )

#     # Draw each node as an image
#     for node, (x, y) in pos.items():
#         gate_info = G.gates.get(node, {})
#         gate_type = gate_info.get('type', 'unknown') if isinstance(gate_info, dict) else str(gate_info)
#         print(gate_type)
#         if gate_type.startswith("('vot'"):
#             gate_type = "atleast"
#         img_path = GATE_IMAGES.get(gate_type, GATE_IMAGES["unknown"])
        


        
#         if os.path.exists(img_path):
#             img = mpimg.imread(img_path)
#             imbox = OffsetImage(img, zoom=0.3)  # Adjust zoom to fit
#             ab = AnnotationBbox(imbox, (x, y), frameon=False)
#             ax.add_artist(ab)
#         else:
#             # Fallback: Draw a red circle if image missing
#             nx.draw_networkx_nodes(
#                 FT, pos, nodelist=[node],
#                 node_shape="o",
#                 node_color="gray",
#                 node_size=750,
#                 ax=ax
#             )

#     # Add labels
#     nx.draw_networkx_labels(
#         FT, pos, custom_labels,
#         ax=ax,
#         font_size=5,
#         font_weight="bold"
#     )

#     plt.title("Fault tree visual")
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#         print(f"Graph saved to {save_path}")
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()

#     return FT
