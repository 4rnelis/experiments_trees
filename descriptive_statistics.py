'''
Here the metric values for each graph
as well as descriptive statistics will be calculated and formatted into excel output file
mean median std min max
'''

import os
import pandas as pd
from visualization import visualize_ftg
from translate import parse_dft_sft
from nx_stats import analyze_graph, get_nx_graph

DATA_FIELDS = [
    "Number of Nodes", "Number of Levels", "Avg. Connector Degree", "Sequentiality", "Connector Mismatch",
    "Connector Heterogeneity", "Branching Factor", "Path Complexity", "Max. Connector Degree",  
    "Label Density", "Gate Diversity", "Graph Entropy", "Graph Density"
]
 
EXCEL_FILE = 'results/ft_metrics.xlsx'

def get_data(folder):
    if not os.path.isdir(folder):
        raise ValueError(f"The folder '{folder}' does not exist")

    results_df = pd.DataFrame(columns=DATA_FIELDS)

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            try:
                results = get_analysis_from_file(filepath)
                results_df.loc[filename] = results
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

    summary_df = pd.DataFrame({
        "mean": results_df.mean(),
        "median": results_df.median(),
        "std": results_df.std(),
        "min": results_df.min(),
        "max": results_df.max()
    })

    # Compute Z-scores per metric for each sample
    z_scores_df = (results_df - results_df.mean()) / results_df.std()

    with pd.ExcelWriter(EXCEL_FILE) as writer:
        results_df.to_excel(writer, sheet_name="Per File Metrics", index=True, index_label="Filename")
        summary_df.to_excel(writer, sheet_name="Summary Stats", index=True, index_label="Metric")
        z_scores_df.to_excel(writer, sheet_name="Z-Scores", index=True, index_label="Filename")

def get_analysis_from_file(filename):
    G = parse_dft_sft(filename=filename)
    FT = get_nx_graph(G)
    _, pos, splines = visualize_ftg(FT, show_plot=False)
    return analyze_graph(FT, pos=pos, splines=splines)

if __name__ == "__main__":
    get_data('ffort_samples')
    # get_analysis_from_file('ffort_samples/cabinets.3-3.dft')
