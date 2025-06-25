#!/usr/bin/env python

#  Copyright (C) 2023 Milan Lopuhaa-Zwakenberg
#
# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software: you can redistribute it and/or modifya
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ##### END GPL LICENSE BLOCK #####

# from func_timeout import func_timeout, FunctionTimedOut
from visualization import visualize_ftg
from translate import parse_dft_sft
from nx_stats import get_nx_graph, analyze_graph

# afdruk = open("logs/sfpa.csv","w")

# for i in range(1, 6):
#     address = 'visual_testing/keep40_'+str(i)+'.dft'
#     G = parse_dft_sft(address)
#     # now = time.time()


#     print(visualize_ftg(G, show_plot=True))
#     # visualize_ftg(G)

# Does not work with dynamic trees with multiple factors currently
# Does not work with special characters in node naming
# lambda is required for some reason due to parsing
address = 'ffort_samples/CAS.dft'
G = parse_dft_sft(address)


FT = get_nx_graph(G)
_, pos, splines = visualize_ftg(G, show_plot=True)
analysis_data = analyze_graph(FT, pos=pos, splines=splines)
for item in analysis_data.items():
    print(f"{item[0]}: {item[1]}")

