#!/usr/bin/env python

#  Copyright (C) 2023 Milan Lopuhaa-Zwakenberg
#
# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software: you can redistribute it and/or modify
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

import sympy as sp
import networkx as nx
import itertools
import matplotlib.pyplot as plt


def unreliability(G,allprobs):
    FT = nx.DiGraph()
    # Translate graph into NX digraph
    lijst = list(G.graph.keys())
    for source in lijst:
        for target in G.graph[source]:
            FT.add_edge(source,target)
    todo = list(reversed(list(nx.topological_sort(FT))))
    dominators = nx.immediate_dominators(FT,G.tle)
    dominees = dict()
    for v in todo:
        dominees_v = [k for k,val in dominators.items() if val == v]
        dominees_v.sort(key = lambda i:todo.index(i),reverse=True)
        dominees.update({v:dominees_v})
    todo = list(reversed(list(nx.topological_sort(FT))))
    n = len(todo)+1
    F = sp.symbols('F1:%d'%n)
    g = dict()
    for v in todo:
        if v in G.bes:
            g.update({v:G.bes[v]*F[0]**0})
        else:
            g_building = 1
            if G.gates[v] == 'or':
                for w in FT.successors(v):
                    g_building  = g_building*(1-F[todo.index(w)])
                g_building = (1-g_building)
            else:
                for w in FT.successors(v):
                    g_building = g_building*F[todo.index(w)]
            for w in dominees[v]:
                if v!= w:
                    g_building = sqfsub(g_building,F[todo.index(w)],g[w])
            g.update({v:g_building})
    if allprobs == False:
        return g[G.tle]
    else:
        h = dict()
        todo2 = list(nx.topological_sort(FT))
        for v in todo2:
            h_building = g[v]
            for w in todo2:
                if nx.has_path(FT,v,w) and todo.index(w) < n and sp.degree(h_building,F[todo.index(w)]) > 0:
                    h_building = sqfsub(h_building,F[todo.index(w)],g[w])
            h.update({v:h_building})
        return h
                    
def sqfmult(p1,p2):
    prod=p1*p2
    tocheck = set.intersection(p1.free_symbols,p2.free_symbols)
    return reduce(prod,tocheck)

def sqfsub(p1,x,p2):
    result = p1.subs(x,1)*p2 + p1.subs(x,0)*(1-p2)
    tocheck = set.intersection(p1.free_symbols,p2.free_symbols)
    return reduce(result,tocheck)

def reduce(p,tocheck):
    if tocheck == set():
        return p
    else:
        generators = []
        for x in tocheck:
            generators.append(x**2-x)
        return sp.reduced(p,generators)[1]
