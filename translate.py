#!/usr/bin/env python
#
#  Copyright (C) 2020,2021 Daniel Basgöze
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


import xml.etree.ElementTree as ET
import lark
import random as r
import math
import argparse
from collections import deque


def write_file(name, data):
    with open(name, 'w') as f:
        d = f.write(data)
    return d

def read_file(name):
    with open(name, 'r') as f:
        d = f.read()
    return d

def remove_if_qt(string):
    if len(string) > 2 and string[0] == '"' and string[-1] == '"':
        return string[1:-1]
    return string

def add_if_not_qt(string):
    if len(string) > 2 and string[0] == '"' and string[-1] == '"':
        return string
    else:
        return f'"{string}"'


def failrate_to_prob(time, failrate):
    return 1 - math.exp(-failrate * time)


def prob_to_failrate(time, prob):
    return -math.log(1 - prob) / time


class FTG():
    def __init__(self, tle, graph, gates, bes):
        self.tle = tle
        self.graph = graph
        self.gates = gates
        self.bes = bes


def find_root(pred):
    rval = list(pred.keys())[0]
    while rval in pred:
        rval = pred[rval]
    return rval


def parse_xml_sft(filename, time):
    tree = ET.parse(filename)
    root = tree.getroot()

    for l in root.findall('.//label/..'):
        l.remove(l.find('./label'))

    events = root.findall('.//define-gate')
    probs = root.findall('.//define-basic-event')

    succ = dict()
    types = dict()
    for e in events:
        name = e.get('name')
        children = [c.get('name') for c in e[0]]
        type = e[0].tag
        # print(type)
        if type not in ['and', 'or', 'atleast', 'gate', 'basic-event']:
            # print(type)
            raise ValueError
        if type in ['gate', 'basic-event']:
            children = [e[0].get('name')]
            type = 'and'
        if type == 'atleast':
            n = e[0].get('min')
            type = ('vot', n)
        succ[name] = children
        types[name] = type

    pred = dict()
    for i, j in succ.items():
        for k in j:
            pred[k] = i
    toplevel = find_root(pred)

    bes = dict()
    for b in probs:
        name = b.get('name')
        if b[0].tag == 'float':
            value = float(b[0].get('value'))
            value = prob_to_failrate(time, value)
        elif b[0].tag == 'exponential':
            value = float(b[0][0].get('value'))
        bes[name] = value

    return FTG(toplevel, succ, types, bes)

def write_xml_scram(filename, f):
    root = ET.Element('opsa-mef')
    ft = ET.SubElement(root, 'define-fault-tree', {"name": "Translation"})

    for g,type in f.gates.items():
        tmp = ET.SubElement(ft, 'define-gate', {"name": remove_if_qt(g)})
        children = set(f.graph[g])
        if len(children) == 1:
            pass
        elif isinstance(type, tuple) and type[0] == 'vot':
            tmp = ET.SubElement(tmp, 'atleast', {'min': str(type[1])})
        else:
            tmp = ET.SubElement(tmp, type)
        for child in children:
            if child in f.bes.keys():
                ET.SubElement(tmp, 'basic-event', {"name": remove_if_qt(child)})
            if child in f.gates.keys():
                ET.SubElement(tmp, 'gate', {"name": remove_if_qt(child)})

    md = ET.SubElement(root, 'model-data')
    for e,p in f.bes.items():
        tmp = ET.SubElement(md, 'define-basic-event', {"name": remove_if_qt(e)})
        tmp = ET.SubElement(tmp, 'exponential')
        ET.SubElement(tmp, 'float', {"value": str(p)})
        ET.SubElement(tmp, 'system-mission-time')

    tree = ET.ElementTree(root)
    tree.write(filename, encoding='UTF-8', xml_declaration=True)

def write_xml_xfta(filename, f):
    root = ET.Element('opsa-mef')

    for g,type in f.gates.items():
        tmp = ET.SubElement(root, 'define-gate', {"name": remove_if_qt(g)})
        children = set(f.graph[g])
        if len(children) == 1:
            pass
        elif isinstance(type, tuple) and type[0] == 'vot':
            tmp = ET.SubElement(tmp, 'atleast', {'min': str(type[1])})
        else:
            tmp = ET.SubElement(tmp, type)
        for child in children:
            if child in f.bes.keys():
                ET.SubElement(tmp, 'basic-event', {"name": remove_if_qt(child)})
            if child in f.gates.keys():
                ET.SubElement(tmp, 'gate', {"name": remove_if_qt(child)})

    md = ET.SubElement(root, 'model-data')
    for e,p in f.bes.items():
        tmp = ET.SubElement(md, 'define-basic-event', {"name": remove_if_qt(e)})
        tmp = ET.SubElement(tmp, 'exponential')
        ET.SubElement(tmp, 'float', {"value": str(p)})
        ET.SubElement(tmp, 'mission-time')

    tree = ET.ElementTree(root)
    tree.write(filename, encoding='UTF-8', xml_declaration=True)

def DFS(graph, current):
    out = []
    ss = set()

    def inner(graph, current):
        nonlocal out
        nonlocal ss
        if current not in graph:
            if current not in ss:
                ss.add(current)
                out.append(current)
        else:
            for i in graph[current]:
                inner(graph, i)

    inner(graph, current)
    return out


def TDLR(graph, current):
    out = []
    ss = set()
    stack = deque([current])
    while stack:
        current = stack.popleft()

        if current not in graph:
            if current not in ss:
                ss.add(current)
                out.append(current)
        else:
            for i in graph[current]:
                stack.append(i)
    return out

def ext_order(graph, root, order_file):
    order = set([tuple(i.split(' ')) for i in read_file(order_file).split('\n')[:-1]])
    if len(set([i[1] for i in order])) > len(order):
        raise ValueError(f'extorder is not ordered i.e. there is an event twice in ext_order')
    order = [i[1] for i in sorted(order, key=lambda x: int(x[0]))]
    dfs_order = DFS(graph, root)
    bes = set(dfs_order)

    rval = []
    for i in order:
        if remove_if_qt(i) in bes:
            rval.append(remove_if_qt(i))
            bes.remove(remove_if_qt(i))
        elif add_if_not_qt(i):
            rval.append(add_if_not_qt(i))
            bes.remove(add_if_not_qt(i))
        else:
            raise ValueError(f'Event {i} not in graph but in extorder')

    for i in dfs_order:
        if i in bes:
            rval.append(i)
    return rval


def print_gate_aralia(name, type, children):
    rval = ''
    if type == 'and' or type == 'or':
        op = ' & ' if type == 'and' else ' | '
        rval += f'{remove_if_qt(name)} := ('
        for c in children:
            rval += remove_if_qt(str(c)) + op
        rval = rval[:-3] + ')\n'
    elif isinstance(type, tuple) and type[0] == 'vot':
        rval += f'{remove_if_qt(name)} := @({type[1]}, ['
        for c in children:
            rval += remove_if_qt(str(c)) + ', '
        rval = rval[:-2] + "])\n"
    else:
        print(type)
    return rval


def print_be_aralia(name, failrate, time):
    return f'p({remove_if_qt(name)}) = {failrate_to_prob(time, failrate)}\n'


def print_dft_aralia(ftg, time, order=None):
    string = 'Translation\n\n'
    if order == None:
        order = ftg.bes
    for name, type in ftg.gates.items():
        children = ftg.graph[name]
        string += print_gate_aralia(name, type, children)
    string += '\n'
    for name in order:
        be = ftg.bes[name]
        string += print_be_aralia(name, be, time)
    return string


def print_gate_galileo(name, type, children):
    if isinstance(type, tuple) and type[0] == 'vot':
        type = f'{type[1]}of{len(children)}'
    string = f'{add_if_not_qt(name)} {type}'
    for c in children:
        string += ' ' + str(c)
    string += ';\n'
    return string


def print_be_galileo(name, failrate):
    return add_if_not_qt(name) + f' lambda={failrate} dorm=1;\n'


def print_dft_galileo(ftg, order=None):
    string = f'toplevel {add_if_not_qt(ftg.tle)};\n\n'
    if order == None:
        order = ftg.bes
    for name in order:
        be = ftg.bes[name]
        string += print_be_galileo(name, be)
    string += '\n'
    for name, type in ftg.gates.items():
        children = ftg.graph[name]
        string += print_gate_galileo(name, type, children)
    return string


###############################################################################
# lark_parser = lark.Lark(r'''
# start: (line | _NEWLINE)+
# line: (toplevel | gate | be) _EOL

# // Top level declaration
# toplevel: "toplevel" quoted_name ";"

# // Gate definitions
# gate: quoted_name gate_type quoted_name+ ";"
# gate_type: bool_op | voting_gate | inspection_gate
# bool_op: "and" | "or"
# voting_gate: INT "of" INT
# inspection_gate: INT "insp" INT

# // Basic event definitions
# be: quoted_name be_param+ ";"
# be_param: attr "=" (NUMBER | INT)
# attr: "lambda" | "phases" | "interval" | "dorm"

# // Name handling
# quoted_name: QUOTED_STRING
# QUOTED_STRING: /"[^"]+"/

# // Basic tokens
# %import common.INT
# %import common.NUMBER
# %import common.WS

# // Line handling
# _NEWLINE: /\r?\n/
# _EOL: ";"

# %ignore WS
# %ignore /\/\/.*/
# ''', parser="lalr")
lark_parser = lark.Lark(r'''
start: (line | _NEWLINE)+
line: (toplevel | gate | be) _EOL
toplevel: "toplevel" NAME
gate: NAME _type children
children: NAME+
_type: SIMPLEBOOL | vot | insp | DYNAMICTYPES  # Added 'insp' here
vot: INT "of" INT
insp: INT "insp" INT  # New rule for inspection gates
be: NAME assignment+
assignment: (ATTR "=" NUMBER)

ATTR: "lambda" | "prob" | "dorm" | "res" | "phases" | "interval"
SIMPLEBOOL: "and" | "or"
DYNAMICTYPES: "csp" | "wsp" | "hsp" | "pand" | "por" | "seq" | "mutex" | "fdep" | "pdep"

NAME: ("\"" CNAME "\"") | CNAME

%import common.CNAME
%import common.INT
%import common.NUMBER
COMMENT: "//" /[^\n]*/ _NEWLINE
_NEWLINE: "\n"
_EOL: ";\n"
%ignore COMMENT
%ignore " "
''', parser="lalr")

class GetGraph(lark.Visitor):
    def __init__(self):
        super(GetGraph,self).__init__()
        self.graph = dict()
        self.gates = dict()
        self.bes = dict()

    def gate(self,tree):
        children = [add_if_not_qt(str(i)) for i in tree.children[2].children]
        typ = str(tree.children[1])
        if isinstance(tree.children[1], lark.Tree) and tree.children[1].data == 'vot':
            typ = ('vot', int(tree.children[1].children[0].value))
        name = add_if_not_qt(str(tree.children[0]))
        self.graph[name] = children
        self.gates[name] = typ

    def be(self,tree):
        name = add_if_not_qt(str(tree.children[0]))
        l = float(list(tree.find_pred(lambda x: x.children[0] == 'lambda'))[0].children[1])
        self.bes[name] = l

    def toplevel(self,tree):
        self.tle = add_if_not_qt(str(tree.children[0]))

def parse_dft_sft(filename):
    dft_str = read_file(filename)
    res = lark_parser.parse(dft_str)

    f = GetGraph()
    f.visit(res)

    return f

###############################################################################

def get_children(event, graph):
    try:
        rval = graph[event]
    except KeyError:
        rval = []
    return rval


def recursive_trim_toplevel(current_event, graph):
    rval = dict()
    children = get_children(current_event, graph)

    for c in children:
        rval.update(recursive_trim_toplevel(c, graph))

    if len(children) > 0:
        rval[current_event] = children
    return rval

def recursive_trim_singulars(current_event, graph):
    rval = dict()
    children = get_children(current_event, graph)

    for c in children:
        rval.update(recursive_trim_singulars(c, graph))

    new_children = []
    restart = True
    while restart:
        restart = False
        for c in children:
            cc = get_children(c, rval)
            if len(cc) == 1:
                print(f"Removed {c} for {current_event}")
                restart = True
                new_children.append(cc[0])
            else:
                new_children.append(c)
        children, new_children = new_children, []

    if len(children) > 0:
        rval[current_event] = children
    return rval


def trim_graph(f):
    graph = recursive_trim_toplevel(f.tle, f.graph)
    #graph = recursive_trim_singulars(f.tle, graph)
    #graph = recursive_trim_toplevel(f.tle, graph)
    if graph == f.graph:
        return f
    else:
        print('More than one tle')

    tle = f.tle
    gates = dict()
    bes = dict()

    event_names = set(graph.keys()).union(set([x for li in graph.values() for x in li]))
    for g,c in f.gates.items():
        if g in event_names:
            gates[g] = c
    for b,p in f.bes.items():
        if b in event_names:
            bes[b] = p

    if gates != f.gates:
        print('Deleted some gates!')
    if bes != f.bes:
        print('Deleted some BEs!')

    return FTG(tle, graph, gates, bes)

###############################################################################

def write_xfta_scripts(model_name, basepath, f):
    write_xfta_vec(model_name, basepath, f)
    write_xfta_prob(model_name, basepath, f)
    write_xfta_mcs(model_name, basepath, f)
    write_xfta_importance(model_name, basepath, f)
    write_xfta_importance_vec(model_name, basepath, f)

def write_xfta_vec(model_name, basepath, f):
    tle = remove_if_qt(f.tle)
    str = ''
    str += f'load model "{model_name}" format=open-psa;\n'
    str += f'build target-model;\n'
    str += f'build BDD {tle};\n'
    str += f'compute probability {tle} mission-time=range(0,10,0.001) output="out.txt";\n'
    write_file(basepath + '/vec.xfta', str)

def write_xfta_prob(model_name, basepath, f):
    tle = remove_if_qt(f.tle)
    str = ''
    str += f'load model "{model_name}" format=open-psa;\n'
    str += f'build target-model;\n'
    str += f'build BDD {tle};\n'
    str += f'compute probability {tle} mission-time=1 output="out.txt";\n'
    write_file(basepath + '/prob.xfta', str)

def write_xfta_mcs(model_name, basepath, f):
    tle = remove_if_qt(f.tle)
    str = ''
    str += f'load model "{model_name}" format=open-psa;\n'
    str += f'build target-model;\n'
    str += f'build BDD {tle};\n'
    str += f'build ZBDD-from-BDD {tle};\n'
    str += f'print minimal-cutsets {tle} mission-time=1 output="out.txt";\n'
    write_file(basepath + '/mcs.xfta', str)

def write_xfta_importance(model_name, basepath, f):
    tle = remove_if_qt(f.tle)
    str = ''
    str += f'load model "{model_name}" format=open-psa;\n'
    str += f'build target-model;\n'
    str += f'build BDD {tle};\n'
    str += f'set option print-probability false;\n'
    str += f'set option print-conditional-probability-1 false;\n'
    str += f'set option print-conditional-probability-0 false;\n'
    str += f'set option print-marginal-importance-factor true;\n'
    str += f'set option print-critical-importance-factor false;\n'
    str += f'set option print-diagnostic-importance-factor false;\n'
    str += f'set option print-risk-achievement-worth false;\n'
    str += f'set option print-risk-reduction-worth false;\n'
    str += f'set option print-differential-importance-measure false;\n'
    str += f'set option print-Barlow-Proschan-factor false;\n'
    str += f'compute importance-measures {tle} mission-time=1 output="out.txt";\n'
    write_file(basepath + '/importance.xfta', str)

def write_xfta_importance_vec(model_name, basepath, f):
    tle = remove_if_qt(f.tle)
    str = ''
    str += f'load model "{model_name}" format=open-psa;\n'
    str += f'build target-model;\n'
    str += f'build BDD {tle};\n'
    str += f'set option print-probability false;\n'
    str += f'set option print-conditional-probability-1 false;\n'
    str += f'set option print-conditional-probability-0 false;\n'
    str += f'set option print-marginal-importance-factor true;\n'
    str += f'set option print-critical-importance-factor false;\n'
    str += f'set option print-diagnostic-importance-factor false;\n'
    str += f'set option print-risk-achievement-worth false;\n'
    str += f'set option print-risk-reduction-worth false;\n'
    str += f'set option print-differential-importance-measure false;\n'
    str += f'set option print-Barlow-Proschan-factor false;\n'
    str += f'compute importance-measures {tle} mission-time=range(0,10,0.01) output="out.txt";\n'
    write_file(basepath + '/importance_vec.xfta', str)

