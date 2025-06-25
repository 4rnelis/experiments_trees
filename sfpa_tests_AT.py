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

import time
from func_timeout import func_timeout, FunctionTimedOut
from sfpa import unreliability
from translate import parse_dft_sft

afdruk = open("logs/sfpa_AT.csv","w")

for i in range(128,256):
    address = 'almost_trees/at_'+str(i)+'.dft'
    G = parse_dft_sft(address)
    now = time.time()
    try:
        p1 = func_timeout(60,unreliability,args=(G,False));
        t = time.time()-now
    except FunctionTimedOut:
        t = 60
    now = time.time()
    #try:
    #    p1 = func_timeout(60,unreliability2,args=(G,False));
    #    t2 = time.time()-now
    #except FunctionTimedOut:
    #    t2 = 60
    #draadje = str(i)+','+str(t)+','+str(t2)
    draadje = str(i)+','+str(t);
    afdruk.write(draadje+"\n");
    print(draadje)
afdruk.close();


afdruk = open("logs/sfpa_AT_full.csv","w")

for i in range(128,256):
    address = 'almost_trees/at_'+str(i)+'.dft'
    G = parse_dft_sft(address)
    now = time.time()
    try:
        p1 = func_timeout(60,unreliability,args=(G,True));
        t = time.time()-now
    except FunctionTimedOut:
        t = 60
    now = time.time()
    #try:
    #    p1 = func_timeout(60,unreliability2,args=(G,False));
    #    t2 = time.time()-now
    #except FunctionTimedOut:
    #    t2 = 60
    #draadje = str(i)+','+str(t)+','+str(t2)
    draadje = str(i)+','+str(t);
    afdruk.write(draadje+"\n");
    print(draadje)
afdruk.close();
