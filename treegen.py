#!/usr/bin/env python

import fault_tree_generator as ftg
import random
import translate
import time
import os

# def generate_small_tree():
#     args = {
#         'ft_name': 'Small_Fault_Tree',
#         'root': 'TOP',
#         'seed': 42,
#         'num_basic': random.randint(10, 30),  # Random size 10-30
#         'num_args': round(random.uniform(2.0, 3.0), 1),
#         'weights_g': [1, 1, 1, 0, 0],
#         'common_b': 0.1,
#         'common_g': 0.1,
#         'parents_b': 2,
#         'parents_g': 2,
#         'min_prob': 0.01,
#         'max_prob': 0.1,
#         'num_house': random.randint(0, 3),
#         'num_ccf': 0
#     }
    
#     factors = ftg.Factors()
#     factors.set_min_max_prob(args['min_prob'], args['max_prob'])
#     factors.set_common_event_factors(args['common_b'], args['common_g'],
#                                    args['parents_b'], args['parents_g'])
#     factors.set_num_factors(args['num_args'], args['num_basic'],
#                           args['num_house'], args['num_ccf'])
#     factors.set_gate_weights(args['weights_g'])
#     factors.calculate()
    
#     return ftg.generate_fault_tree(args['ft_name'], args['root'], factors)

# if __name__ == "main":
#     pritn
#     print(generate_small_tree())



for i in range(1, 10):
    factors = ftg.Factors()
    factors.set_min_max_prob(0,1)
    # parents = random.randrange(2, 6)
    # children = random.randrange(2, 6)
    # factors.set_common_event_factors(0.01,0.15,parents,children)
    factors.set_common_event_factors(0.01,0.15,5,5)

    factors.set_num_factors(num_args=2.5,num_basic=400,num_house=0,num_ccf=0)
    factors.set_gate_weights([1,1,1])
    factors.calculate()
    FT = ftg.generate_fault_tree('test','root',factors)
    afdruk = open("temp.xml","w")
    FT.to_xml(afdruk)
    afdruk.close()
    FT = translate.parse_xml_sft("temp.xml",1)
    afdruk = open("visual_testing/at_"+str(i)+".dft","w")
    afdruk.write(translate.print_dft_galileo(FT))
    afdruk.close()
    
os.remove("temp.xml")
