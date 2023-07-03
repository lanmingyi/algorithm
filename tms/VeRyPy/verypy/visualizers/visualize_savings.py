# -*- coding: utf-8 -*-

import re
from tms.VeRyPy.verypy.visualizers.shared_visualize import visualize_cli, visualize_procedure, VISUALIZE
from tms.VeRyPy.verypy.util import sol2routes

MAKE_ANIM = True
SHOW_INFEASIBLE = True

popped_re = re.compile("Popped savings s_\{([0-9]+),([0-9]+)\}")
merged_re = re.compile("Merged, resulting solution is (\[.*?\])")

to_restore_routes=[]
def _process_savings_debug_line(line, normalization_parameters, currentK,
                       #output here
                       rays, active_nodes, active_ray_idxs,
                       points_of_interest,
                       candidate_routes, infeasible_routes, complete_routes,
                       labels):
    
    global to_restore_routes
    
    newK = None
    changed = False
        
    mmo = merged_re.search(line)    
    if mmo:
        infeasible_routes[:]=[]
        complete_routes[:] = sol2routes(eval(mmo.group(1)))
        to_restore_routes = list(complete_routes)
        changed = True

    pmo = popped_re.search(line)    
    if pmo:
        complete_routes[:] = to_restore_routes
        active_nodes[:] = [int(pmo.group(1)), int(pmo.group(2))]
        changed = False
        
    if SHOW_INFEASIBLE and "Reject merge due" in line:
        left_merge_node, right_merge_node = active_nodes[0],active_nodes[1]
        left_route_idx, right_route_idx = None,None
        for ri, r in enumerate(complete_routes):
            if left_merge_node in r:
                left_route_idx = ri
            if right_merge_node in r:
                right_route_idx = ri
        
        # make the merged route
        
        if complete_routes[left_route_idx][1]==left_merge_node and\
           complete_routes[right_route_idx][1]==right_merge_node:
            infeasible_routes[:]=[ complete_routes[left_route_idx][:0:-1]+
                                   complete_routes[right_route_idx][1:] ]
        elif complete_routes[left_route_idx][1]==left_merge_node and\
             complete_routes[right_route_idx][-2]==right_merge_node:
            infeasible_routes[:]=[ complete_routes[left_route_idx][:0:-1]+
                                   complete_routes[right_route_idx][-2::-1] ]
        elif complete_routes[left_route_idx][-2]==left_merge_node and\
             complete_routes[right_route_idx][1]==right_merge_node:
            infeasible_routes[:]=[ complete_routes[left_route_idx][:-1]+
                                   complete_routes[right_route_idx][1:] ]
        elif complete_routes[left_route_idx][-2]==left_merge_node and\
             complete_routes[right_route_idx][-2]==right_merge_node:
            infeasible_routes[:]=[ complete_routes[left_route_idx][:-1]+
                                   complete_routes[right_route_idx][-2::-1] ]
        complete_routes[left_route_idx] = []
        complete_routes[right_route_idx] = []
        
        #print "REMOVEME", left_route_idx, right_route_idx, infeasible_routes
        
        changed = True
            
    return changed, newK
    
if __name__=="__main__":
    algo_output, problem_name, keep_files = visualize_cli("parallel_savings")
    visualize_procedure(algo_output, "parallel_savings", problem_name, selector=VISUALIZE.ALL,
              make_anim=MAKE_ANIM, keep_files=keep_files,
              process_debug_line_callback = _process_savings_debug_line)