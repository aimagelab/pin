MIN_OBS_GOAL_OCCUPATION = 0.05
MAX_GOAL_DEPTH = 5.0

SEMANTIC_GOAL_ID = 100000
SEMANTIC_CATEGORY_DISTRACTOR_ID = 100001
SEMANTIC_GENERIC_DISTRACTOR_ID = 100002

def get_goal_seen_metrics(obs, goal_seen, goal_seen_cond):
    if SEMANTIC_GOAL_ID in obs['semantic']:
        # print("Goal seen")
        goal_seen = 1.0
        
    goal_locations = (obs['semantic'] == SEMANTIC_GOAL_ID)
    goal_occupancy = goal_locations.sum()
    if goal_occupancy > 0:
        goal_depth = obs['depth'][goal_locations[:,:,0]].mean()
        if (goal_occupancy > (MIN_OBS_GOAL_OCCUPATION * obs['semantic'].shape[0] * obs['semantic'].shape[1])) and (goal_depth < MAX_GOAL_DEPTH):
            # print("Goal seen with conditions")
            goal_seen_cond = 1.0
            
    return goal_seen, goal_seen_cond

def get_found_goal_metrics(obs, matched, agent_found_goal, found_goal, correct_matches_list, category_matches_list, generic_distractor_matches_list, other_matches_list,
                           num_matched):
    if agent_found_goal:
        found_goal = 1.0
        
    if matched:
        if SEMANTIC_GOAL_ID in obs['semantic']:
            correct_matches_list.append(1.0)
        elif SEMANTIC_CATEGORY_DISTRACTOR_ID in obs['semantic']:
            category_matches_list.append(1.0)
        elif SEMANTIC_GENERIC_DISTRACTOR_ID in obs['semantic']:
            generic_distractor_matches_list.append(1.0)
        else:
            other_matches_list.append(1.0)
        num_matched += 1
        
    return found_goal, correct_matches_list, category_matches_list, generic_distractor_matches_list, other_matches_list, num_matched
    