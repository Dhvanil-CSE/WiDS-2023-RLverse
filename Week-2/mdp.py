import numpy as np
import math
fname="mdp-10-5.txt"
mdp_dict={'transitions':{},'gamma': None}
with open("mdp-10-5.txt","r") as file:
    for line in file:
        tokens=line.split()
        if tokens[0]=='states':
            mdp_dict["n_state"]=int(tokens[1])
        elif tokens[0]=="actions":
            mdp_dict["n_actions"]=int(tokens[1])
        elif tokens[0] == 'tran':
            state = int(tokens[1])
            action = int(tokens[2])
            next_state = int(tokens[3])
            reward = float(tokens[4])
            probability = float(tokens[5])
            if state not in mdp_dict['transitions']:
                mdp_dict['transitions'][state] = {}
            if action not in mdp_dict['transitions'][state]:
                mdp_dict['transitions'][state][action] = []
            mdp_dict['transitions'][state][action].append({'fin_state': next_state,'reward': reward, 'probability': probability})
        elif tokens[0] == 'gamma':
            mdp_dict['gamma'] = float(tokens[1])

# print(mdp_dict)

# Prev_v=np.zeros(mdp_dict['n_state'])
V=np.zeros(mdp_dict['n_state'], dtype=np.float64)
A=np.zeros(mdp_dict['n_state'], dtype=np.float64)

theta=math.e**-20
# print(theta)

while True:
    Q=np.zeros((mdp_dict['n_state'],mdp_dict['n_actions']))
    for s in mdp_dict['transitions']:
        # vk=0
    # print('S :',s)
        for act in mdp_dict['transitions'][s]:
        # print('Act :',act)
            for poss in range(len(mdp_dict['transitions'][s][act])):
                Q[s][act]+=float(mdp_dict['transitions'][s][act][poss]['probability']*(mdp_dict['transitions'][s][act][poss]['reward']+mdp_dict['gamma']*V[mdp_dict['transitions'][s][act][poss]['fin_state']]))
                # vk+=float(mdp_dict['transitions'][s][act][poss]['probability']*(mdp_dict['transitions'][s][act][poss]['reward']+mdp_dict['gamma']*Prev_v[mdp_dict['transitions'][s][act][poss]['fin_state']]))
    if np.max(np.abs(V-np.max(Q,axis=1)))<=theta:
        V=np.max(Q,axis=1)
        A=np.argmax(Q,axis=1)
        break
    V=np.max(Q,axis=1)
    A=np.argmax(Q,axis=1)
    # print(V)
    # print(A)

file.close()
fname=f"sol-{fname}"
with open(fname,'w') as f:
    for i in range(len(V)):
        f.write(f"{V[i]} {A[i]}\n")
f.close()

