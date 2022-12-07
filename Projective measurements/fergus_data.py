import numpy as np
import time
import pickle
import os
import matplotlib.pyplot as pl
from sep_decoder import transfer,ht,lp,load_transfer_matrices

def sep_dynamics(data,Q):
    (depth,L) = data.shape  
    Q2 = Q+1
    if Q>L//2:
        Q2 = Q-1
    hash_table,_ = ht.get_hash_table(L,Q)
    hash_table2,_= ht.get_hash_table(L,Q2)
    pairings = lp.local_pairings(L,Q,hash_table)
    pairings2 = lp.local_pairings(L,Q2,hash_table2)
    dim = len(hash_table)
    dim2 = len(hash_table2)
    
    total = 0

    transfer_matrices = load_transfer_matrices(pairings,dim,L,Q)
    transfer_matrices2 = load_transfer_matrices(pairings2,dim2,L,Q2)
    
    initial_state_Q = np.eye(dim)/dim**0.5
    initial_state_Q2 = np.eye(dim2)/dim2**0.5

    N=1
    traj = data
    state_Q_is_zero = False
    state_Q2_is_zero = False
    state_Q = initial_state_Q.copy()
    state_Q2 = initial_state_Q2.copy()

    log_Z = []
    log_Z2 = []
    total += N
    for t in range(depth)[:]:
        
        if t%2 == 0: #even layer
            for x in range(0,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
            
                state_Q, state_Q_is_zero = transfer(x,dim,pairings,transfer_matrices, outcomes,state_Q,log_Z,state_Q_is_zero,L,Q,debug=False)

                state_Q2, state_Q2_is_zero = transfer(x,dim2,pairings2,transfer_matrices2 ,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,Q2)
                
                if state_Q_is_zero and state_Q2_is_zero:
                    print('hurray2',np.sum(state_Q2),np.sum(state_Q))
        else:
            for x in range(1,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
                debug=False

                state_Q, state_Q_is_zero = transfer(x,dim,pairings,transfer_matrices, outcomes,state_Q,log_Z,state_Q_is_zero,L,Q,debug=False)

                state_Q2, state_Q2_is_zero = transfer(x,dim2,pairings2,transfer_matrices2 ,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,Q2)

                if state_Q_is_zero and state_Q2_is_zero:
                    print('hurray2',np.sum(state_Q2),np.sum(state_Q))
                

        # print(t,time.time()-start,total)
        if state_Q_is_zero:              
            break
    
        if state_Q2_is_zero:
            break
        
    if state_Q_is_zero:
        p_Q = 0
    elif state_Q2_is_zero:
        p_Q = 1
    else:
        ratio_p = np.exp(np.sum(log_Z) - np.sum(log_Z2))
        p_Q = 1/(1+1/ratio_p)
        p_Q2 = 1/(1+ratio_p)
    
    return p_Q

    
def get_p_L(filename):
    temp = filename.split(',')
    p = float(temp[-1][:-4])
    L = int(temp[1])
    return p,L

filedir = 'U_1_data_Fergus/useful/'
p_set = set()
for file in os.listdir(filedir)[:]:
    temp = file.split(',')
    p,L = get_p_L(file)
    p_set.add(p)
p_list = list(sorted(p_set))[:]
print(p_list,len(p_list),sorted(p_set))

def get_sep_data(N_samples,L,p):
    p_suc = {}
    for file in os.listdir(filedir)[:]:
        p_data,L_data = get_p_L(file)
        if p_data != p or L_data != L:
            continue
        data_raw = np.load(filedir+file)[:N_samples]
        for i in [0,1]:
            Q = int(np.sum(0.5*(data_raw[0,i,:,-1]+1)))
            p_suc[Q] = []
            for j in range(N_samples):
                p_suc[Q].append(sep_dynamics(data_raw[j,i,:,:-1].T,Q))
    return p_suc


L_list = [6,8,10,12]
p_suc_dic = {}
N_samples = 1000
file_destination = 'U_1_data_Fergus/sep_data'
for L in L_list:
    p_suc_dic[L]={}
    for p in p_list[:]:
        start = time.time()
        p_suc_dic[L][p]=get_sep_data(10,L,p)
        print(L,p,time.time()-start)
        with open(file_destination,'wb') as f:
            pickle.dump(p_suc_dic,f)

def plot_success_ent(data,L_list,p_list,N_samples=-1):
    ent = {}
    err = {}
    for L in L_list[:]:
        ent[L] = []
        err[L] = []
        for p in p_list:
            tempQ = list(data[L][p][L//2])[:N_samples]
            tempQ2 = list(data[L][p][L//2+1])[:N_samples]
            print("L=",L,"p=",p,"data_size:",len(tempQ),len(tempQ2))
            # ent.append(np.sum(np.array(suc_list)>0.5)/len(suc_list))
            ent_list = [(-x*np.log(x) - (1-x)*np.log(1-x)) if 0<x<1 else 0 for x in tempQ2+tempQ]
            ent[L].append(np.average(ent_list))
            err[L].append(np.std(ent_list)/(len(ent_list)-1)**0.5)

        pl.errorbar(p_list,ent[L],yerr=err[L],ls='-',marker='o',label='L='+str(L))

    pl.xlabel(r'$p$',fontsize=16)
    pl.ylabel(r'$S_{suc}$',fontsize=16)
    pl.legend(fontsize=16)
    pl.tight_layout()
# plot_success_ent(p_suc_dic,L_list[:-1],p_list[:-1])
