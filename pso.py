# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.cluster import KMeans

import plotter
from functions import *

class Particle:
    position = None
    velocity = None
    

    l_best_pos = None
    l_best_err = None

    def __init__(self, init_range):
   
        self.position = np.random.uniform(low= -init_range, high=init_range, size=(2,))
        self.velocity = np.array([0., 0.])

    def get_l_best_pos(self):
        return l_best_pos

    def get_l_best_err(self):
        return l_best_err

    def set_l_best_pos(self):
        self.l_best_pos = self.position

    def set_l_best_err(self, error):
        self.l_best_err = error

    def update(self, g_best_pos, w, c1, c2):
        g_best_pos = np.array(g_best_pos)
        r1 = np.random.rand()
        r2 = np.random.rand()
       
        next_velocity = w * self.velocity + c1 * r1 * (self.l_best_pos - self.position) + \
                        c2 * r2 * (g_best_pos - self.position) 
        self.position = self.position + next_velocity
        

class PSO:
    PSO_list = []
    error_func = None
    c1 = None
    c2 = None
    init_w = None
    w_decay_iters = None
    w_decay_weight = None
    w_decay_gamma = None
    
    #kmeans
    p_cluster_labels = None

    def __init__(self, init_range=2048, p_num=100, error_func=rosenbrock, c1=2, c2=2, init_w=100, w_decay_iters=20, w_decay_weight=25):
        """
            init_range     :   the range that particle will spawn in [-init_range, init_range].
            p_num          :   the number of paricle
            error_func     :   the error function
            c1             :   parameters for updating position
            c2             :   parameters for updating position
            init_w         :   parameters for updating position
            w_decay_iters  :   the iterations that w will start to decay.
            w_decay_weight :   exp(-w_decay_weight * x)
            
        """
        self.PSO_list = []
        self.error_func = error_func
        for i in range(p_num):
            self.PSO_list.append(Particle(init_range))

        self.c1 = c1
        self.c2 = c2
        self.init_w = init_w

        self.w_decay_iters = w_decay_iters
        self.w_decay_weight =w_decay_weight

        #plotter.plot3d_init()
        #plotter.plot_func(self.error_func)

    def get_global_best(self):
        """
            return the position of the lowest error.
        """
        best_err_list = []
        best_pos_list = []
        for p in self.PSO_list:
            best_err_list.append(p.l_best_err)
            best_pos_list.append(p.l_best_pos)
        
        g_best_err = min(best_err_list)
        g_best_pos = best_pos_list[best_err_list.index(g_best_err)]
        
        del best_err_list
        del best_pos_list

        return g_best_pos, g_best_err 

    def update(self, iterations, decay=True, plot=False):
        if (iterations > self.w_decay_iters ) and decay:
            w = self.init_w * np.exp( - self.w_decay_weight * (iterations  - self.w_decay_iters))
        else:
            w = self.init_w

        for p in self.PSO_list:
            curr_error = self.error_func(p.position[0], p.position[1])

            if plot:
                #plotter.plot_3d_dot(p.position[0], p.position[1], curr_error, c='y', marker='o')
                #plotter.plot_show(block=False)
                pass
            if (type(p.l_best_err) == type(None) or type(p.l_best_pos) == type(None)) or curr_error < p.l_best_err:
                # initialization or error  lower than the current error
                p.set_l_best_err(curr_error)
                p.set_l_best_pos()

        g_best_pos, g_best_err = self.get_global_best()

        for p in self.PSO_list:
            p.update(g_best_pos, w, self.c1, self.c2)

        
    def run(self, iterations, is_kmeans=False, n_clusters=3, **args):
   
        for i in range(iterations):
            if is_kmeans and ((i % 20 == 0) and i != 0):
                
                position_list = []
                for p in self.PSO_list:
                    position_list.append(p.position)
                
                position_list = np.array(position_list)
                
                sk_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.array(position_list))
                
                #k_clusters_center = sk_kmeans.cluster_centers_
                labels = sk_kmeans.predict(position_list)
                
                    
                if self.p_cluster_labels is None: 
                    self.p_cluster_labels = labels
                elif not (self.p_cluster_labels - labels).all():
                    self.p_cluster_labels = labels
                else:
                    # kmeans stable
                    self.w_decay_iters = i
                    is_kmeans = False

                #plotter.plot_3d_dot(position_list[:,0], position_list[:,1], zs=0, marker='^')
                #plotter.plot_show()
                
                
            self.update(i, **args)
        #plotter.plot_show()

if __name__ == "__main__":
    
    iterations_for_testing = 1
    
    best_err_list = []
    for i in range(iterations_for_testing):
        pso = PSO(init_range=2048, p_num=30, error_func=rosenbrock, c1=2, c2=2, init_w=1000, w_decay_iters=60, w_decay_weight=5)
        iterations = 300
        pso.run(iterations, is_kmeans=True, decay=True, plot=True)
        g_best_pos, g_best_err = pso.get_global_best()
        #print("times: "+str(i) + " ", l_best_pos, rosenbrock(l_best_pos[0], l_best_pos[1]))
        best_err_list.append(g_best_err)

    mean = np.mean(np.array(best_err_list))
    std = np.std(np.array(best_err_list))

    print(" mean ± std: ", round(mean, 3),  " ± ", round(std, 5))

