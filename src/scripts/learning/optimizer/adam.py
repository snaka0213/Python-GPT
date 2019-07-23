#!/usr/bin/env python3
import numpy as np
import settings

### Adam: https://arxiv.org/pdf/1412.6980.pdf ###
'''
    
    def learn(self):
        moment_1, moment_2 = np.zeros(self.M), np.zeros(self.M) # momentum in Adam
        epsilon_hat = epsilon*np.sqrt(1-beta_2) # constant in Adam
        
        graph, feature_vector_list = self._graph, self._feature_vector_list
        N = len(feature_vector_list) # the size of data_set
        for epoch in range(Epoch):
            # an epoch
            index_list = list(range(N))
            for step in range(N//batch_size):
                samples_index = random.sample(index_list, batch_size)
                for i in samples_index:
                    index_list.remove(i)
                
                grad = (1/batch_size)*gradient(graph, feature_vector_list, samples_index, self.normal)
                self.normal += eta*grad
        
    '''
