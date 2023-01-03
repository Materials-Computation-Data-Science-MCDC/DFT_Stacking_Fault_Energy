#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:28:17 2022

@author: gauravarora
"""

#This function take input from the code which genrates charge density arrays including a_array and y_array for position
#and then read from variable in which the information is stored and then make profile for each plane



def ploting_charge_densities(data, str_id):
    import os
    import numpy as np    
    import matplotlib.pyplot as plt
    from matplotlib.colors import from_levels_and_colors
    
    levels = np.linspace(0, 0.7, num = 30)
    
    nvals = len(levels) -1
    #colors = np.random.random((nvals, 3))
    
    np.random.seed(30)
    colors = np.random.random_sample(size=(nvals,3))
    
    cmap, norm = from_levels_and_colors(levels, colors)
    
    def NormalizeData(data):
        return ((data - np.min(data)) / (np.max(data) - np.min(data)))*3
    
    fig,ax = plt.subplots(1,1,figsize = (10,3.75))
    style  = 'tab20'
    
    
    #define the intensity
    #min_intensity, max_intensity = 0.2, 0.65
    #max_intensities = np.linspace(min_intensity,max_intensity,num = 40)
    max_intensities = [0.06]
    
    for i in range(len(data)): # as of now it is just 1
        for j in max_intensities:
            density = np.array(data[i].iloc[320:,:])
            
            #density = NormalizeData(density)
            
            max_density = np.max(density)
            min_density = np.min(density)
            
            #print(str(max_density) + ' ' + str(min_density))
            
            #density = density - min_density
            
            #density = density + 0.064
            #print(j)
            #density[np.where(density > 0.5)] = 0.5

            figure_ = ax.contourf(data[i].iloc[0:160,:], data[i].iloc[160:320,:], 
                                  density, cmap = cmap, norm = norm)
            plt.colorbar(figure_)

            plt.axis('off')
            plt.tight_layout()
            plt.savefig('contour_plots/' + str(str_id) + '-layer-124_user.png', format = "png")
            plt.close()
            #plt.savefig('testing/layer-1.png', format = "png")
            del figure_
            #print(i)
    #if i == (len(data)-1):
        #print('All plots are made')
    #end_time = time.time()
    #total_time = end_time - start_time
    #print("Total time in sec for making profiles = {:.2f}".format(total_time))
    return(min_density)

