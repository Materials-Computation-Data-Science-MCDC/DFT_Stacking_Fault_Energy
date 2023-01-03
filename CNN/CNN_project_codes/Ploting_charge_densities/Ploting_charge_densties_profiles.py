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
    
    fig,ax = plt.subplots(1,1,figsize = (10,3.75))
    style  = 'gist_rainbow'
    
    #define the intensity
    #min_intensity, max_intensity = 0.2, 0.65
    #max_intensities = np.linspace(min_intensity,max_intensity,num = 40)
    max_intensities = [0.65]
    
    for i in range(len(data)): # as of now it is just 1
        for max_intensity in max_intensities:
            density = np.array(data[i].iloc[320:,:])
            density[np.where(density > max_intensity)] = max_intensity

            figure_ = ax.contourf(data[i].iloc[0:160,:], data[i].iloc[160:320,:], 
                                  density, 200, cmap = style)

            plt.axis('off')
            plt.tight_layout()
            plt.savefig('contour_plots/' + str(str_id) + '-layer-124.png', format = "png")
            plt.close()
            #plt.savefig('testing/layer-1.png', format = "png")
            del figure_
            #print(i)
    #if i == (len(data)-1):
        #print('All plots are made')
    #end_time = time.time()
    #total_time = end_time - start_time
    #print("Total time in sec for making profiles = {:.2f}".format(total_time))
    return()
