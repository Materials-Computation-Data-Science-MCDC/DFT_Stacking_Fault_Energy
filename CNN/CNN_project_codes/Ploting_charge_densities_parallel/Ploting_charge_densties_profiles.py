#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:28:17 2022

@author: gauravarora
"""

#This function take two input from the code which is to used previously for extracting data from CHGCAR. Information about the charge denisty in the form of array and name of the file is uded as input for runnning this code. This code reads the charge denisty information and plot contour plot in parallel because making contour plots takes huge amount of time if ran on just one core.

## Number of cores should be decided based on the number of planes or the lenfth of the data being input to the code. The number of cores should be completely divisible by the length of the data or the number of the planes.

### This code will divide the data equally on the bnumber of cores and then print the name of the processes or the name of the cores. Generated images should be renamed according to the processes name in increasing order. 


def ploting_charge_densities(main_data, name_of_the_file):
    print(f'######## Making contour plots for {name_of_the_file} file ######')
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Input the number of cores you want to distribute the task on. Make sure that the number of cores should be completely divisble by the length of the main_data.It can be user input or pre-defined as convenient.
    ## Number of cores = num_of_divison
    
    #num_of_division = int(input('Enter the number of cores. Make sure to have number of cores completely divisible by the number of planes in data'))
    num_of_division = 32
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Importing the libraries

    import os
    import sys
    import time
    import numpy as np
    import multiprocessing
    import matplotlib.pyplot as plt
    from multiprocessing import Process, current_process
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # starting the timer 
    start_time = time.time()
    
    # Defining the name for the figure and axis, including the size of the image
    fig,ax = plt.subplots(1,1,figsize = (10,3.75))
    
    # Defining the style of the contour plot
    style  = 'gist_rainbow'
    
    # Checking if the number of cores is compatible with the length of the data by dividing and getting the remainder.
    if len(main_data) % num_of_division == 0:
        pass
    else:
        print('Number of cores or number of divisions are not completely divisibel by the length of the data or the number of the plnanes for which contour plots are to be made')
        sys.exit()
    
    # Size of the chunk of the data to be distributed on each core. Number of images or contour plots will be equal to the size of the chunk
    size_of_chunk = int(len(main_data)/num_of_division)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Dividing the data into chunks to be distrbuted on the number of cores
    ## Defining the name of the array in which data would be stored. Size of this array would be equal to the number of cores given.
    chunks = []
    for q in range(num_of_division):
        init = size_of_chunk * q
        final = size_of_chunk * (q+1)
        var = main_data[init:final]
        chunks.append(var)
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Saving the contour plots in the directory
    ## Checking if directory exist then stop the code and ask the user to remove the old directory or rename the old directory 
    #if os.path.isdir(f'Contour_plots/{name_of_the_file}'):
    #    print('Directory exist, please rename the old directory or delete it')
    #    sys.exit()
    #else:
    #    pass
    
    ## Making the directory with name Contour plots and sub-directory with name as that of CHGCAR file 
    os.makedirs(f'Contour_plots/{name_of_the_file}')
    
    ## Defining the intensity or the scale using which the plots should be made.
    ### Change this variable in order to highlight diffferent aspect of charge density plots (max_intensity)
    max_intensity = 0.65
    
    ## Defining the length or the position of the charge density data in the main data. It is appended at the last in array while generating the data with X and Y positions at the top
    size_of_data = int(len(main_data[0]) / 3)
    
    first_slice = int((size_of_data) * 1)
    second_slice = int((size_of_data) * 2)
    third_slice = int((size_of_data) * 3)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Defining the function for making contour plots
    #
    def making_images(sub_data):
        for i in range (len(sub_data)):
                        
            # Uncomment the next line if charge denisty plots needs to be plotted with pre-defined intensity given by variable name max_intensity
            density[np.where(density > max_intensity)] = max_intensity
            
            # Ploting X_array, Y_array, and corresponding denisty
            X_array = np.array(sub_data[i].iloc[0:first_slice,:])
            Y_array = np.array(sub_data[i].iloc[first_slice:second_slice,:])
            density = np.array(sub_data[i].iloc[second_slice:third_slice,:])
            ax.contourf(X_array, Y_array, density, 200, cmap = style)

            #Switching off the axis because we are making images 
            plt.axis('off')
            plt.tight_layout()
            
            #This is the command which would be used while calling this function to run in parallel. The name of the processes will be used to generate images in an order
            process_name = p.name
            
            # Saving the figure in png format with name of the CHGCAR file and process name and layer number for that specific chunk of data running on the core
            plt.savefig(f'Contour_plots/{name_of_the_file}/{process_name}-layer-' + str(i) + '.png', format = "png", bbox_inches='tight')
            
        return()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Making the code run in parallel
    
    print('#### Printing the processes names for multiprocesses in order they were initiated #####')
    
    #Initailizing the list of appending various processes to be run in parallel
    processes = []       
    for ii in range(len(chunks)):
        # Calling the function defined above for making images and argument used is the chunks list we defined or made earlier in the code
        p = multiprocessing.Process(target=making_images, args=[chunks[ii]])
        print(p.name)
        p.start()
        processes.append(p)
        
    for process in processes:
        process.join()
                                                                  
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time in sec for making profiles = {:.2f}".format(total_time))
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    return()
