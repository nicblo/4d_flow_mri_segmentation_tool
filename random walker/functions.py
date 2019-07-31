# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:27:23 2019

@author: nblon
"""
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tkinter import *
from PIL import Image, ImageTk
from skimage._shared.utils import warn
from mayavi.mlab import *
from bokeh.layouts import row, column, layout
from bokeh.models import Slider, Button, LinearColorMapper
from bokeh.models.annotations import ColorBar
from bokeh.models.widgets import RadioButtonGroup, Tabs, Panel
from bokeh.models.widgets.markups import Div
from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import curdoc
from scipy.ndimage import gaussian_filter
import random_walker_nb_3D as rw
from scipy.io import loadmat
import scipy.ndimage.morphology as morph
import skimage.morphology as mp
import os

def load_parec_data(subject = 'AH'):
    #loads the parec data of the selected subject and returns an array with the arrays for M, P, S, #timesteps, #slices in dataset
    cwd = os.getcwd()
    #choose subject, default if no argument passed = AH
    
    if subject == 'AH':
        basepath = cwd + '/../ibt_4dFlow/AH/'
        path_s = basepath + 'an_27052015_1027340_4_2_wipqflow_fbclearV4_S.rec'
        path_m = basepath + 'an_27052015_1027340_4_2_wipqflow_fbclearV4_M.rec'
        path_p = basepath + 'an_27052015_1027340_4_2_wipqflow_fbclearV4_P.rec'
        
    elif subject == 'CB':
        basepath = cwd + '/../ibt_4dFlow/CB/'
        path_s = basepath + 'ch_11122015_1428290_4_2_wipqflow_fb_experiment1V4_S.rec'
        path_m = basepath + 'ch_11122015_1428290_4_2_wipqflow_fb_experiment1V4_M.rec'
        path_p = basepath + 'ch_11122015_1428290_4_2_wipqflow_fb_experiment1V4_P.rec'
        
    elif subject == 'DG':
        basepath = cwd + '/../ibt_4dFlow/DG/'
        path_s = basepath + 'da_15072015_1612350_3_2_wipqflow_fbclearV4_S.rec'
        path_m = basepath + 'da_15072015_1612350_3_2_wipqflow_fbclearV4_M.rec'
        path_p = basepath + 'da_15072015_1612350_3_2_wipqflow_fbclearV4_P.rec'
        
    elif subject == 'JR':
        basepath = cwd + '/../ibt_4dFlow/JR/'
        path_s = basepath + 'ju_27052015_1208240_5_1_wipqflow_fbclearV42.rec'
        path_m = basepath + 'ju_27052015_1142050_4_2_wipqflow_fbclearV4_M.rec'
        path_p = basepath + 'ju_27052015_1142050_4_2_wipqflow_fbclearV4_P.rec'
        
    elif subject == 'LT':
        basepath = cwd + '/../ibt_4dFlow/LT/'
        path_s = basepath + 'lo_27112015_1256300_2_2_wipqflow_fb_experiment1V4_S.rec'
        path_m = basepath + 'lo_27112015_1256300_2_2_wipqflow_fb_experiment1V4_M.rec'
        path_p = basepath + 'lo_27112015_1256300_2_2_wipqflow_fb_experiment1V4_P.rec'
        
        #if wrong input is given, break function and return the warning in console
    else:
        warn('Invalid subject, valid subjects are: AH,CB,DG,JR and LT')
    
    #load the data into arrays
    data_s = nib.parrec.load(path_s).get_data()
    data_m = nib.parrec.load(path_m).get_data()
    data_p = nib.parrec.load(path_p).get_data()
    
    #calculate the numer of timesteps and slices, note that the timesteps have to be divided by 2 as we have a magnitude and phase image for each time step
    num_times = int(data_s.shape[3]/2)
    num_slices = int(data_s.shape[2])
    
    #return the desired vector of the laoded data
    parec_data = [data_m,data_p,data_s,num_times,num_slices]
    
    return parec_data

def create_separated_arrays(parec_data):
    data_m = parec_data[0]
    data_p = parec_data [1]
    data_s = parec_data[2]
    num_times = parec_data[3]
    num_slices = parec_data[4]
    
    
    #separate the velocity components out of the vector with velocity and proton density image
    vs_vec = np.zeros((int(data_s.shape[0]),int(data_s.shape[1]),num_slices,num_times))
    vm_vec = np.zeros((int(data_m.shape[0]),int(data_m.shape[1]),num_slices,num_times))
    vp_vec = np.zeros((int(data_p.shape[0]),int(data_p.shape[1]),num_slices,num_times))
    m_vec = np.zeros((int(data_p.shape[0]),int(data_p.shape[1]),num_slices,num_times))
    
    
    for t in range(num_times):
        vs_vec[:,:,:,t] = data_s[:,:,:,t*2+1]
        vm_vec[:,:,:,t] = data_m[:,:,:,t*2+1]
        vp_vec[:,:,:,t] = data_p[:,:,:,t*2+1]
        m_vec[:,:,:,t] = data_p[:,:,:,t*2]
        
    separated_arrays = [m_vec,vm_vec,vp_vec,vs_vec]
    
    return separated_arrays

def load_gt_data(subject = 'AH'):
    cwd = os.getcwd()
    
    if subject == 'AH' or 'CB' or 'DG' or 'JR' or 'LT':
        basepath = cwd+'/../ibt_4dFlow/3Dlabels/'+subject+'/'
        mat = loadmat(basepath + 'mask.mat')
        mask_array = mat['mask']
    else:
        raise ValueError('Invalid subject: Passed subject has to be one of the following: AH,CB,DG,JR,LT')
    
    mask_array = np.flip(mask_array,2)
    mask_array = np.rot90(mask_array,1,(0,1))
    
  
    return mask_array[:,:,:,0,0]

#closing operation for postprocessing the segmentation to remove holes from the inside to avoid wrong seeds if used for 4D initialization
#takes a 3D volume and returns a 3D volume where every slice is eroded with a "circular" 3x3 kernel
#the rw data is then eroded and diluted and markers are assigned to the two classes, no markers are placed in the overlap so that the RW algorithm can fill these gaps
def erode_seg_markers(rw_data):
    
    closing_kernel = np.array([[[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]],
                
                       [[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]],
                
                       [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]], dtype=bool)
    
    erosion_kernel = closing_kernel
    dilation_kernel = erosion_kernel
    
    rw_bool = np.array(rw_data, dtype=bool)
    
    closed_seg = morph.binary_closing(rw_bool,structure=closing_kernel)
#    skeletonized_seg = mp.skeletonize_3d(closed_seg)
#    medial_axis_seg = mp.medial_axis(closed_seg)
    
    thinned_seg = np.zeros(rw_data.shape)
    for i in range(rw_data.shape[2]):
        thinned_seg[:,:,i] = mp.thin(closed_seg[:,:,i],max_iter = 5)
    eroded_seg = morph.binary_erosion(closed_seg,structure=erosion_kernel,iterations=3)
    dilated_seg = morph.binary_dilation(closed_seg,structure=dilation_kernel,iterations=6)
    
       
    fg_markers = np.zeros(rw_data.shape)   
    bg_markers = np.zeros(rw_data.shape)   
    markers = np.zeros(rw_data.shape)   
    
#    fg_markers = (np.logical_and(eroded_seg,dilated_seg))*1
    fg_markers = (np.logical_and(thinned_seg,dilated_seg))*1
    bg_markers = (np.logical_and(np.logical_not(markers),np.logical_not(dilated_seg)))*2
    
    markers = fg_markers + bg_markers
    
    return markers, fg_markers, bg_markers

def load_random_seeds(gt_data, npoints, slice_number = 0):
    seed_mask = np.zeros(gt_data.shape)
    nfg_points = np.count_nonzero(gt_data[:,:,slice_number])
    nbg_points = np.size(gt_data[:,:,slice_number])-nfg_points
    
    fg_points_ind = np.nonzero(gt_data[:,:,slice_number]==1)
    bg_points_ind = np.nonzero(gt_data[:,:,slice_number]==0)
    
    if nfg_points < npoints:
        raise ValueError('Chosen # of desired seeds exceeds the number of foreground pixels in this slice')
                         
    randfg = np.random.randint(0,nfg_points,npoints)
    randbg = np.random.randint(0,nbg_points,npoints)
    
    fg_seed_coords = list()
    bg_seed_coords = list()
    for x in range(npoints):
        fg_seed_coords.append(np.array([fg_points_ind[0][randfg[x]],fg_points_ind[1][randfg[x]]]))
        bg_seed_coords.append(np.array([bg_points_ind[0][randbg[x]],bg_points_ind[1][randbg[x]]]))
    
    for x in range(npoints):
        seed_mask[fg_seed_coords[x][0],fg_seed_coords[x][1],slice_number] = 1
        seed_mask[bg_seed_coords[x][0],bg_seed_coords[x][1],slice_number] = 2
    
    return seed_mask
    

def calculate_dice(seg_array,gt_array):

    dice = np.sum(seg_array[gt_array==1])*2.0 / (np.sum(seg_array) + np.sum(gt_array))
    
    return dice

def error_map(rw_seg,gt_data):
    error_map = np.logical_xor(rw_seg,gt_data)
    error_map1 = np.logical_and(error_map,gt_data)
    error_map2 = np.logical_and(error_map,rw_seg)
    
    return error_map1, error_map2

def plot_unceratinty_map(prob_map):
    prob_map[prob_map<0.5] = 1-prob_map[prob_map<0.5]
    uncertainty_map = 2*(prob_map-0.5)
    
    return uncertainty_map

def correct_segmentation(gt_data,rw_labels,markers_3d,npoints=5,img_slice=6):
    
    errormap_fg, errormap_bg = error_map(np.round(rw_labels[0,:,:,img_slice]),gt_data[:,:,img_slice])
    
    seed_mask = np.zeros(markers_3d.shape)
    nfg_points = np.count_nonzero(errormap_fg)
    nbg_points = np.count_nonzero(errormap_bg)
    
    fg_points_ind = np.nonzero(errormap_fg==1)
    bg_points_ind = np.nonzero(errormap_bg==1)
                         
    randfg = np.random.randint(0,nfg_points,npoints)
    randbg = np.random.randint(0,nbg_points,npoints)
    
    fg_seed_coords = list()
    bg_seed_coords = list()
    for x in range(npoints):
        fg_seed_coords.append(np.array([fg_points_ind[0][randfg[x]],fg_points_ind[1][randfg[x]]]))
        bg_seed_coords.append(np.array([bg_points_ind[0][randbg[x]],bg_points_ind[1][randbg[x]]]))
    
    for x in range(npoints):
        seed_mask[fg_seed_coords[x][0],fg_seed_coords[x][1],img_slice] = 1
        seed_mask[bg_seed_coords[x][0],bg_seed_coords[x][1],img_slice] = 2
    
    markers_3d_2 = markers_3d + seed_mask
    
    return markers_3d_2
    
    
    
def create_pcmra_images(parec_data, option = 'no_avg', save = False, subject='AH'):
    #function creates pcmra images
    #options include the ability to save the data for later use as a .npy array in a file in the working path
    #other optiuon allows to choose between keeping the times dimension or collapsing/averaging over the time slices
    
    separated_arrays = create_separated_arrays(parec_data)
    
    vm_vec = separated_arrays[0]
    vp_vec = separated_arrays[1]
    vs_vec = separated_arrays[2]
    m_vec = separated_arrays[3]
    
    num_times = parec_data[3]
    num_slices = parec_data[4]
    
    # create PCMRA slices by summing over the square of all velocity vectors, multiplying it by the squared magnitude image,
    # summing over all times and finally normalizing
    # with 1/numer of timesteps and the sqrt
    # data_s/m/p are strucured like : [px,px,slice numer,timestep] where the timesteps alternate with the magnitude (even slices) and velocity (odd slices)
    pcmra_img = np.zeros((int(vm_vec.shape[0]),int(vm_vec.shape[1]),num_slices))
    

    #account for the fact that we want to know the velocity magnitude and not normalize large negative flows to zero
    
    vs_vec_magn = np.sqrt(np.square(vs_vec))
    vm_vec_magn = np.sqrt(np.square(vm_vec))
    vp_vec_magn = np.sqrt(np.square(vp_vec))
    
    #normalize the vectors to 1
    
    vs_vec_max = vs_vec_magn.max()
    vs_vec_min = vs_vec_magn.min()
    vs_vec_norm = (vs_vec_magn - vs_vec_min)/(vs_vec_max - vs_vec_min)
    
    vm_vec_max = vm_vec_magn.max()
    vm_vec_min = vm_vec_magn.min()
    vm_vec_norm = (vm_vec_magn - vm_vec_min)/(vm_vec_max - vm_vec_min)
    
    vp_vec_max = vp_vec_magn.max()
    vp_vec_min = vp_vec_magn.min()
    vp_vec_norm = (vp_vec_magn - vp_vec_min)/(vp_vec_max - vp_vec_min)
    
    m_vec_max = m_vec.max()
    m_vec_min = m_vec.min()
    m_vec_norm = (m_vec - m_vec_min)/(m_vec_max - m_vec_min)
    
    if option == 'no_avg':
        pcmra_img = np.zeros(parec_data[2].shape)
         
        for s in range(num_slices):
            for x in range(vm_vec.shape[0]):
                for y in range(vm_vec.shape[1]):
                    for t in range (num_times):
                        v_squared = np.square(vs_vec_norm[x,y,s,t]) + np.square(vm_vec_norm[x,y,s,t]) + np.square(vp_vec_norm[x,y,s,t])
                        m_squared = np.square(m_vec_norm[x,y,s,t])
                        pcmra_img[x,y,s,t] = np.sqrt(m_squared*v_squared)
        
        return pcmra_img
        
        if save == True:
            file = open('saved_pcmra_img_noavg_' + str(subject) + ' .npy',"w+")
            np.save('saved_pcmra_img_noavg_' + str(subject) + ' .npy',pcmra_img)
            file.close()
    
    elif option == 'avg':
        pcmra_img = np.zeros((parec_data[2,0],parec_data[2,1],parec_data[2,2]))
         
        for s in range(num_slices):
            for x in range(vm_vec.shape[0]):
                for y in range(vm_vec.shape[1]):
                    v_squared = 0
                    m_squared = 0
                    for t in range (num_times):
                        v_squared += np.square(vs_vec_norm[x,y,s,t]) + np.square(vm_vec_norm[x,y,s,t]) + np.square(vp_vec_norm[x,y,s,t])
                        m_squared += np.square(m_vec_norm[x,y,s,t])
                        pcmra_img[x,y,s] += np.sqrt(m_squared*v_squared)
                        
        return pcmra_img
        
        if save == True:
            file = open('saved_pcmra_img_avg_' + str(subject) + ' .npy',"w+")
            np.save('saved_pcmra_img_avg_' + str(subject) + ' .npy',pcmra_img)
            file.close()
            
    else:
        warn('Invalid options input! Valid inputs are: no_avg, avg')
    
def GetCoord(img_array):
    
    event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))
    cont_coord_0 = []
    
    if __name__ == "parecload":
        root = Tk()
    
        #setting up a tkinter canvas with scrollbars
        frame = Frame(root, bd=2, relief=SUNKEN)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        xscroll = Scrollbar(frame, orient=HORIZONTAL)
        xscroll.grid(row=1, column=0, sticky=E+W)
        yscroll = Scrollbar(frame)
        yscroll.grid(row=0, column=1, sticky=N+S)
        canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky=N+S+E+W)
        xscroll.config(command=canvas.xview) 
        yscroll.config(command=canvas.yview)
        frame.pack(fill=BOTH,expand=1)
    
        #adding the image
        plt.figure()
        plt.imshow(img_array.T, cmap='gray')
        plt.xticks([], []); plt.yticks([], [])
        plt.gca().set_axis_off()
    #    plt.margins(0,0)
        plt.savefig('CoordImg' + '.png', bbox_inches = 'tight',
        pad_inches = 0)
        plt.close()
        
        aorta_img = Image.open('.//CoordImg.png')
        mrimage_conv = ImageTk.PhotoImage(aorta_img)
        canvas.create_image(0,0,image=mrimage_conv,anchor="nw")
        canvas.config(scrollregion=canvas.bbox(ALL))
    
        #function to be called when mouse is clicked
        def printcoords(event):
            #appending clicked coordinates to cont_coord_0 to form string of coordinates
            cx, cy = event2canvas(event, canvas)
            cont_coord_0.append(cx)
            cont_coord_0.append(cy)
            canvas.create_oval(cx, cy, cx+3, cy+3, fill='red')
            
            
        #mouseclick event
        canvas.bind("<ButtonPress-1>",printcoords)
    
        root.mainloop()
    
   
    return cont_coord_0

#markers are placed on tkinter window, which opens a new image that has previously been saved as a .png that has a different
#suîze than the original array that we use to compute the random walker so here we need to interpolate these points into a new array
def rescale_markers(markers,array_dim):
    coord = np.nonzero(markers)
    values = markers[np.nonzero(markers)]
    #-1 because we want to use these as index ranges that start at 0
    x_scale_old = markers.shape[0]-1
    y_scale_old = markers.shape[1]-1
    x_scale_new = array_dim[0]-1
    y_scale_new = array_dim[1]-1
    
    #create array with all zero values to write the markers into
    scaled_array = np.zeros((array_dim))
    
    new_coord_x = (np.round(coord[0]/x_scale_old*x_scale_new)).astype(int)
    new_coord_y = (np.round(coord[1]/y_scale_old*y_scale_new)).astype(int)
    
    for i in range(np.size(new_coord_x)):
        scaled_array[new_coord_x[i],new_coord_y[i]] = values[i]
    
    return scaled_array

#function to normalize the input arrays (intensity and velocity) to a range between 0 to 1 and -1 to 1
#magnitude normalization is a simple division by the largest value
#velocity normalization first calculates the largest magnitude and then uses the components of this vector to normalize the x,y and z directions seperately
def normalize_arrays(arrays):
    #dimension of normalized_arrays: 128 x 128 x 20 x 25 x 4
    normalized_arrays = np.zeros((arrays.shape))
    #normalize magnitude channel
    normalized_arrays[...,0] = arrays[...,0]/np.amax(arrays[...,0])
    #normalize velocities
    #calculate the velocity magnitude at each voxel
    velocity_arrays = gaussian_filter(np.array(arrays[...,1:4]),0.5)
    
    velocity_mag_array = np.sqrt(np.square(velocity_arrays[...,0])+np.square(velocity_arrays[...,1])+np.square(velocity_arrays[...,2]))
    #find max value of 95th percentile (to minimize effect of outliers) of magnitude array and its index
    vpercentile =  np.percentile(velocity_mag_array,95)
    velocity_mag_array[velocity_mag_array>vpercentile] = 1.0
    vmax = np.amax(velocity_mag_array)
    
    normalized_arrays[...,1] = velocity_arrays[...,0]
    normalized_arrays[...,2] = velocity_arrays[...,1]
    normalized_arrays[...,3] = velocity_arrays[...,2]
        
    normalized_arrays[normalized_arrays>vmax] = vmax
    normalized_arrays[normalized_arrays<-vmax] = -vmax
    
    normalized_arrays[...,1] /= vmax
    normalized_arrays[...,2] /= vmax
    normalized_arrays[...,3] /= vmax
    
    
#    print('normalized arrays: max=' + str(np.amax(normalized_arrays)) + ' min:' + str(np.amin(normalized_arrays)))
    
    return normalized_arrays

#function to create dummy array to test algorithm with known data (velocity vectors are known)
def create_dummy_array(xc,yc,zc,nc):
    dummy_array = np.zeros((xc,yc,zc,nc))
    #create largest "velocity contrast" in dot product
    dummy_array[:,:,:,1:] = -1
    #fill magnitude image with respective data
    #we choose a simple rectangle and create
    padding_x, padding_y, padding_z = int(xc/3), int(yc/3), int(zc/3)
    dummy_array[padding_x:xc-padding_x,padding_y:yc-padding_y,padding_z:zc-padding_z,:] = 1
    
    
    return(dummy_array)

#computes the dotproducts using the three channels of the passed array between adjacent px in four directions and returns array
def dot_products(data):
    
    dot_deep = (data[:,:,:-1,0]*data[:,:,1:,0]+data[:,:,:-1,1]*data[:,:,1:,1]+data[:,:,:-1,2]*data[:,:,1:,2])
    dot_right = (data[:,:-1,:,0]*data[:,1:,:,0]+data[:,:-1,:,1]*data[:,1:,:,1]+data[:,:-1,:,2]*data[:,1:,:,2])
    dot_down = (data[:-1,:,:,0]*data[1:,:,:,0]+data[:-1,:,:,1]*data[1:,:,:,1]+data[:-1,:,:,2]*data[1:,:,:,2])
    dot_down = np.concatenate((dot_down,np.zeros((1,128,20))),axis=0)
    dot_deep = np.concatenate((dot_deep,np.zeros((128,128,1))),axis=2)
    dot_right = np.concatenate((dot_right,np.zeros((128,1,20))),axis=1)    
    
    array = np.stack((dot_down,dot_right,dot_deep),axis=3)
    return array

def norm(x,y,z):
    normed_array = np.sqrt(np.square(x)+np.square(y)+np.square(z))
    return normed_array
    
def compute_similarity_maps(data,a,b,c):
    
    spacing = [1,1,1]
    
    intensity_img = data[...,0]
    
    A_ravel = rw._compute_delta_I_3d(data[...,0], spacing) ** 2
    B_ravel = rw._compute_eucl_dist_3d(data, spacing)
    C_ravel = -((rw._compute_dotproduct_3d(data, spacing))-1)
    
    dim1 = data.shape[0]*data.shape[1]*(data.shape[2]-1)
    dim2 = dim1+(data.shape[0]-1)*data.shape[1]*data.shape[2]
    
    A_list = [A_ravel[:dim1],A_ravel[dim1:dim2],A_ravel[dim2:]]
    B_list = [B_ravel[:dim1],B_ravel[dim1:dim2],B_ravel[dim2:]]
    C_list = [C_ravel[:dim1],C_ravel[dim1:dim2],C_ravel[dim2:]]

    Coords1 = np.nonzero(intensity_img[:,:,:-1]>0.1)
    Coords2 = np.nonzero(intensity_img[:,:-1,:]>0.1)
    Coords3 = np.nonzero(intensity_img[:-1,:,:]>0.1)
    
    A1 = np.reshape(A_list[0],(data.shape[0],data.shape[1],data.shape[2]-1))
    B_i1 = np.reshape(B_list[0],(data.shape[0],data.shape[1],data.shape[2]-1))
    B1 = np.zeros(A1.shape)
    B1[Coords1] = B_i1[Coords1]
    C_i1 = np.reshape(C_list[0],(data.shape[0],data.shape[1],data.shape[2]-1))
    C1 = np.zeros(A1.shape)
    C1[Coords1] = C_i1[Coords1]
    
    A2 = np.reshape(A_list[1],(data.shape[0],data.shape[1]-1,data.shape[2]))
    B_i2 = np.reshape(B_list[1],(data.shape[0],data.shape[1]-1,data.shape[2]))
    B2 = np.zeros(A2.shape)
    B2[Coords2] = B_i2[Coords2]
    C_i2 = np.reshape(C_list[1],(data.shape[0],data.shape[1]-1,data.shape[2]))
    C2 = np.zeros(A2.shape)
    C2[Coords2] = C_i2[Coords2]
    
    A3 = np.reshape(A_list[2],(data.shape[0]-1,data.shape[1],data.shape[2]))
    B_i3 = np.reshape(B_list[2],(data.shape[0]-1,data.shape[1],data.shape[2]))
    B3 = np.zeros(A3.shape)
    B3[Coords3] = B_i3[Coords3]
    C_i3 = np.reshape(C_list[2],(data.shape[0]-1,data.shape[1],data.shape[2]))
    C3 = np.zeros(A3.shape)
    C3[Coords3] = C_i3[Coords3]
    
    
    
    A1_a = A1*a/data[...,0].std()
    B1_b = B1*b/data[...,1:].std()
    C1_c = C1*c
    
    A2_a = A2*a/data[...,0].std()
    B2_b = B2*b/data[...,1:].std()
    C2_c = C2*c
    
    A3_a = A3*a/data[...,0].std()
    B3_b = B3*b/data[...,1:].std()
    C3_c = C3*c
    
    a1_exponent = np.exp(-A1_a)
    b1_exponent = np.exp(-B1_b)
    c1_exponent = np.exp(-C1_c)
    a2_exponent = np.exp(-A2_a)
    b2_exponent = np.exp(-B2_b)
    c2_exponent = np.exp(-C2_c)
    a3_exponent = np.exp(-A3_a)
    b3_exponent = np.exp(-B3_b)
    c3_exponent = np.exp(-C3_c)
    
    similarity_data = [[A1,A1_a,a1_exponent,B1,B1_b,b1_exponent,C1,C1_c,c1_exponent],[A2,A2_a,a2_exponent,B2,B2_b,b2_exponent,C2,C2_c,c2_exponent],[A3,A3_a,a3_exponent,B3,B3_b,b3_exponent,C3,C3_c,c3_exponent]]
    
    return similarity_data
    
def interactive_plot_3d(rw_data, raw_data, subject,a,b,c): 
#    bokeh serve --show 20190430_rw_newsim_parameter.py
  
    similarity_data = compute_similarity_maps(raw_data,a,b,c)
    data = rw_data
    data2 = similarity_data
    source = ColumnDataSource(data=dict(image=[np.flipud(data[:, :, 0].T)]))
    
    color_mapper = LinearColorMapper(palette="Greys256" , low=0.0, high=1.0)
    
    plot = figure(title="RW aorta segmentation")
    plot.image(image='image', x=0, y=0, dw=data.shape[0], dh=data.shape[1], source=source, color_mapper=color_mapper)
    
    source_dict = {}
    name_dict = {1: 'A', 2: 'A_a', 3: 'exp(-A_a)', 4: 'B', 5: 'B_b', 6: 'exp(-B_b)',7: 'C', 8: 'C_c', 9: 'exp(-C_c)'}
    plot_dict = {}
    color_mapper_dict = {}
    colorbar_dict = {}
#    
    for x in name_dict:
        source_dict["source{0}".format(x)]=ColumnDataSource(data=dict(image=[np.flipud(data2[0][x-1][:, :, 0].T)]))
        plot_dict['plot{0}'.format(x)] = figure(title="Similarity maps {0}".format(name_dict[x]), height=300,  match_aspect=True)
        color_mapper_dict['color_mapper{0}'.format(x)] = LinearColorMapper(palette="Inferno256", low=min(np.amin(data2[0][x-1]),np.amin(data2[1][x-1]),np.amin(data2[2][x-1])), high=max(np.amax(data2[0][x-1]),np.amax(data2[1][x-1]),np.amax(data2[2][x-1])))
        colorbar_dict['colorbar{0}'.format(x)] = ColorBar(color_mapper = color_mapper_dict["color_mapper{0}".format(x)], location=(0,0))
        plot_dict["plot{0}".format(x)].image(image='image', x=0, y=0, dw=data2[0][x-1].shape[0], dh=data2[0][x-1].shape[1], source=source_dict["source{0}".format(x)], color_mapper=color_mapper_dict["color_mapper{0}".format(x)])
        plot_dict["plot{0}".format(x)].add_layout(colorbar_dict["colorbar{0}".format(x)], 'right')
    
    
    gt_data = load_gt_data(subject=subject)
    gt_data = np.rot90(gt_data,1,(0,1))
    source10 = ColumnDataSource(data=dict(image=[gt_data[:, :, 0]]))
    plot10 = figure(title="GT data")
    plot10.image(image='image', x=0, y=0, dw=gt_data.shape[0], dh=gt_data.shape[1], source=source10, color_mapper=color_mapper)
    
    slider = Slider(start=0, end=(rw_data.shape[2]-1), value=0, step=1, title="Scroll through z-axis")
    slider2 = Slider(start=0, end=(rw_data.shape[2]-1), value=0, step=1, title="Scroll through z-axis",default_size=200)
    button = Button(label='Toggle segmentation')
    button2 = RadioButtonGroup(labels=['X-direction',"Y-direction","Z-direction"], active=0)
    
    gt_data2 = np.rot90(gt_data,1,(1,0))
    dice3d = calculate_dice(np.round(data),gt_data2)
    dice_3d = Div(text="The 3D dice score of current segmentation is:{0}".format(dice3d))
    
    dice2d = np.zeros(data.shape[2])
    for x in range(data.shape[2]):
        dice2d[x] = calculate_dice(np.round(data[:,:,x]),gt_data2[:,:,x])
    dice_2d =Div(text="The 2D dice score of current slice is:{0}".format(dice2d[0]))
    
    def update(attr, old, new):
        source10.data = dict(image=[gt_data[:, :, slider.value]])
        dice_2d.text = "The 2D dice score of current slice is:{0}".format(dice2d[slider.value])
        if button.label == 'Toggle segmentation':
            source.data = dict(image=[np.flipud(data[:, :, slider.value].T)])
        else:
            source.data = dict(image=[np.round(np.flipud(data[:, :, slider.value].T))])
    
    def update2(attr, old, new):        
        for x in name_dict:
            source_dict['source{0}'.format(x)].data = dict(image=[np.flipud(data2[button2.active][x-1][:, :, slider2.value].T)])
    
    def change_button_value():
        if button.label == 'Toggle segmentation':
            button.label='Toggle prob. map'
        else:
            button.label='Toggle segmentation'
        update('value',slider.value,slider.value)
        
    slider.on_change('value', update)
    slider2.on_change('value', update2)
    button.on_click(change_button_value)
    button2.on_change('active', update2)
    
    fig0 = row(button,slider)
    fig1 = row(plot,plot10,dice_3d,dice_2d)
    fig2 = row(button2,slider2)
    fig3 = column(row(plot_dict['plot1'],plot_dict['plot2'],plot_dict['plot3']),row(plot_dict['plot4'],plot_dict['plot5'],plot_dict['plot6']),row(plot_dict['plot7'],plot_dict['plot8'],plot_dict['plot9']))
    
    l1 = layout([[fig0],[fig1]], sizing_mode='fixed')
    l2 = layout([[fig2],[fig3]], sizing_mode='fixed')
    
    tab1 = Panel(child=l1,title="Random Walker")
    tab2 = Panel(child=l2,title="Similarity Maps")
    tabs = Tabs(tabs=[tab1, tab2])
#    tabs = Tabs(tabs=[tab1])
    
    curdoc().add_root(tabs)
 