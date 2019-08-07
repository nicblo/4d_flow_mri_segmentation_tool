# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:27:23 2019

@author: nblon
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import functions as pl
import random_walker_nb_4D as rw4D
import random_walker_nb_3D as rw3D
from tkinter import *
from PIL import Image, ImageTk
import time


#----------------------------------------------------------------------------------------------------------
# 4D Random Walker Segmentation Tool for 4D MRI Flow Images
#----------------------------------------------------------------------------------------------------------

class MainWindow():
    
    
    # set subject name to be loaded
    subjects = ['AH','CB','DG','JR','LT']
    subject = 4 # 'AH', 'CB', 'DG', 'JR', 'LT', 0, 1, .... 141
    
    #parec data contains the arrays with m,p and s (velocity components) as well as ints with the number of timesteps and number of slices
    #load parec datsa using custom function in pl (see imports)
    if subject in subjects:
        parec_data = pl.load_parec_data(subject = subject)
        
        #returns the arrays separated by velocity and magnitude components (4 arrays, magnitude first (index 0))
        #transpose the arrays by permuting the "channel" dimension to the back so we have: x,y,z,t,channel
        separated_arrays = np.transpose(np.array(pl.create_separated_arrays(parec_data)),(1,2,3,4,0))
        
         #normalize the arrays so that velocity and magnitude are in the range [0,1]
         #also this function uses the 95th percentile to normalize the data and clips any datapoint that is larger to 1.0 to get rid of outliers
        separated_arrays= pl.normalize_arrays2(separated_arrays)
        
        #extract the array that contains the magnitude images
        magnitude_array = separated_arrays[...,0]
        img_array = separated_arrays[...,0]        #get array dimensions
        
        x_size = magnitude_array.shape[1]
        y_size = magnitude_array.shape[0]
        z_size = magnitude_array.shape[2]
        t_size = magnitude_array.shape[3]
        

        
        #in original data the images alternate between magnitude and velocity images, so the time steps and the actual volume depth are 0.5x this size
        img_timestep = round(t_size/2)
        img_slice =round(z_size/2)
        
    else:
        separated_arrays = pl.load_npy_data(subject)
        separated_arrays = pl.normalize_arrays2(separated_arrays)
        
#        plt.figure()
#        plt.imshow(pl.norm(separated_arrays[...,1],separated_arrays[...,2],separated_arrays[...,3])[:,:,20,20], cmap='Greys')
#        plt.show()
        
        
        #extract the array that contains the magnitude images
        magnitude_array = separated_arrays[...,0]
        img_array = separated_arrays[...,0]
        
        #get array dimensions
        x_size = magnitude_array.shape[1]
        y_size = magnitude_array.shape[0]
        z_size = magnitude_array.shape[2]
        t_size = magnitude_array.shape[3]
        
        img_timestep = t_size-1
        img_slice = z_size-1
        
#    print(separated_arrays.shape)
#    print(np.unique(separated_arrays[...,2],return_counts=True))
     
    #create placeholders / initialize variables for the different arrays and lists that the GUI uses
    rw_data = np.zeros(magnitude_array.shape)
    rw_labels = np.zeros((2,magnitude_array.shape[0],magnitude_array.shape[1],magnitude_array.shape[2],magnitude_array.shape[3]))
    rw_labels3D = np.zeros((2,magnitude_array.shape[0],magnitude_array.shape[1],magnitude_array.shape[2]))

    fg_coord_list = []
    bg_coord_list = []
    
    markers_3d_t = np.empty(magnitude_array.shape)
    
    markers_visible = False
    
    
    #----------------------------------------------------------------------------------------------------------
    # Define the needed callbacks and functions for the GUI to run
    #----------------------------------------------------------------------------------------------------------
    
    
    
    #function that updates the displayed images based on states of buttons (buttonXY['text'] and the arrays (img_array, fg_markers, bg_markers, gt_data, rw_data))
    #function modifies arrays, creates a plot, saves this as an img and then loads the image into the Canvas of the GUI
    def update_image(self):
        
        
        if self.button7['text'] == "Vel. Magnitude":
            if self.button_overlap['text'] == "No Overlap":
                plt.figure()
                if self.subject in self.subjects:
                    plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep].T, cmap='gray', alpha = 0.8)
                else:
                    plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep], cmap='gray', alpha = 0.8)
                plt.imshow(self.rw_data[:,:,self.img_slice,self.img_timestep], cmap='Reds', alpha = 0.2)
                if self.markers_visible:
                    plt.imshow(self.markers_3d_t_cpy[:,:,self.img_slice,self.img_timestep], cmap='Greens',interpolation='none',alpha = 0.6)
                plt.xticks([], []); plt.yticks([], [])
                plt.gca().set_axis_off()
                plt.margins(0,0)
                plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
                plt.close()
            else:
                plt.figure()
                if self.subject in self.subjects:
                    plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep].T, cmap='gray')
                else:
                    plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep], cmap='gray')
                if self.markers_visible:
                    plt.imshow(self.markers_3d_t_cpy [:,:,self.img_slice,self.img_timestep], cmap='Greens',interpolation='none',alpha = 0.6)
                plt.xticks([], []); plt.yticks([], [])
                plt.gca().set_axis_off()
                plt.margins(0,0)
                plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
                plt.close()

        
        elif self.button7['text'] == "Normal":
            if self.button_overlap['text'] == "No Overlap":
                plt.figure()
                if self.subject in self.subjects:
                    plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep].T, cmap='gray', alpha = 0.8)
                else:
                    plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep], cmap='gray', alpha = 0.8)
                plt.imshow(self.rw_data[:,:,self.img_slice,self.img_timestep], cmap='Reds', alpha = 0.2)
                if self.markers_visible:
                    plt.imshow(self.markers_3d_t_cpy[:,:,self.img_slice,self.img_timestep], cmap='Greens',interpolation='none',alpha = 0.6)
                plt.xticks([], []); plt.yticks([], [])
                plt.gca().set_axis_off()
                plt.margins(0,0)
                plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
                plt.close()
            else:
                plt.figure()
                if self.subject in self.subjects:
                    plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep].T, cmap='gray')
                else:
                    plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep], cmap='gray')
                if self.markers_visible:
                    plt.imshow(self.markers_3d_t_cpy[:,:,self.img_slice,self.img_timestep], cmap='Greens', interpolation='none',alpha = 0.6)
                plt.xticks([], []); plt.yticks([], [])
                plt.gca().set_axis_off()
                plt.margins(0,0)
                plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
                plt.close()
        
        else:
            plt.figure()
            plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep], cmap='gray')
            plt.imshow(self.markers_3d_t[:,:,self.img_slice,self.img_timestep], cmap='Greens', interpolation='none',alpha = 0.6)
            plt.xticks([], []); plt.yticks([], [])
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
            plt.close()
        
        
        plt.figure()
        plt.imshow(self.rw_data[:,:,self.img_slice,self.img_timestep], cmap='gray')
        plt.xticks([], []); plt.yticks([], [])
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.savefig('Tksegimg.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        
        #load the image and display on canvas
        pngimage = Image.open('.//Tkimg.png').resize(size=(self.x_size*3,self.y_size*3),resample = Image.BICUBIC )
        self.img =  ImageTk.PhotoImage(image=pngimage)
        self.canvas.create_image(0,0, anchor=NW, image=self.img)
                
        pngsegimage = Image.open('.//Tksegimg.png').resize(size=(self.x_size*3,self.y_size*3),resample = Image.BICUBIC )
        self.segimg =  ImageTk.PhotoImage(image=pngsegimage)
        self.canvas2.create_image(0,0, anchor=NW, image=self.segimg)

        return      

    #----------------------------------------------------------------------------------------------------------
    # use slider1 to set the desired slice
    def update_z_axis(self):
        self.img_slice = self.slider1.get()
        self.update_image()
        
        return
        
    #----------------------------------------------------------------------------------------------------------
    # use slider2 to set the desired timestep   
    def update_t_axis(self):
        self.img_timestep = self.slider2.get()
        self.update_image()
        
        return
    #----------------------------------------------------------------------------------------------------------
    # loop to manage what to display (magnitude, overlap, markers, etc.)
    def display_mode(self):
          
        if self.button7['text'] == "Vel. Magnitude":
            self.button7.configure(text="Normal")
            self.img_array = pl.norm(self.separated_arrays[...,1],self.separated_arrays[...,2],self.separated_arrays[...,3])
        
        elif self.button7['text'] == "Normal":
            self.button7.configure(text="Vel. Magnitude")
            self. img_array = self.separated_arrays[...,0]
                    
        self.update_image()
            
        return
    
        #----------------------------------------------------------------------------------------------------------
    
    def display_mode2(self):
       
        if self.button_overlap['text'] == "Toggle Overlap":
            self.button_overlap.configure(text="No Overlap")
        
        elif self.button_overlap['text'] == "No Overlap":
            self.button_overlap.configure(text="Overlap")
        
        elif self.button_overlap['text'] == "Overlap":
            self.button_overlap.configure(text="No Overlap")
        
        self.update_image()
        
        return
        
    #----------------------------------------------------------------------------------------------------------
    def mousecallback(self,event):
        x, y = event.x, event.y
        coord_tuple = (x,y)
        if x > 0 and y > 0 and x < self.x_size*3-1 and y < self.y_size*3-1:
            if self.v.get() == 1:
                if coord_tuple not in self.fg_coord_list:
                    self.canvas.create_oval(x, y, x+3, y+3, fill='green')
                    self.fg_coord_list.append(coord_tuple)
            elif self.v.get() ==2:
                if coord_tuple not in self.bg_coord_list:
                    self.canvas.create_oval(x, y, x+3, y+3, fill='blue')
                    self.bg_coord_list.append(coord_tuple)
            else:
                return
    
    #----------------------------------------------------------------------------------------------------------
    
    def scribble_draw(self):
        self.canvas.bind("<B1-Motion>", self.mousecallback)
        self.slider1.config(state = DISABLED)
        self.slider2.config(state = DISABLED)
        self.button6.config(state = NORMAL)
        
        return

    #----------------------------------------------------------------------------------------------------------
    
    def toggle_segmentation(self):
        if self.button4['text'] == "Probability map":
            self.button4.configure(text="Segmentation")
            self.rw_data = self.rw_labels[0,...]
            self.update_image()

        else:
            self.button4.configure(text="Probability map")
            self.rw_data = np.round(self.rw_labels[0,...])
            self.update_image()
            
        return
    
    #----------------------------------------------------------------------------------------------------------
    
    def add_scribble(self):
        self.v.set(0)
        self.button1.deselect()
        self.button2.deselect()
        self.slider1.config(state = NORMAL)
        self.slider2.config(state = NORMAL)
        self.canvas.delete("all")
        self.init_markers()
        
        if len(np.unique(self.markers_3d_t)) > 1:
            self.markers_visible = True
            self.markers_3d_t_cpy = np.copy(self.markers_3d_t)
            self.markers_3d_t_cpy  = np.ma.masked_where(self.markers_3d_t < 1, self.markers_3d_t)    
        
        self.update_image()  
        
        return

    #----------------------------------------------------------------------------------------------------------
    
    def init_markers(self):
        if self.fg_coord_list or self.bg_coord_list:
            self.markers_2d = np.zeros(self.magnitude_array[:,:,0,0].shape)
            for t in self.fg_coord_list:
                self.markers_2d[round(t[1]/3)-1,round(t[0]/3)-1] = 1
            for t in self.bg_coord_list:
                self.markers_2d[round(t[1]/3)-1,round(t[0]/3)-1] = 2
                    
            self.markers_3d_t[:,:,self.img_slice,self.img_timestep] = self.markers_2d
            self.fg_coord_list = []
            self.bg_coord_list = []
            
        return
        
    #----------------------------------------------------------------------------------------------------------
    
    def run_random_walker4D(self):
        self.init_markers()
        
        alpha_a = 0.2
        beta_b = 0.4
        gamma_g = 1.0 - alpha_a - beta_b
        a ,b ,c = 200.0, 6.0, 500.0
    
        self.rw_labels = rw4D.random_walker(self.separated_arrays,self.markers_3d_t,mode='cg_mg',return_full_prob=True, alpha=alpha_a, beta=beta_b, gamma=gamma_g, a=a,b=b,c=c)
        
        self.toggle_segmentation()
               
        return
        
    #----------------------------------------------------------------------------------------------------------
    
    def run_random_walker3D(self):
        self.init_markers()
        
        alpha_a = 0.2
        beta_b = 0.4
        gamma_g = 1.0 - alpha_a - beta_b
        a ,b ,c = 200.0, 6.0, 500.0
    
        self.rw_labels3D = rw3D.random_walker(self.separated_arrays[...,self.img_timestep,:],self.markers_3d_t[...,self.img_timestep],mode='cg_mg',return_full_prob=True, alpha=alpha_a, beta=beta_b, gamma=gamma_g, a=a,b=b,c=c)      
        
        self.eroded_markers3D, self.fg_markers, self.bg_markers = pl.erode_seg_markers(np.round(self.rw_labels3D[0,:,:,:]))
        
        self.rw_labels[...,self.img_timestep] = self.rw_labels3D
        
        self.add_scribble()
        
        self.toggle_segmentation()
        
        return
        
    #----------------------------------------------------------------------------------------------------------
    
    def load_seg_mark(self):
        for t in range(self.t_size):
            self.markers_3d_t[...,t] = self.eroded_markers3D
        self.fg_coord_list = []
        self.bg_coord_list = []
        
        return
    #----------------------------------------------------------------------------------------------------------   
    
    def save_seg(self):
        save_path = os.path.join(os.getcwd(),'output')
        seg_name = 'rw_seg_{}.npy'.format(self.subject)
        
        np.save(os.path.join(save_path,seg_name),np.round(self.rw_labels[0,...]))
        return
    
    def run_seg(self):
        
        self.load_seg_mark()
        
        self.run_random_walker4D()
        
        return
    #----------------------------------------------------------------------------------------------------------
    # Here all the elements of the GUI are defined (Buttons, Sliders, Canvas and Labels) with their sizes, values and callbacks (command)
    #----------------------------------------------------------------------------------------------------------
    
    def __init__(self, main):
        self.canvas = Canvas(main,width=self.x_size*3, height=self.y_size*3, background='white')
        self.canvas.grid(row=1,column=2,rowspan=7,sticky=W)
        
        self.canvas2 = Canvas(main,width=self.x_size*3, height=self.y_size*3, background='white')
        self.canvas2.grid(row=1,column=3,rowspan=7,sticky=W)
                
        self.slider1 = Scale(main, from_=0, to=self.z_size-1, length=200, tickinterval=5, orient=HORIZONTAL, label = "Z-Axis",command= lambda x: self.update_z_axis())
        self.slider1.set(round(self.z_size/2))
        self.slider1.grid(row=0,column =1,padx=5,pady=5)
        
        self.slider2 = Scale(main, from_=0, to=self.t_size-1, length=200, tickinterval=5, orient=HORIZONTAL,label = "T-Axis", command= lambda x: self.update_t_axis())
        self.slider2.set(round(self.t_size/2))
        self.slider2.grid(row=1,column =1,padx=5,pady=5)
        
        self.button_run4d = Button(main, text='Run 4D', width=20, command= lambda : self.run_random_walker4D())
        self.button_run4d.grid(row=2,column =1,padx=5,pady=5)
        
        self.button_run3d = Button(main, text='Run 3D', width=20, command= lambda : self.run_random_walker3D())
        self.button_run3d.grid(row=3,column =1,padx=5,pady=5)
        
        self.button_run = Button(main, text='3D -> Run 4D', width=20, command= lambda : self.run_seg())
        self.button_run.grid(row=4,column =1,padx=5,pady=5)
               
        self.v = IntVar()
        self.button1  = Radiobutton(main, text="Scribble FG",variable=self.v, value=1,indicatoron=0, width=20, command= lambda : self.scribble_draw())
        self.button2  = Radiobutton(main, text="Scribble BG",variable=self.v, value=2,indicatoron=0, width=20, command= lambda : self.scribble_draw())
        self.button1.grid(row=5,column =1,padx=5,pady=5)
        self.button2.grid(row=6,column =1,padx=5,pady=5)
        
        self.button4 = Button(main, text="Probability map", width = 20, command= lambda: self.toggle_segmentation())
        self.button4.grid(row=8,column =3,padx=5,pady=5)
                
        self.button6 = Button(main, text='Add Scribble', width = 20, command = lambda: self.add_scribble())
        self.button6.grid(row=7, column=1,padx=5,pady=5)
        self.button6.config(state = DISABLED)
        
        self.button7  = Button(main, text="Vel. Magnitude", width=20, command= lambda: self.display_mode())
        self.button7.grid(row=8,column =2,padx=5,pady=5)
        
        self.button_overlap  = Button(main, text="Toggle Overlap", width=20, command= lambda: self.display_mode2())
        self.button_overlap.grid(row=9,column =2,padx=5,pady=5)
        
        self.button10 = Button(main, text="Save segmentation", width=20, command= lambda: self.save_seg())
        self.button10.grid(row=9, column=1, padx=5, pady=5)
    
        self.button_last = Button(main, text='Quit', width=20, command=main.destroy)
        self.button_last.grid(row=100,column =1,padx=5,pady=5)
        
#        self.label1 = Label(text="==== INFO ===== Draw some seeds")
#        self.label1.grid(row=0,column =2,padx=5,pady=5)
        
        
#----------------------------------------------------------------------------------------------------------
# TKinter main looop
#----------------------------------------------------------------------------------------------------------
root = Tk()
root.title('Random Walker Segmentation GUI')
MainWindow(root)
root.mainloop()