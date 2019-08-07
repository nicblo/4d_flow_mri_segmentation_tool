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
from mayavi import mlab as mlab
import uuid

#----------------------------------------------------------------------------------------------------------
# 4D Random Walker Segmentation Tool for 4D MRI Flow Images
#----------------------------------------------------------------------------------------------------------

class MainWindow():
    
    #set mlab parameters so that no window is opened while the 3D view is saved as an image
    mlab.options.offscreen = True
    mlab.gcf().scene.parallel_projection = True
    
    #set default mlab view parameters
    azimuth = 100
    elevation = 20
    distance= 200
    focalpoint = (75,70,10)
    
    # set subject name to be loaded
    subject = 'JR' # AH, CB, DG, JR, LT, F1, F2
    
    #parec data contains the arrays with m,p and s (velocity components) as well as ints with the number of timesteps and number of slices
    #load parec datsa using custom function in pl (see imports)
    parec_data = pl.load_parec_data(subject = subject)
    
    #returns the arrays separated by velocity and magnitude components (4 arrays, magnitude first (index 0))
    #transpose the arrays by permuting the "channel" dimension to the back so we have: x,y,z,t,channel
    separated_arrays = np.transpose(np.array(pl.create_separated_arrays(parec_data)),(1,2,3,4,0))
    
    #normalize the arrays so that velocity and magnitude are in the range [0,1]
    #also this function uses the 95th percentile to normalize the data and clips any datapoint that is larger to 1.0 to get rid of outliers
    separated_arrays= pl.normalize_arrays(separated_arrays)
    
    #extract the array that contains the magnitude images
    magnitude_array = separated_arrays[...,0]
    img_array = separated_arrays[...,0]
    print('img_data shape: {}'.format(img_array.shape))
    
    #get array dimensions
    x_size = magnitude_array.shape[0]
    y_size = magnitude_array.shape[1]
    z_size = magnitude_array.shape[2]
    t_size = magnitude_array.shape[3]
    
    #in original data the images alternate between magnitude and velocity images, so the time steps and the actual volume depth are 0.5x this size
    img_timestep = round(t_size/2)
    img_slice =round(z_size/2)
    
    
#    widthc = 1200    
    
    #create placeholders / initialize variables for the different arrays and lists that the GUI uses
    gt_data = pl.load_gt_data(subject = subject)
    print('gt data.shape: {}'.format(gt_data.shape))
    rw_data = np.zeros(magnitude_array.shape)
    flow_data1 = np.zeros(magnitude_array.shape)
    flow_data2 = np.zeros(magnitude_array.shape)
    flow_data3 = np.zeros(magnitude_array.shape)
    rw_labels = np.zeros((2,magnitude_array.shape[0],magnitude_array.shape[1],magnitude_array.shape[2],magnitude_array.shape[3]))
    rw_labels3D = np.zeros((2,magnitude_array.shape[0],magnitude_array.shape[1],magnitude_array.shape[2]))

    fg_coord_list = []
    bg_coord_list = []
    
    fg_coord_lists = np.empty((z_size,),dtype=object)
    
    markers_3d_t = np.zeros(magnitude_array.shape)
    
    
    
    
    #----------------------------------------------------------------------------------------------------------
    # Define the needed callbacks and functions for the GUI to run
    #----------------------------------------------------------------------------------------------------------
    
    
    
    #function that updates the displayed images based on states of buttons (buttonXY['text'] and the arrays (img_array, fg_markers, bg_markers, gt_data, rw_data))
    #function modifies arrays, creates a plot, saves this as an img and then loads the image into the Canvas of the GUI
    def update_image(self):
        
        #decide what to show
        if self.button7['text'] == "FG Markers":
            plt.figure()
            plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep].T, cmap='gray', alpha = 0.8)
            plt.imshow(self.rw_data[:,:,self.img_slice,self.img_timestep].T, cmap='Reds', alpha = 0.4)
            plt.xticks([], []); plt.yticks([], [])
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
            plt.close()
        elif self.button7['text'] == "BG Markers":
            plt.figure()
            plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep].T, cmap='gray', alpha = 0.8)
            plt.imshow(self.fg_markers[:,:,self.img_slice].T, cmap='Reds', alpha = 0.3)
            plt.xticks([], []); plt.yticks([], [])
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
            plt.close()
        elif self.button7['text'] == "Vel. Magnitude":
            plt.figure()
            plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep].T, cmap='gray', alpha = 0.8)
            plt.imshow(self.bg_markers[:,:,self.img_slice].T, cmap='Reds', alpha = 0.3)
            plt.xticks([], []); plt.yticks([], [])
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
            plt.close()
        else:
            plt.figure()
            plt.imshow(self.img_array[:,:,self.img_slice,self.img_timestep].T, cmap='gray')
            plt.xticks([], []); plt.yticks([], [])
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
            plt.close()
        
        #plot and save the figure
        plt.figure()
        plt.imshow(self.gt_data[:,:,self.img_slice].T, cmap='gray')
        plt.xticks([], []); plt.yticks([], [])
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.savefig('Tkgtimg.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        
        plt.figure()
        plt.imshow(self.rw_data[:,:,self.img_slice,self.img_timestep].T, cmap='gray')
        plt.xticks([], []); plt.yticks([], [])
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.savefig('Tksegimg.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        
        #load the image and display on canvas
        pngimage = Image.open('.//Tkimg.png').resize(size=(self.x_size*3,self.y_size*3),resample = Image.BICUBIC )
        self.img =  ImageTk.PhotoImage(image=pngimage)
        self.canvas.create_image(0,0, anchor=NW, image=self.img)
        
        pnggtimage = Image.open('.//Tkgtimg.png').resize(size=(self.x_size*3,self.y_size*3),resample = Image.BICUBIC )
        self.gtimg =  ImageTk.PhotoImage(image=pnggtimage)
        self.canvas2.create_image(0,0, anchor=NW, image=self.gtimg)
        
        pngsegimage = Image.open('.//Tksegimg.png').resize(size=(self.x_size*3,self.y_size*3),resample = Image.BICUBIC )
        self.segimg =  ImageTk.PhotoImage(image=pngsegimage)
        self.canvas3.create_image(0,0, anchor=NW, image=self.segimg)

        return
    
    #----------------------------------------------------------------------------------------------------------
    # Function to perform update on 3D image (separate function as this is computational heavy)
    # again creating the figure (with mlab in 3D instead of plt in 2D), save it and then load the image to the GUI
    
    def update_3Dimage(self):
        
        self.azimuth = self.slider3.get()
        self.elevation = self.slider4.get()
        
        mlab.figure()
        mlab.contour3d(self.gt_data, colormap = 'gray')
        mlab.axes(xlabel='X', ylabel='Y', zlabel='Z') #Display axis
        mlab.orientation_axes()
        mlab.view(azimuth=self.azimuth, elevation=self.elevation, distance=self.distance, focalpoint = self.focalpoint)
        mlab.savefig('gt_3d.png')
        mlab.close()
        
        mlab.figure()
        mlab.contour3d(self.rw_data[:,:,:,self.img_timestep], colormap = 'gray')
        mlab.axes(xlabel='X', ylabel='Y', zlabel='Z') #Display axis
        mlab.orientation_axes()
        mlab.view(azimuth=self.azimuth, elevation=self.elevation, distance=self.distance, focalpoint = self.focalpoint)
        mlab.savefig('segm_3d.png')
        mlab.close()
        
#        mlab.figure()
#        mlab.flow(self.flow_data1,self.flow_data2,self.flow_data3)
#        mlab.axes(xlabel='X', ylabel='Y', zlabel='Z') #Display axis
#        mlab.orientation_axes()
#        mlab.view(azimuth=self.azimuth, elevation=self.elevation)
#        mlab.savefig('flow_3d.png')
#        mlab.close()
        
        png3dgtimage = Image.open('.//gt_3d.png').resize(size=(self.x_size*3,self.y_size*3),resample = Image.BICUBIC )
        self.gtimg3D =  ImageTk.PhotoImage(image=png3dgtimage)
        self.canvas4.create_image(0,0, anchor=NW, image=self.gtimg3D)
        
        png3dsegimage = Image.open('.//segm_3d.png').resize(size=(self.x_size*3,self.y_size*3),resample = Image.BICUBIC )
        self.segimg3D =  ImageTk.PhotoImage(image=png3dsegimage)
        self.canvas5.create_image(0,0, anchor=NW, image=self.segimg3D)
        
#        png3dflowimage = Image.open('.//flow_3d.png').resize(size=(self.x_size*3,self.y_size*3),resample = Image.BICUBIC )
#        self.flowimg3D =  ImageTk.PhotoImage(image=png3dflowimage)
#        self.canvas6.create_image(0,0, anchor=NW, image=self.flowimg3D)
        
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
            self.img_array = pl.norm(self.separated_arrays[...,1],self.separated_arrays[...,1],self.separated_arrays[...,1])
            
        elif self.button7['text'] == "Normal":
            self.button7.configure(text="Overlap image")
            self. img_array = self.separated_arrays[...,0]
            
        elif self.button7['text'] == "Overlap image":
            self.button7.configure(text="FG Markers")
            self. img_array = self.separated_arrays[...,0]
            
        elif self.button7['text'] == "FG Markers":
            self.button7.configure(text="BG Markers")
            self. img_array = self.separated_arrays[...,0]
        else:
            self.button7.configure(text="Vel. Magnitude")
        self.update_image()
            
        return
        
    #----------------------------------------------------------------------------------------------------------
    def mousecallback(self,event):
        x, y = event.x, event.y
        coord_tuple = (x,y)
        if x > 0 and y > 0 and x < self.x_size*3 and y < self.y_size*3:
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
    
    def resetvalues(self):
        self.v.set(0)
        self.button1.deselect()
        self.button2.deselect()
        self.slider1.config(state = NORMAL)
        self.slider2.config(state = NORMAL)
        self.canvas.delete("all")
        self.update_image()
        self.update_3Dimage()
        self.fg_coord_list = []
        self.bg_coord_list = []
        self.markers_3d_t = np.zeros(self.magnitude_array.shape)
        
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
    
    def save_scribble(self):
        if not self.fg_coord_list or not self.bg_coord_list:
            print("Nothing to save, draw scribble first")
        else:
            markers = np.zeros(self.magnitude_array[:,:,:,0].shape)
            for t in self.fg_coord_list:
                markers[round(t[0]/3),round(t[1]/3),self.img_slice] = 1
            for t in self.bg_coord_list:
                markers[round(t[0]/3),round(t[1]/3),self.img_slice] = 2
            
            timestring = round(time.time())
            file = open('markers_scribble_' + str(timestring) + str(self.subject) +' .npy',"w+")
            np.save('markers_scribble_' + str(timestring) + str(self.subject) +' .npy',markers)
            file.close()
        
        return
    
    #----------------------------------------------------------------------------------------------------------
    
    def add_scribble(self):
        self.v.set(0)
        self.button1.deselect()
        self.button2.deselect()
        self.slider1.config(state = NORMAL)
        self.slider2.config(state = NORMAL)
        self.canvas.delete("all")
        self.update_image()
        self.update_3Dimage()    
        self.init_markers()
        
        return

    #----------------------------------------------------------------------------------------------------------
    
    def init_markers(self):
        if self.fg_coord_list or self.bg_coord_list:
            self.markers_2d = np.zeros(self.magnitude_array[:,:,0,0].shape)
            for t in self.fg_coord_list:
                self.markers_2d[round(t[0]/3),round(t[1]/3)] = 1
            for t in self.bg_coord_list:
                self.markers_2d[round(t[0]/3),round(t[1]/3)] = 2
                    
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
#        a ,b ,c = 0,0,0
        self.dice, self.dice3D, self.dice4D = 0, 0, 0
        self.abc = [0,0,0]
    
        self.rw_labels = rw4D.random_walker(self.separated_arrays,self.markers_3d_t,mode='cg_mg',return_full_prob=True, alpha=alpha_a, beta=beta_b, gamma=gamma_g, a=a,b=b,c=c)
#        print(self.rw_labels.shape)
        self.dice = pl.calculate_dice(np.round(self.rw_labels[0,:,:,self.img_slice,self.img_timestep]),self.gt_data[:,:,self.img_slice])
        self.dice3D = pl.calculate_dice(np.round(self.rw_labels[0,:,:,:,self.img_timestep]),self.gt_data[:,:,:])
        for t in range(self.t_size):
            self.dice4D += pl.calculate_dice(np.round(self.rw_labels[0,:,:,:,t]),self.gt_data[:,:,:])
            
        self.dice4D /= self.t_size
    
        self.abc = [alpha_a,beta_b,gamma_g]
        
        self.label1.config(text="Dice for seeded slice: "+str(round(self.dice,3)))
        self.label2.config(text="3D Dice: "+str(round(self.dice3D,3))+" 4D dice: "+str(round(self.dice4D,3)))
        self.label3.config(text="Alpha: "+str(round(self.abc[0],2))+" Beta: "+str(round(self.abc[1],2))+" Gamma: "+str(round(self.abc[2],2)))
        
        self.toggle_segmentation()
        
        return
        
    #----------------------------------------------------------------------------------------------------------
    
    def run_random_walker3D(self):
        self.init_markers()
        
        alpha_a = 0.2
        beta_b = 0.4
        gamma_g = 1.0 - alpha_a - beta_b
        a ,b ,c = 200.0, 6.0, 500.0
#        a ,b ,c = 0,0,0
        self.dice, self.dice3D, self.dice4D = 0, 0, 0
        self.abc = [0,0,0]
    
        self.rw_labels3D = rw3D.random_walker(self.separated_arrays[...,self.img_timestep,:],self.markers_3d_t[...,self.img_timestep],mode='cg_mg',return_full_prob=True, alpha=alpha_a, beta=beta_b, gamma=gamma_g, a=a,b=b,c=c)
        self.dice = pl.calculate_dice(np.round(self.rw_labels3D[0,:,:,self.img_slice]),self.gt_data[:,:,self.img_slice])
        self.dice3D = pl.calculate_dice(np.round(self.rw_labels3D[0,:,:,:]),self.gt_data[:,:,:])
        
        
        self.eroded_markers3D, self.fg_markers, self.bg_markers = pl.erode_seg_markers(np.round(self.rw_labels3D[0,:,:,:]))
        
        self.rw_labels[...,self.img_timestep] = self.rw_labels3D
    
        self.abc = [alpha_a,beta_b,gamma_g]
        
        self.label1.config(text="Dice for seeded slice: "+str(round(self.dice,3)))
        self.label2.config(text="3D Dice: "+str(round(self.dice3D,3))+" 4D dice: -")
        self.label3.config(text="Alpha: "+str(round(self.abc[0],2))+" Beta: "+str(round(self.abc[1],2))+" Gamma: "+str(round(self.abc[2],2)))
        
        self.toggle_segmentation()
        
        return
        
    #----------------------------------------------------------------------------------------------------------
    
    def load_seg_mark(self):
#        self.markers_3d_t[...,self.img_timestep] = -np.round(self.rw_labels[0,...,self.img_timestep])+2
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
        
    #----------------------------------------------------------------------------------------------------------
    # Here all the elements of the GUI are defined (Buttons, Sliders, Canvas and Labels) with their sizes, values and callbacks (command)
    #----------------------------------------------------------------------------------------------------------
    
    def __init__(self, main):
        self.canvas = Canvas(main,width=self.x_size*3, height=self.y_size*3, background='white')
        self.canvas.grid(row=1,column=2,rowspan=7,sticky=W)
        
        self.canvas2 = Canvas(main,width=self.x_size*3, height=self.y_size*3, background='white')
        self.canvas2.grid(row=1,column=3,rowspan=7,sticky=W)
        
        self.canvas3 = Canvas(main,width=self.x_size*3, height=self.y_size*3, background='white')
        self.canvas3.grid(row=1,column=4,rowspan=7,sticky=W)
        
        self.canvas4 = Canvas(main,width=self.x_size*3, height=self.y_size*3, background='white')
        self.canvas4.grid(row=9,column=2,rowspan=7,sticky=W)
        
        self.canvas5 = Canvas(main,width=self.x_size*3, height=self.y_size*3, background='white')
        self.canvas5.grid(row=9,column=3,rowspan=7,sticky=W)
        
        self.canvas6 = Canvas(main,width=self.x_size*3, height=self.y_size*3, background='white')
        self.canvas6.grid(row=9,column=4,rowspan=7,sticky=W)

        
        self.slider1 = Scale(main, from_=0, to=self.z_size-1, length=200, tickinterval=5, orient=HORIZONTAL, label = "Z-Axis",command= lambda x: self.update_z_axis())
        self.slider1.set(round(self.z_size/2))
        self.slider1.grid(row=0,column =1,padx=5,pady=5)
        
        self.slider2 = Scale(main, from_=0, to=self.t_size-1, length=200, tickinterval=5, orient=HORIZONTAL,label = "T-Axis", command= lambda x: self.update_t_axis())
        self.slider2.set(round(self.t_size/2))
        self.slider2.grid(row=1,column =1,padx=5,pady=5)
        
        self.slider3 = Scale(main, from_=0, to=360,resolution=20, length=200, tickinterval=90, orient=HORIZONTAL,label = "Azimuth", command= lambda x: self.update_3Dimage())
        self.slider3.set(0)
        self.slider3.grid(row=10,column =1,padx=5,pady=5)
        
        self.slider4 = Scale(main, from_=0, to=180,resolution=20, length=200, tickinterval=90, orient=HORIZONTAL,label = "Elevation", command= lambda x: self.update_3Dimage())
        self.slider4.set(180)
        self.slider4.grid(row=11,column =1,padx=5,pady=5)
        
        self.button0 = Button(main, text='Run 4D', width=20, command= lambda : self.run_random_walker4D())
        self.button0.grid(row=2,column =1,padx=5,pady=5)
               
        self.v = IntVar()
        self.button1  = Radiobutton(main, text="Scribble FG",variable=self.v, value=1,indicatoron=0, width=20, command= lambda : self.scribble_draw())
        self.button2  = Radiobutton(main, text="Scribble BG",variable=self.v, value=2,indicatoron=0, width=20, command= lambda : self.scribble_draw())
        self.button1.grid(row=4,column =1,padx=5,pady=5)
        self.button2.grid(row=5,column =1,padx=5,pady=5)
        
        self.button3 = Button(main, text="Reset", width = 20, command= lambda: self.resetvalues())
        self.button3.grid(row=100,column =2,padx=5,pady=5)
        
        self.button4 = Button(main, text="Probability map", width = 20, command= lambda: self.toggle_segmentation())
        self.button4.grid(row=8,column =4,padx=5,pady=5)
        
        self.button5 = Button(main, text='Save Scribble', width = 20, command = lambda: self.save_scribble())
        self.button5.grid(row=7,column =1,padx=5,pady=5)
        
        self.button6 = Button(main, text='Add Scribble', width = 20, command = lambda: self.add_scribble())
        self.button6.grid(row=6, column=1,padx=5,pady=5)
        self.button6.config(state = DISABLED)
        
        self.button7  = Button(main, text="Overlap image", width=20, command= lambda: self.display_mode())
        self.button7.grid(row=8,column =2,padx=5,pady=5)
        
        self.button8 = Button(main, text="Run 3D", width=20, command= lambda: self.run_random_walker3D())
        self.button8.grid(row=3,column =1,padx=5,pady=5)
        
        self.button9 = Button(main, text="Load segm -> markers", width=20, command= lambda: self.load_seg_mark())
        self.button9.grid(row=8, column=1, padx=5, pady=5)
        
        self.button10 = Button(main, text="Save segmentation", width=20, command= lambda: self.save_seg())
        self.button10.grid(row=9, column=1, padx=5, pady=5)
    
    
        self.button_last = Button(main, text='Quit', width=20, command=main.destroy)
        self.button_last.grid(row=100,column =1,padx=5,pady=5)
        
        self.label1 = Label(text="Dice for seeded slice: -")
        self.label1.grid(row=0,column =2,padx=5,pady=5)
        
        self.label2 = Label(text="3D Dice: - 4D dice: -")
        self.label2.grid(row=0,column =3,padx=5,pady=5)
        
        self.label3 = Label(text="Alpha: - Beta: - Gamma: -")
        self.label3.grid(row=0,column =4,padx=5,pady=5)
        
#----------------------------------------------------------------------------------------------------------
# TKinter main looop
#----------------------------------------------------------------------------------------------------------
root = Tk()
root.title('Random Walker Segmentation GUI')
MainWindow(root)
root.mainloop()