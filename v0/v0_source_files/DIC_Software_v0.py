# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:38:14 2022

@author: s1879083
"""
import tkinter as tk
import numpy as np
from PIL import ImageTk, Image
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Use for convolving images for pixel tracking
from numpy.fft import fft2, ifft2
import glob
# Used for subpixel analysis
import itertools
# Used for storing data
# import pandas as pd
# Used for timing code
import time
import datetime


class GraphPlot():
    def __init__(self, master):
        self.master = master

        def close_app():

            self.master.destroy()
            sys.exit("Programme closed")

        def next_screen():
            if self.screen_number==0:
                self.photo_path = self.photo_path.get()
                self.start_photo = self.start_photo.get()
                self.load_path = self.load_path.get()
                self.output_path = self.output_path.get()
                self.output_file = self.output_file.get()

            self.screen_number+=1

            if self.screen_number==1:
                for widget in self.master.winfo_children()[4:]:
                    widget.destroy()
                reference_length_screen()

            if self.screen_number==2:
                self.reference_length = self.reference_length.get()
                self.reference_line_length = ((self.ref_line_coords['x'] - self.ref_line_coords['x2']) ** 2 +
                                              (self.ref_line_coords['y'] - self.ref_line_coords['y2']) ** 2) ** (1 / 2)
                self.reference_line_length = self.reference_line_length/self.photo_canvas.winfo_width()
                self.scale_factor = float(self.reference_length)/self.reference_line_length

                for widget in self.master.winfo_children()[4:]:
                    widget.destroy()
                tracking_zone_screen()

            if self.screen_number==3:
                # Get dimensions of plotted image
                self.ref_im_w = self.photo_canvas.winfo_width()
                self.ref_im_h = self.photo_canvas.winfo_height()

                # Get coordinates of tracking zone
                # Co-ordinate of cetre of box
                self.tz_x0 = np.mean((self.ref_rect_coords['x'],self.ref_rect_coords['x2']))/self.ref_im_w
                self.tz_y0 = np.mean((self.ref_rect_coords['y'],self.ref_rect_coords['y2']))/self.ref_im_h
                self.tz_w = abs(self.ref_rect_coords['x']-self.ref_rect_coords['x2'])/self.ref_im_w
                self.tz_h = abs(self.ref_rect_coords['y']-self.ref_rect_coords['y2'])/self.ref_im_h


                for widget in self.master.winfo_children()[4:]:
                    widget.destroy()
                tracking_margin_screen()
            if self.screen_number==4:
                try:
                    self.ext_area_yb = float(self.vmarg.get())
                    self.ext_area_xr = float(self.hmarg.get())

                    self.ext_area_yt = np.min(((self.tz_y0-self.tz_h/2),self.ext_area_yb))
                    self.ext_area_xl = np.min(((self.tz_x0-self.tz_w/2),self.ext_area_xr))
                except:
                    0
                for widget in self.master.winfo_children()[3:]:
                    widget.destroy()
                output_plots_screen()
        def previous_screen():
            if self.screen_number>=1:
                self.screen_number+=-1

                if self.screen_number==0:
                    for widget in self.master.winfo_children()[4:]:
                        widget.destroy()
                    input_information_screen()
                if self.screen_number==1:
                    for widget in self.master.winfo_children()[4:]:
                        widget.destroy()
                    reference_length_screen()

                if self.screen_number==2:

                    for widget in self.master.winfo_children()[4:]:
                        widget.destroy()

                    tracking_zone_screen()

                if self.screen_number==3:

                    for widget in self.master.winfo_children()[3:]:
                        widget.destroy()
                    tk.Button(self.master, text= "Next", command= next_screen).place(relx= .95, rely= .95, anchor= tk.CENTER)

                    tracking_margin_screen()
        # Create GUI window
        # Setting the Tkinter window and the canvas in place
        self.ws = master.winfo_screenwidth()
        self.hs = master.winfo_screenheight()
        ww = self.ws*0.75
        hw = self.hs*0.75
        self.canvas = tk.Canvas(self.master, width = ww, height = hw, bg = 'grey')
        self.canvas.grid()
        self.master.update()
        self.w = self.canvas.winfo_width()
        self.h = self.canvas.winfo_height()
        self.canvas.focus_set()

        tk.Button(self.master, text= "Close", command= close_app).place(relx= .95, rely= .05, anchor= tk.CENTER)
        tk.Button(self.master, text= "Previous", command= previous_screen).place(relx= .9, rely= .95, anchor= tk.CENTER)
        tk.Button(self.master, text= "Next", command= next_screen).place(relx= .95, rely= .95, anchor= tk.CENTER)

        # Variables used in analysis
        self.screen_number = 0
        self.ref_line_coords = {"x":-1,"y":-1,"x2":-2,"y2":-2} # none flag indicates start of line
        self.ref_rect_coords = {"x":-1,"y":-1,"x2":-2,"y2":-2} # none flag indicates start of line
        self.lines = []
        self.rects_list = []
        self.reference_length = 1000
        self.ext_area_yb = 0.025
        self.ext_area_xr = 0.025

        self.photo_path = r'C:\Users\S1879083\OneDrive - University of Edinburgh\RC3 Madagascar\Data from fieldwork Madagascar\Data for DIC-Load analysis\Data from Rafik\Frame test--DIC@5s\Test28----@2-75'
        self.start_photo = 'G0538359.JPG'
        self.load_path = r'C:\Users\S1879083\OneDrive - University of Edinburgh\RC3 Madagascar\Data from fieldwork Madagascar\Data for DIC-Load analysis\Data from Rafik\Frame test--DIC@5s\Test28----@2-75\@2---OBLB75.csv'
        self.output_path = r'C:\Users\S1879083\OneDrive - University of Edinburgh\Edi_Python\Video_tracking\Video_tracking_app\2022_06_08_Blank_app_template'
        self.output_file = 'Output_data.csv'

        def poly_matrix(x, y, order=2):
            """
            Function for generating a 2d gaussian function from a given grid of data

            Parameters
            ----------
            x : Numpy Array
                X-coordinates of gaussian to generate.
            y : Numpy Array
                Y-coordinates of gaussian to generate.
            order : Int, optional
                The order of the 2D gaussian to generate. The default is 2.

            Returns
            -------
            G : Numpy 2D array
                A 2D gaussian function for coordinates (x,y).

            """
            # Calculate number of columns needed for gaussian
            ncols = (order + 1)**2
            # Initialize empty array
            G = np.zeros((x.size, ncols))
            # Calculate the product of the order
            ij = itertools.product(range(order+1), range(order+1))
            # Iterate through values of product to calculate 2D gaussian
            for k, (i, j) in enumerate(ij):
                G[:, k] = x**i * y**j
            return G
        def input_information_screen():

            self.photo_path = tk.StringVar(self.master,self.photo_path)
            self.load_path = tk.StringVar(self.master,self.load_path)
            self.start_photo = tk.StringVar(self.master,self.start_photo)
            self.output_path = tk.StringVar(self.master,self.output_path)
            self.output_file = tk.StringVar(self.master,self.output_file)

            tk.Label(self.master, text= "Enter:\n-the filepath to the folder containing GoPro photos,"+
                     "\n-the name of the first photo to use (including file extension),"+
                     "\n-the filepath to the load cell data,"+
                     "\n-The filepath where the output data should be saved,"+
                     "\n-The name of the output file (including file extension)",justify=tk.LEFT).place(relx= .05,rely= .15,
                                                                                                        anchor= tk.W)

            self.lab1 = tk.Label(self.master,text='Photo folder path',name='lab1').place(relx= .05, rely= .25, anchor= tk.W)
            tk.Entry(self.master, width= 150,textvariable=self.photo_path).place(relx= .05, rely= .3, anchor= tk.W)

            self.lab2 = tk.Label(self.master,text='First photo to use').place(relx= .05, rely= .35, anchor= tk.W)
            tk.Entry(self.master, width= 150,textvariable=self.start_photo).place(relx= .05, rely= .4, anchor= tk.W)

            self.lab3 = tk.Label(self.master,text='Filepath to load cell data').place(relx= .05, rely= .45, anchor= tk.W)
            tk.Entry(self.master, width= 150,textvariable=self.load_path).place(relx= .05, rely= .5, anchor= tk.W)

            self.lab4 = tk.Label(self.master,text='Output folder path').place(relx= .05, rely= .55, anchor= tk.W)
            tk.Entry(self.master, width= 150,textvariable=self.output_path).place(relx= .05, rely= .6, anchor= tk.W)

            self.lab5 = tk.Label(self.master,text='Output filename').place(relx= .05, rely= .65, anchor= tk.W)
            tk.Entry(self.master, width= 150,textvariable=self.output_file).place(relx= .05, rely= .7, anchor= tk.W)
        def click(e):
            if len(self.lines)>0:
                self.photo_canvas.delete(self.lines[-1])
            self.ref_line_coords["x"]=None
            # define start point for line
            self.ref_line_coords["x"] = e.x
            self.ref_line_coords["y"] = e.y
            # create a line on this point and store it in the list
            self.lines.append(self.photo_canvas.create_line(self.ref_line_coords["x"],
                                                      self.ref_line_coords["y"],
                                                      self.ref_line_coords["x"],
                                                      self.ref_line_coords["y"], width=2, fill='red'))
        def drag(e):
            # update the coordinates from the event
            self.ref_line_coords["x2"] = e.x
            self.ref_line_coords["y2"] = e.y
            # Change the coordinates of the last created line to the new coordinates
            self.photo_canvas.coords(self.lines[-1], self.ref_line_coords["x"],
                               self.ref_line_coords["y"],
                               self.ref_line_coords["x2"],
                               self.ref_line_coords["y2"])
        def reference_length_screen():
            self.photo_canvas = tk.Canvas(self.master, width = int(ww*0.7), height = int(ww*0.7*0.75), bg = 'black')
            self.photo_canvas.place(x=int(0.05*ww),y=int(0.05*hw), anchor= 'nw')
            self.img = Image.open(self.photo_path+'\\'+self.start_photo).resize((int(ww*0.7),int(ww*0.7*0.75)), Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(self.img)
            self.photo_canvas.create_image(0,0, image=self.img, anchor='nw')

            self.reference_length = tk.StringVar(self.master, value=str(self.reference_length))
            self.lines.append(self.photo_canvas.create_line(self.ref_line_coords["x"],
                                                      self.ref_line_coords["y"],
                                                      self.ref_line_coords["x2"],
                                                      self.ref_line_coords["y2"], width=2, fill='red'))


            tk.Label(self.master, text= "Click and drag on image to set known"+
                     "\nlength and update reference length box"+
                     "\nwith length in mm before clicking next.",justify=tk.LEFT).place(relx= .8,rely= .15,
                                                                                                        anchor= tk.W)
            self.photo_canvas.bind("<ButtonPress-1>", click)
            self.photo_canvas.bind("<B1-Motion>", drag)

            tk.Label(self.master, text= "Reference length [mm]").place(relx= .8, rely= .75, anchor= tk.W)
            tk.Entry(self.master, width= 10,textvariable=self.reference_length).place(relx= .8, rely= .8, anchor= tk.W)
        def click_rect(e):
            if len(self.rects_list)>0:
                self.photo_canvas.delete(self.rects_list[-1])
            self.ref_rect_coords["x"]=None
            # define start point for line
            self.ref_rect_coords["x"] = e.x
            self.ref_rect_coords["y"] = e.y
            # create a line on this point and store it in the list
            self.rects_list.append(self.photo_canvas.create_rectangle(self.ref_rect_coords["x"],
                                                                 self.ref_rect_coords["y"],
                                                                 self.ref_rect_coords["x"],
                                                                 self.ref_rect_coords["y"], width=2, outline='red'))
        def drag_rect(e):
            # update the coordinates from the event
            self.ref_rect_coords["x2"] = e.x
            self.ref_rect_coords["y2"] = e.y
            # Change the coordinates of the last created line to the new coordinates
            self.photo_canvas.coords(self.rects_list[-1], self.ref_rect_coords["x"],
                                     self.ref_rect_coords["y"],self.ref_rect_coords["x2"],self.ref_rect_coords["y2"])
        def tracking_zone_screen():
            self.photo_canvas = tk.Canvas(self.master, width = int(ww*0.7), height = int(ww*0.7*0.75), bg = 'black')
            self.photo_canvas.place(x=int(0.05*ww),y=int(0.05*hw), anchor= 'nw')
            self.img = Image.open(self.photo_path+'\\'+self.start_photo).resize((int(ww*0.7),int(ww*0.7*0.75)), Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(self.img)
            self.photo_canvas.create_image(0,0, image=self.img, anchor='nw')
            self.canvas.update()

            self.rects_list.append(self.photo_canvas.create_rectangle(self.ref_rect_coords["x"],
                                                                 self.ref_rect_coords["y"],
                                                                 self.ref_rect_coords["x2"],
                                                                 self.ref_rect_coords["y2"], width=2, outline='red'))

            tk.Label(self.master, text= "Click and drag on image to select"+
                     "\narea of image to track before\nclicking next.",justify=tk.LEFT).place(relx= .8,rely= .15,anchor= tk.W)
            self.photo_canvas.bind("<ButtonPress-1>", click_rect)
            self.photo_canvas.bind("<B1-Motion>", drag_rect)

        def update_margins():
            self.ext_area_yb = float(self.vmarg.get())
            self.ext_area_xr = float(self.hmarg.get())

            self.ext_area_yt = np.min(((self.tz_y0-self.tz_h/2),self.ext_area_yb))
            self.ext_area_xl = np.min(((self.tz_x0-self.tz_w/2),self.ext_area_xr))
            self.photo_canvas.delete(self.rects)
            self.rects = self.photo_canvas.create_rectangle(self.photo_canvas.winfo_width()*((self.tz_x0-self.tz_w/2)-self.ext_area_xl),
                                                self.photo_canvas.winfo_height()*((self.tz_y0-self.tz_h/2)-self.ext_area_yt),
                                                self.photo_canvas.winfo_width()*((self.tz_x0+self.tz_w/2)+self.ext_area_xr),
                                                self.photo_canvas.winfo_height()*((self.tz_y0+self.tz_h/2)+self.ext_area_yb),
                                                width=3, outline='blue')

        def tracking_margin_screen():
            self.photo_canvas = tk.Canvas(self.master, width = int(ww*0.7), height = int(ww*0.7*0.75), bg = 'black')
            self.photo_canvas.place(x=int(0.05*ww),y=int(0.05*hw), anchor= 'nw')
            self.img = Image.open(self.photo_path+'\\'+self.start_photo).resize((int(ww*0.7),int(ww*0.7*0.75)), Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(self.img)
            self.photo_canvas.create_image(0,0, image=self.img, anchor='nw')
            self.canvas.update()


            self.ext_area_yt = np.min(((self.tz_y0-self.tz_h/2),self.ext_area_yb))
            self.ext_area_xl = np.min(((self.tz_x0-self.tz_w/2),self.ext_area_xr))
            self.vmarg = tk.StringVar(self.master, value=str(self.ext_area_yb))
            self.hmarg = tk.StringVar(self.master, value=str(self.ext_area_xr))

            tk.Label(self.master, text= "Vertical search margin\n[fraction of image height]").place(relx= .8, rely= .65, anchor= tk.W)
            tk.Entry(self.master, width= 10,textvariable=self.vmarg).place(relx= .8, rely= .7, anchor= tk.W)
            tk.Label(self.master, text= "Horizontal search margin\n[fraction of image width]").place(relx= .8, rely= .75, anchor= tk.W)
            tk.Entry(self.master, width= 10,textvariable=self.hmarg).place(relx= .8, rely= .8, anchor= tk.W)
            tk.Button(self.master, text= "Update margins", command= update_margins,font=('bold')).place(relx= .8, rely= .85, anchor= 'w')


            self.photo_canvas.create_rectangle(self.photo_canvas.winfo_width()*(self.tz_x0-self.tz_w/2),
                                                            self.photo_canvas.winfo_height()*(self.tz_y0-self.tz_h/2),
                                                            self.photo_canvas.winfo_width()*(self.tz_x0+self.tz_w/2),
                                                            self.photo_canvas.winfo_height()*(self.tz_y0+self.tz_h/2),
                                                            width=2, outline='red')

            self.rects = self.photo_canvas.create_rectangle(self.photo_canvas.winfo_width()*((self.tz_x0-self.tz_w/2)-self.ext_area_xl),
                                                self.photo_canvas.winfo_height()*((self.tz_y0-self.tz_h/2)-self.ext_area_yt),
                                                self.photo_canvas.winfo_width()*((self.tz_x0+self.tz_w/2)+self.ext_area_xr),
                                                self.photo_canvas.winfo_height()*((self.tz_y0+self.tz_h/2)+self.ext_area_yb),
                                                width=3, outline='blue')
            tk.Label(self.master, text= "Update margins as a fraction of image"+
                     "\ndimensions (if needed) using the boxes"+
                     "\nbelow and the \"Update margins\" button.",justify=tk.LEFT).place(relx= .8,rely= .15,anchor= tk.W)


        def update_plots():
            plt_data = self.x_data-self.x_data[0]
            self.plotted_data.set_data(self.load_data[:len(plt_data)],plt_data)
            self.ax.set_xlim(np.min(self.load_data[:len(plt_data)])-1e-10,np.max(self.load_data[:len(plt_data)]))
            self.ax.set_ylim(np.min(plt_data)-1e-10,np.max(plt_data))
            self.fig.tight_layout()

            self.plot_canvas.draw()

            self.img = Image.open(self.current_picture).resize((int(ww*0.5),int(ww*0.5*0.75)), Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(self.img)
            self.photo_canvas.create_image(0,0, image=self.img, anchor='nw')
            self.rects = self.photo_canvas.create_rectangle(self.photo_canvas.winfo_width()*(self.tz_x-self.tz_w/2),
                                                            self.photo_canvas.winfo_height()*(self.tz_y-self.tz_h/2),
                                                            self.photo_canvas.winfo_width()*(self.tz_x+self.tz_w/2),
                                                            self.photo_canvas.winfo_height()*(self.tz_y+self.tz_h/2),
                                                            width=2, outline='red')
            self.photo_canvas.create_rectangle(self.photo_canvas.winfo_width()*((self.tz_x-self.tz_w/2)-self.ext_area_xl),
                                                self.photo_canvas.winfo_height()*((self.tz_y-self.tz_h/2)-self.ext_area_yt),
                                                self.photo_canvas.winfo_width()*((self.tz_x+self.tz_w/2)+self.ext_area_xr),
                                                self.photo_canvas.winfo_height()*((self.tz_y+self.tz_h/2)+self.ext_area_yb),
                                                width=3, outline='blue')


            self.photo_canvas.update()
            self.canvas.update()
        def np_fftconvolve(x, y,mode='valid'):
            """
            x and y must be real 2-d numpy arrays.

            mode must be "full" or "valid".
            """
            x_shape = np.array(x.shape)
            y_shape = np.array(y.shape)
            z_shape = x_shape + y_shape - 1
            z = ifft2(fft2(x, z_shape) * fft2(y, z_shape)).real

            if mode == "valid":
                # To compute a valid shape, either np.all(x_shape >= y_shape) or
                # np.all(y_shape >= x_shape).
                valid_shape = x_shape - y_shape + 1
                if np.any(valid_shape < 1):
                    valid_shape = y_shape - x_shape + 1
                    if np.any(valid_shape < 1):
                        raise ValueError("empty result for valid shape")
                start = (z_shape - valid_shape) // 2
                end = start + valid_shape
                z = z[start[0]:end[0], start[1]:end[1]]

            return z
        def run_analysis():
            self.tz_x = self.tz_x0
            self.tz_y = self.tz_y0

            self.subpixel_tracking=True

            if self.subpixel_tracking==True:
                # Set number of pixels (centre+/-n) for subpixel analysis
                n_subpix=1

                # CREATE GRID FOR FITTING POLYNOMIAL
                subpix_x = np.array(list(np.arange(-n_subpix,n_subpix+1))*(1+(n_subpix*2)))
                subpix_y = np.arange(-n_subpix,n_subpix+1).repeat(1+(n_subpix*2))
                G = poly_matrix(subpix_x, subpix_y, order=2)

                # CREATE GRID FOR INTERPOLATION OF MAX OF POLYNOMIAL
                nx, ny = 100, 100
                xx, yy = np.meshgrid(np.arange(-n_subpix,n_subpix, 1/(nx)),
                                      np.arange(-n_subpix,n_subpix, 1/(ny)))
                GG = poly_matrix(xx.ravel(), yy.ravel(), order=2)


            # Get list of photos in folder
            flnms = [file for file in glob.glob(self.photo_path+"/*.jpg", recursive = True)]
            indx = flnms.index(self.photo_path+"\\"+self.start_photo)
            flnms = flnms[indx:]

            ts= []
            timestamps = []
            for fl in flnms:
                tmp = Image.open(fl)._getexif()[36867]
                tmp = time.mktime(datetime.datetime.strptime(tmp, "%Y:%m:%d %H:%M:%S").timetuple())
                ts.append(tmp)
                timestamps.append(datetime.datetime.fromtimestamp(tmp))

            self.time_data = np.array(ts)-ts[0]
            del(fl,tmp)

            try:
                str2date = lambda x: datetime.datetime.strptime(x.decode("utf-8"), '%d/%m/%Y %H:%M:%S').timestamp()

                load_dat = np.genfromtxt(self.load_path,delimiter=',',dtype=None,usecols=[0,1,2],
                                         names=('Timestamp','ms','Load_val'),converters = {0: str2date})
                load_ts = load_dat['Timestamp']
            except:
                try:
                    str2date = lambda x: datetime.datetime.strptime(x.decode("utf-8"), '%m/%d/%Y %H:%M').timestamp()

                    load_dat = np.genfromtxt(self.load_path,delimiter=',',dtype=None,usecols=[0,1,2],
                                             names=('Timestamp','ms','Load_val'),converters = {0: str2date})
                except:
                    str2date = lambda x: datetime.datetime.strptime(x.decode("utf-8"), '%d/%m/%Y %H:%M').timestamp()

                    load_dat = np.genfromtxt(self.load_path,delimiter=',',dtype=None,usecols=[0,1,2],
                                             names=('Timestamp','ms','Load_val'),converters = {0: str2date})
                load_ts = load_dat['Timestamp']
                load_ts = load_ts+load_dat['ms']/1000


            # Linearly interpolate loads to timestamps of photos
            load_ts = np.interp(self.time_data,load_dat['ms']/1000,load_ts)
            load_ts = [datetime.datetime.fromtimestamp(i) for i in load_ts]
            self.load_data = np.interp(self.time_data,load_dat['ms']/1000,load_dat['Load_val'])
            # Linearly interpolate loads to timestamps of photos
            # Linearly interpolate loads to timestamps of photos

            self.previous_picture = self.photo_path+'\\'+self.start_photo

            # Load initial image for analysis
            self.base_frame = np.asarray(Image.open(self.previous_picture).convert('L'))
            # Extract original pixel info from base frame
            box = np.array([[int(np.round(len(self.base_frame)*(self.tz_y0-(self.tz_h/2)))),
                                  int(np.round(len(self.base_frame)*(self.tz_y0+(self.tz_h/2))))],
                                 [int(np.round(len(self.base_frame[0])*(self.tz_x0-(self.tz_w/2)))),
                                  int(np.round(len(self.base_frame[0])*(self.tz_x0+(self.tz_w/2))))]])



            self.ext_area_yt = np.min(((self.tz_y0-self.tz_h/2),self.ext_area_yb))
            self.ext_area_xl = np.min(((self.tz_x0-self.tz_w/2),self.ext_area_xr))

            search_box = np.array([[int(np.round(len(self.base_frame)*((self.tz_y0-self.tz_h/2)-self.ext_area_yt))),
                                          int(np.round(len(self.base_frame)*((self.tz_y0+self.tz_h/2)+self.ext_area_yb)))],
                                         [int(np.round(len(self.base_frame[0])*((self.tz_x0-self.tz_w/2)-self.ext_area_xl))),
                                          int(np.round(len(self.base_frame[0])*((self.tz_x0+self.tz_w/2)+self.ext_area_xr)))]])

            self.x_data = []
            self.y_data = []
            for i in range(1,10):#len(flnms)):


                pix0 = self.base_frame[box[0,0]:box[0,1],box[1,0]:box[1,1]]
                pix0 = pix0-np.mean(pix0)

                self.x_data = self.x_data+[self.tz_x*self.scale_factor]
                self.y_data = self.y_data+[self.tz_y*self.scale_factor]

                self.current_picture = flnms[i]
                new_frame = np.asarray(Image.open(flnms[i]).convert('L'))

                pix1 = new_frame[search_box[0,0]:search_box[0,1],search_box[1,0]:search_box[1,1]]

                corr = np_fftconvolve(pix1-np.mean(pix1),pix0[::-1,::-1],mode='valid')


                corr_max = np.unravel_index(corr.argmax(), corr.shape)

                if self.subpixel_tracking==True:
                    # Take surrounding pixels for subpixel analysis
                    try:
                        sub_pix0 = corr[corr_max[0]-n_subpix:corr_max[0]+n_subpix+1,
                                        corr_max[1]-n_subpix:corr_max[1]+n_subpix+1]


                        # Use least-squares to fit guassian to the subpixels
                        m = np.linalg.lstsq(G, sub_pix0.ravel()-np.mean(sub_pix0),rcond=1)[0]

                        # Reshape results
                        zz = np.reshape(np.dot(GG, m), xx.shape)
                        # Identify peak of Gaussian
                        sub_corr_max = np.array(np.unravel_index(zz.argmax(), zz.shape)).astype(float)

                        # Scale to be percentage of image height
                        sub_corr_max[0] = sub_corr_max[0]/nx
                        sub_corr_max[1] = sub_corr_max[1]/ny
                        sub_corr_max = sub_corr_max-(n_subpix)
                        corr_max=corr_max+sub_corr_max
                    except:
                        print('Edge of image')


                self.tz_x = self.tz_x+((corr_max[1]/len(new_frame[0]))-self.ext_area_xr)
                if self.tz_x<self.tz_w/2:
                    self.tz_x=self.tz_w/2
                elif self.tz_x>1:
                    self.tz_x=1
                self.tz_y = self.tz_y+((corr_max[0]/len(new_frame))-self.ext_area_yt)
                if self.tz_y<self.tz_h/2:
                    self.tz_y=self.tz_h/2
                elif self.tz_y>1:
                    self.tz_y=1


                box = np.array([[int(np.round(len(self.base_frame)*(self.tz_y-self.tz_h/2))),
                                      int(np.round(len(self.base_frame)*(self.tz_y+self.tz_h/2)))],
                                     [int(np.round(len(self.base_frame[0])*(self.tz_x-self.tz_w/2))),
                                      int(np.round(len(self.base_frame[0])*(self.tz_x+self.tz_w/2)))]])

                self.ext_area_yt = np.min(((self.tz_y-self.tz_h/2),self.ext_area_yb))
                self.ext_area_xl = np.min(((self.tz_x-self.tz_w/2),self.ext_area_xr))

                search_box = np.array([[int(np.round(len(self.base_frame)*((self.tz_y-self.tz_h/2)-self.ext_area_yt))),
                                              int(np.round(len(self.base_frame)*((self.tz_y+self.tz_h/2)+self.ext_area_yb)))],
                                             [int(np.round(len(self.base_frame[0])*((self.tz_x-self.tz_w/2)-self.ext_area_xl))),
                                              int(np.round(len(self.base_frame[0])*((self.tz_x+self.tz_w/2)+self.ext_area_xr)))]])



                update_plots()
                self.base_frame=new_frame

            headers = ['Load cell timestamp','GoPro timestamp','Load cell output [kN]','Horizontal displacement [mm]','Vertical displacement [mm]']
            output_data = [list(load_ts[:len(self.x_data)]),
                                         list(timestamps[:len(self.x_data)]),
                                         list(self.load_data[:len(self.x_data)]),
                                         list(self.x_data-self.x_data[0]),
                                         list(self.y_data-self.y_data[0]),
                                        ]
            output_data = np.array(output_data).T
            print(np.shape(output_data))



            with open(self.output_path+'/'+self.output_file,'w') as f:
                for item in headers:
                    f.write(item + ',')
                f.write('\n')
                for sublist in output_data:
                    for item in sublist:
                        try:
                            f.write(str(item) + ',')
                        except:
                            f.write(item.strftime("%Y-%m-%d %H:%M:%S.%f"))
                    f.write('\n')

            # output_data.to_csv(self.output_path+'/'+self.output_file, index=False, header=True)
            tk.Label(self.master, text= "Analysis has completed and data has been saved successfully to output file!").place(relx= .5,
                                                                                                    rely= .1, anchor= tk.CENTER)

        def output_plots_screen():
            self.photo_canvas = tk.Canvas(self.master, width = int(ww*0.5), height = int(ww*0.5*0.75), bg = 'black')
            self.photo_canvas.place(x=int(0.95*ww),y=int(0.5*hw), anchor= 'e')
            self.img = Image.open(self.photo_path+'\\'+self.start_photo).resize((int(ww*0.5),int(ww*0.5*0.75)), Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(self.img)
            self.photo_canvas.create_image(0,0, image=self.img, anchor='nw')

            self.photo_canvas.update()
            self.rects = self.photo_canvas.create_rectangle(self.photo_canvas.winfo_width()*(self.tz_x0-self.tz_w/2),
                                                            self.photo_canvas.winfo_height()*(self.tz_y0-self.tz_h/2),
                                                            self.photo_canvas.winfo_width()*(self.tz_x0+self.tz_w/2),
                                                            self.photo_canvas.winfo_height()*(self.tz_y0+self.tz_h/2),
                                                            width=2, outline='red')
            self.photo_canvas.create_rectangle(self.photo_canvas.winfo_width()*((self.tz_x0-self.tz_w/2)-self.ext_area_xl),
                                                self.photo_canvas.winfo_height()*((self.tz_y0-self.tz_h/2)-self.ext_area_yt),
                                                self.photo_canvas.winfo_width()*((self.tz_x0+self.tz_w/2)+self.ext_area_xr),
                                                self.photo_canvas.winfo_height()*((self.tz_y0+self.tz_h/2)+self.ext_area_yb),
                                                width=3, outline='blue')
            self.fig = plt.Figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_ylabel('Displacement [mm]')
            self.ax.set_xlabel('Load [N]')
            self.x_data = []
            self.y_data = []
            self.load_data = []
            self.time_data = []
            self.plotted_data, = self.ax.plot(self.time_data,self.x_data,'kx')
            self.fig.tight_layout()

            self.plot_canvas = FigureCanvasTkAgg(self.fig,master=self.master)
            self.plot_canvas.get_tk_widget().place(width=0.45*ww,relx= .0, rely= .5,anchor= tk.W)
            # self.plot_canvas.pack(side="top",fill='both',expand=True)
            self.plot_canvas.draw()

            tk.Button(self.master, text= "Run", command= run_analysis).place(relx= .95, rely= .95, anchor= tk.CENTER)
            tk.Label(self.master, text= "Press \"Run\" button to start analysis."
                     ,justify=tk.LEFT).place(relx= .8,rely= .125,anchor= tk.W)
            self.canvas.update()

        input_information_screen()


root = tk.Tk()
GraphPlot(root)
root.mainloop()