# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:38:14 2022

@author: s1879083
"""
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import ImageTk, Image, ImageGrab
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

class DIC_app(ttk.Frame):
    ''' Advanced zoom of the image '''
    def __init__(self, master):
        ''' Initialize the main Frame '''
        ttk.Frame.__init__(self, master=master)
        self.master.title('Digital Image Correlation')
        self.master = master

        # Setting the Tkinter window and the canvas in place
        ws = master.winfo_screenwidth()
        hs = master.winfo_screenheight()
        self.ww = ws*0.75
        self.hw = hs*0.75


        self.canvas = tk.Canvas(self.master, width = self.ww, height = self.hw, bg = 'grey')
        self.canvas.grid()
        self.master.update()
        self.w = self.canvas.winfo_width()
        self.h = self.canvas.winfo_height()
        self.canvas.focus_set()


        tk.Button(self.master, text= "Close", command= self.close_app).place(relx= .95, rely= .05, anchor= tk.CENTER)
        tk.Button(self.master, text= "Previous", command= self.previous_screen).place(relx= .9, rely= .95, anchor= tk.CENTER)
        tk.Button(self.master, text= "Next", command= self.next_screen).place(relx= .95, rely= .95, anchor= tk.CENTER)


        # Variables for GUI
        self.screen_number = 0
        self.error_state = 0
        # Variables used in input information screen
        self.photo_path = r'C:\Users\S1879083\OneDrive - University of Edinburgh\RC3 Madagascar\Data from fieldwork Madagascar\Data for DIC-Load analysis\Data from Rafik\Frame test--DIC@5s\Test28----@2-75'
        self.start_photo = 'G0538359.JPG'
        self.load_path = r'C:\Users\S1879083\OneDrive - University of Edinburgh\RC3 Madagascar\Data from fieldwork Madagascar\Data for DIC-Load analysis\Data from Rafik\Frame test--DIC@5s\Test28----@2-75\@2---OBLB75.csv'
        self.output_path = r'C:\Users\S1879083\OneDrive - University of Edinburgh\Edi_Python\Video_tracking\Video_tracking_app\v1'
        self.output_file = 'Output_data.csv'

        # Variables used in reference length screen
        self.line_marker_count = 0
        self.reference_length = 1000

        # Variables used in tracking area co-ordinates
        self.track_areas = []
        self.track_areas_labels = []
        self.track_areas_co_ords0 = []

        # Variables used in margins
        self.margin_x = 200
        self.margin_y = 75
        self.margin_areas = []

        # Variables used in output plots

        self.start_up_screen()
    def close_app(self):
        self.master.destroy()
        sys.exit("Programme closed")
    def create_window(self):
        self.error_window = tk.Toplevel(root)
        self.error_window.geometry("500x200")
        tk.Label(self.error_window,
          text =self.error_message).place(relx= .5, rely= .5, anchor= tk.CENTER)

    def next_screen(self):
        if self.screen_number==0:
            # self.photo_path = self.photo_path.get()
            # self.start_photo = self.start_photo.get()
            if (self.start_photo[-4:]!='.jpg') and (self.start_photo[-4:]!='.JPG'):
                self.start_photo=self.start_photo+'.JPG'

            # self.load_path = self.load_path.get()
            # self.output_path = self.output_path.get()
            self.output_file = self.output_file.get()
            if self.output_file[-4:]!='.csv':
                self.output_file=self.output_file+'.csv'


            try:
                # Check if input folder and photos exist
                self.flnms = [file for file in glob.glob(self.photo_path+"/*.jpg", recursive = True)]
                self.error_message = 'Folder with input photos\ndoes not exist!'
                glob.glob(self.photo_path)[0]

                self.error_message = 'Start photo does\n not exist!'
                indx = self.flnms.index(self.photo_path+"\\"+self.start_photo)
                self.flnms = self.flnms[indx:]

                # Check if output folder exists
                self.error_message = 'Output folder does\nnot exist!'
                glob.glob(self.output_path)[0]
                self.error_state=0
                self.info_canvas.destroy()

                self.reference_length_screen()
            except:
                self.create_window()
                self.error_state=1
                self.start_up_screen()


        elif self.screen_number==1:
            try:

                self.reference_length = self.reference_length.get()
                self.reference_line_length = (abs(self.rl0_x - self.rl1_x) ** 2 +
                                              abs(self.rl0_y - self.rl1_y) ** 2) ** (1 / 2)
                if self.reference_line_length==0:
                    raise Exception('Reference length equals zero!')
                self.scale_factor = float(self.reference_length)/self.reference_line_length

                self.photo_canvas.destroy()
                self.ref_length_canvas.destroy()


                self.error_state=0

                self.track_area_screen()

            except:

                self.error_message = 'Please select reference length!'
                self.create_window()
                self.error_state=1
                self.photo_canvas.destroy()
                self.ref_length_canvas.destroy()
                self.reference_length_screen()

        elif self.screen_number==2:
            if len(self.track_areas_co_ords0)>0:
                self.photo_canvas.destroy()
                self.track_area_canvas.destroy()
                for co in self.track_areas_co_ords0:
                    if co[1]>co[3]:
                        co[1],co[3] = co[3],co[1]
                    if co[0]>co[2]:
                        co[0],co[2] = co[2],co[0]

                self.error_state=0
                self.track_margins_screen()
            else:
                self.error_message = 'Number of tracked areas\nmust be greater than zero!'
                self.create_window()
                self.error_state=1
                self.track_area_screen()

        elif self.screen_number==3:
            self.img_tx = 0
            self.img_ty = 0
            self.img_bx, self.img_by = self.image_base.size
            self.scale_image()
            x=root.winfo_rootx()+self.photo_canvas.winfo_x()
            y=root.winfo_rooty()+self.photo_canvas.winfo_y()
            x1=x+self.photo_canvas.winfo_width()
            y1=y+self.photo_canvas.winfo_height()
            ImageGrab.grab().crop((x,y,x1,y1)).save(self.output_path+'/Referance_image_'+self.output_file[:-4]+'.png')
            self.photo_canvas.destroy()
            self.track_margins_canvas.destroy()
            self.outputs_screen()

        if (self.screen_number<4) and (self.error_state==0):
            self.screen_number+=1


    def previous_screen(self):
        if self.screen_number>=1:
            self.screen_number+=-1

            if self.screen_number==0:
                self.reference_length = self.reference_length.get()
                self.photo_canvas.destroy()
                self.ref_length_canvas.destroy()
                self.start_up_screen()
            elif self.screen_number==1:
                self.photo_canvas.destroy()
                self.track_area_canvas.destroy()
                self.reference_length_screen()
            elif self.screen_number==2:
                self.photo_canvas.destroy()
                self.track_margins_canvas.destroy()
                self.track_area_screen()
            elif self.screen_number==3:
                self.outputs_canvas.destroy()
                self.run_button.destroy()
                self.track_margins_screen()

    def outputs_screen(self):
        self.run_button = tk.Button(self.master, text= "Run ", command= self.run_analysis)
        self.run_button.place(relx= .95, rely= .95, anchor= tk.CENTER)
        self.outputs_canvas = tk.Canvas(self.master, width = self.ww, height = 0.8*self.hw, bg = 'grey')
        self.canvas.create_window(0,0.1*self.hw,anchor='nw', window=self.outputs_canvas)
        # Add image to outputs
        self.photo_canvas_width = 0.5*self.ww
        self.photo_canvas_height = 0.75*0.5*self.ww
        self.photo_canvas = tk.Canvas(self.master, width = self.photo_canvas_width,
                                      height = self.photo_canvas_height, bg = 'white')

        self.outputs_canvas.create_window(0,  0.4*self.hw,anchor='w', window=self.photo_canvas)
        self.photo_canvas.update()  # wait till canvas is created


        self.image_base = Image.open(self.photo_path+'\\'+self.start_photo)  # open image
        # Initial display area of image
        self.img_tx = 0
        self.img_ty = 0
        self.img_bx, self.img_by = self.image_base.size

        # Crop image to fit display
        self.img_r0 = self.image_base.crop((self.img_tx,self.img_ty,self.img_bx,self.img_by))
        self.r0_width,self.r0_height = self.img_r0.size

        self.image_scale = np.min((self.photo_canvas_width/self.r0_width,
                                    self.photo_canvas_height/self.r0_height))


        self.img_r = self.img_r0.resize((int(self.r0_width*self.image_scale),int(self.r0_height*self.image_scale)), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img_r)
        self.displayed_image = self.photo_canvas.create_image(0, 0.75*0.25*self.ww, image=self.img, anchor='w')

        self.update_track_area()

        # Add outputs plot
        px = 1/plt.rcParams['figure.dpi']
        self.fig = plt.Figure(figsize=(self.photo_canvas_width*px,self.photo_canvas_height*px),dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Displacement [mm]')
        self.ax.set_ylabel('Load [kN]')
        self.output_data = []
        self.load_data = []
        self.time_data = []

        # Add option menu
        self.options = [['Tracked area '+str(i+1)+' - Horizontal displacement [mm]',
                         'Tracked area '+str(i+1)+' - Vertical displacement [mm]'] for i in range(len(self.track_areas))]
        self.options = [x for xs in self.options for x in xs]
        self.option_var = 0
        self.option_menu = ttk.OptionMenu(
            self.outputs_canvas,
            tk.StringVar(self,self.option_var),
            self.options[0],
            *self.options,
            command=self.option_changed)
        self.option_menu.place(width=0.5*self.ww,relx= .5, rely= .95,anchor= tk.W)

        self.plotted_data, = self.ax.plot(self.load_data,self.output_data,'kx-',label=self.options[0])
        self.ax.legend(loc=1)
        self.fig.tight_layout()

        self.plot_canvas = FigureCanvasTkAgg(self.fig,master=self.outputs_canvas)
        self.plot_canvas.get_tk_widget().place(width=0.5*self.ww,relx= .5, rely= .5,anchor= tk.W)



        self.plot_canvas.draw()

    def option_changed(self,new_option):
        self.option_var = self.options.index(new_option)
        self.plotted_data.set_label(self.options[self.option_var])
        # self.ax.legend.remove()
        self.ax.legend(loc=1)
        if len(self.output_data)>0:
            plt_data = np.array(self.output_data)
            plt_data = (plt_data-plt_data[0])*self.scale_factor
            try:
                self.plotted_data.set_data(plt_data[:,self.option_var],self.load_data[:len(plt_data)])
                self.ax.set_xlim(np.min(plt_data[:,self.option_var])-1e-10,np.max(plt_data[:,self.option_var]))

            except:
                self.plotted_data.set_data(plt_data[self.option_var],self.load_data[:len(plt_data)])
                self.ax.set_xlim(np.min(plt_data[self.option_var])-1e-10,np.max(plt_data[self.option_var]))

            self.ax.set_ylim(np.min(self.load_data[:len(plt_data)])-1e-10,np.max(self.load_data[:len(plt_data)]))
            self.fig.tight_layout()
        self.plot_canvas.draw()

    def poly_matrix(self,x, y, order=2):
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

    def np_fftconvolve(self, x, y,mode='valid'):
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

    def run_analysis(self):
        self.track_areas_co_ords = [np.copy(i) for i in self.track_areas_co_ords0]
        self.subpixel_tracking=True

        if self.subpixel_tracking==True:
            # Set number of pixels (centre+/-n) for subpixel analysis
            n_subpix=1

            # CREATE GRID FOR FITTING POLYNOMIAL
            subpix_x = np.array(list(np.arange(-n_subpix,n_subpix+1))*(1+(n_subpix*2)))
            subpix_y = np.arange(-n_subpix,n_subpix+1).repeat(1+(n_subpix*2))
            G = self.poly_matrix(subpix_x, subpix_y, order=2)

            # CREATE GRID FOR INTERPOLATION OF MAX OF POLYNOMIAL
            nx, ny = 100, 100
            xx, yy = np.meshgrid(np.arange(-n_subpix,n_subpix, 1/(nx)),
                                  np.arange(-n_subpix,n_subpix, 1/(ny)))
            GG = self.poly_matrix(xx.ravel(), yy.ravel(), order=2)




        ts= []
        timestamps = []
        for fl in self.flnms:
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
                try:
                    str2date = lambda x: datetime.datetime.strptime(x.decode("utf-8"), '%d/%m/%Y %H:%M').timestamp()

                    load_dat = np.genfromtxt(self.load_path,delimiter=',',dtype=None,usecols=[0,1,2],
                                             names=('Timestamp','ms','Load_val'),converters = {0: str2date})
                except:
                    str2date = lambda x: datetime.datetime.strptime(x.decode("utf-8"), '%m/%d/%Y %H:%M:%S %p').timestamp()

                    load_dat = np.genfromtxt(self.load_path,delimiter=',',dtype=None,usecols=[0,1,2],
                                             names=('Timestamp','ms','Load_val'),converters = {0: str2date})
            load_ts = load_dat['Timestamp']
            load_ts = load_ts+load_dat['ms']/1000


        # Linearly interpolate loads to timestamps of photos
        load_ts = np.interp(self.time_data,load_dat['ms']/1000,load_ts)
        load_ts = [datetime.datetime.fromtimestamp(i) for i in load_ts]
        self.load_data = np.interp(self.time_data,load_dat['ms']/1000,load_dat['Load_val'])

        # Load initial image for analysis
        self.previous_picture = self.flnms[0]
        self.base_frame = np.asarray(Image.open(self.previous_picture).convert('L'))

        self.output_data = []
        for i in range(1,len(self.flnms)):
            # Load new image for analysis
            self.current_picture = self.flnms[i]
            new_frame = np.asarray(Image.open(self.current_picture).convert('L'))

            # Empty lists for storing outputs
            self.output_data = self.output_data+[[it for lst in self.track_areas_co_ords for it in lst[:2]]]


            for j in range(len(self.track_areas_co_ords)):
                box = self.track_areas_co_ords[j]
                # Extract original pixel info from base frame
                pix0 = self.base_frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                pix0 = pix0-np.mean(pix0)


                tmp_margin_xl = self.margin_x
                tmp_margin_xr = self.margin_x
                tmp_margin_yt = self.margin_y
                tmp_margin_yb = self.margin_y

                if int(box[1]-tmp_margin_yt)<0:
                    tmp_margin_yt = int(box[1])
                if int(box[0]-tmp_margin_xl)<0:
                    tmp_margin_xl = int(box[0])

                # Extract pixel info from new frame
                pix1 = new_frame[int(box[1]-tmp_margin_yt):int(box[3]+tmp_margin_yb),
                                 int(box[0]-tmp_margin_xl):int(box[2]+tmp_margin_xr)]

                pix1 = pix1-np.mean(pix1)

                corr = self.np_fftconvolve(pix1,pix0[::-1,::-1],mode='valid')

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

                # Update co-ordinates
                corr_max = list(corr_max)
                corr_max[0] = corr_max[0]-tmp_margin_yt
                corr_max[1] = corr_max[1]-tmp_margin_xl

                if self.track_areas_co_ords[j][0]+corr_max[1]>1:
                    self.track_areas_co_ords[j][0]+=corr_max[1]
                    self.track_areas_co_ords[j][2]+=corr_max[1]

                if self.track_areas_co_ords[j][1]+corr_max[0]>1:
                    self.track_areas_co_ords[j][1]+=corr_max[0]
                    self.track_areas_co_ords[j][3]+=corr_max[0]

            self.update_plots()
            self.base_frame=new_frame


        output_dat = np.array(self.output_data)
        output_dat = output_dat-output_dat[0]
        output_dat = output_dat*self.scale_factor
        headers = ['Load cell timestamp','GoPro timestamp','Load cell output [kN]'
                   ]
        output_data = [list(load_ts[:len(output_dat)]),
                                     list(timestamps[:len(output_dat)]),
                                     list(self.load_data[:len(output_dat)])
                                    ]
        for ik in range(len(self.track_areas_co_ords)):
            headers = headers+self.options
            output_data = output_data+[list(output_dat[:,(ik*2)]),list(output_dat[:,(ik*2)+1])]

        output_data = np.array(output_data).T
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
        tk.Label(self.outputs_canvas, text= "Analysis has completed and data has been saved successfully to output file!").place(relx= .5,
                                                                                                rely= .01, anchor= 'n')

    def update_plots(self):
        self.image_base = Image.open(self.current_picture)  # open image

        self.r0_width,self.r0_height = self.image_base.size

        self.image_scale = np.min((self.photo_canvas_width/self.r0_width,
                                    self.photo_canvas_height/self.r0_height))


        self.img_r = self.image_base.resize((int(self.r0_width*self.image_scale),int(self.r0_height*self.image_scale)), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img_r)
        self.displayed_image = self.photo_canvas.create_image(0,0.75*0.25*self.ww, image=self.img, anchor='w')

        for ta in self.track_areas:
            self.photo_canvas.delete(ta)
        for tal in self.track_areas_labels:
            self.photo_canvas.delete(tal)

        self.track_areas = []
        self.track_areas_labels = []
        for co in self.track_areas_co_ords:
            self.track_areas.append(self.photo_canvas.create_rectangle(co[0]*self.image_scale,
                                                                       co[1]*self.image_scale,
                                                                       co[2]*self.image_scale,
                                                                       co[3]*self.image_scale,
                                                                        width=2,outline='red'))
            self.track_areas_labels.append(self.photo_canvas.create_text((co[0])*self.image_scale,
                                                                        (co[1])*self.image_scale,
                                                                        text=str(len(self.track_areas)),anchor='se',
                                                                        fill='red',font='bold'))

        self.photo_canvas.update()

        # Update plots
        plt_data = np.array(self.output_data)
        plt_data = (plt_data-plt_data[0])*self.scale_factor
        try:
            self.plotted_data.set_data(plt_data[:,self.option_var],self.load_data[:len(plt_data)])
            self.ax.set_xlim(np.min(plt_data[:,self.option_var])-1e-10,np.max(plt_data[:,self.option_var]))

        except:
            self.plotted_data.set_data(plt_data[self.option_var],self.load_data[:len(plt_data)])
            self.ax.set_xlim(np.min(plt_data[self.option_var])-1e-10,np.max(plt_data[self.option_var]))

        self.ax.set_ylim(np.min(self.load_data[:len(plt_data)])-1e-10,np.max(self.load_data[:len(plt_data)]))
        self.fig.tight_layout()

        self.plot_canvas.draw()

    def browsefunc_photo(self):
        filename = tk.filedialog.askopenfilename()
        self.photo_path = '/'.join(filename.split('/')[:-1])
        self.start_photo = filename.split('/')[-1]
        self.photo_label.config(text=filename)

    def browsefunc_load_cell(self):
        filename = tk.filedialog.askopenfilename()
        self.load_path = filename
        self.lc_label.config(text=filename)

    def browsefunc_output_path(self):
        filename = tk.filedialog.askdirectory()
        self.output_path = filename
        self.output_path_label.config(text=filename)

    def start_up_screen(self):
        self.info_canvas = tk.Canvas(self.master, width = self.ww, height = 0.8*self.hw, bg = 'grey')
        self.canvas.create_window(0,0.1*self.hw,anchor='nw', window=self.info_canvas)

        self.output_file = tk.StringVar(self.master,self.output_file)

        tk.Label(self.info_canvas, text= "Browse to select:"+
                 "\n-the first photo to use in the analysis,"+
                 "\n-the load cell data CSV file,"+
                 "\n-the folder where the output data should be saved,"+
                 "\n and add the name of the output file (including file extension)",justify=tk.LEFT).place(relx= .05,rely= .15,
                                                                                                    anchor= tk.W)


        tk.Label(self.info_canvas,text='Select first photo to use').place(relx= .05, rely= .3, anchor= tk.W)
        tk.Button(self.info_canvas, text= "Browse",
                  command= self.browsefunc_photo).place(relx= 0.05, rely= .35, anchor= 'w')
        self.photo_label = tk.Label(self.info_canvas,text=self.start_photo)
        self.photo_label.place(relx= .1, rely= .35, anchor= tk.W)

        tk.Label(self.info_canvas,text='Select load cell data').place(relx= .05, rely= .45, anchor= tk.W)
        tk.Button(self.info_canvas, text= "Browse",
                  command= self.browsefunc_load_cell).place(relx= 0.05, rely= .5, anchor= 'w')
        self.lc_label = tk.Label(self.info_canvas,text=self.load_path)
        self.lc_label.place(relx= .1, rely= .5, anchor= tk.W)


        tk.Label(self.info_canvas,text='Output folder path').place(relx= .05, rely= .6, anchor= tk.W)
        tk.Button(self.info_canvas, text= "Browse",
                  command= self.browsefunc_output_path).place(relx= 0.05, rely= .65, anchor= 'w')
        self.output_path_label = tk.Label(self.info_canvas,text=self.output_path)
        self.output_path_label.place(relx= .1, rely= .65, anchor= tk.W)

        tk.Label(self.info_canvas,text='Output filename').place(relx= .05, rely= .75, anchor= tk.W)
        tk.Entry(self.info_canvas, width= 150,textvariable=self.output_file).place(relx= .05, rely= .8, anchor= tk.W)

    def reference_length_screen(self):
        # Create canvas and put image on it
        self.ref_length_canvas = tk.Canvas(self.master, width = 0.25*self.ww, height = 0.8*self.hw, bg = 'grey')
        self.canvas.create_window(0.75*self.ww,0.1*self.hw,anchor='nw', window=self.ref_length_canvas)

        tk.Label(self.ref_length_canvas, text= "Click and drag on image to set known"+
                 "\nlength and update reference length box"+
                 "\nwith length in mm before clicking next.\n \n"+
                 "Zoom using scroll click/right click and drag.",justify=tk.LEFT).place(relx= .15,rely= .15,
                                                                                                    anchor= tk.W)
        tk.Label(self.ref_length_canvas, text= "Reference length [mm]").place(relx= .15, rely= .75, anchor= tk.W)
        self.reference_length = tk.StringVar(self.master, value=str(self.reference_length))
        tk.Entry(self.ref_length_canvas, width= 10,textvariable=self.reference_length).place(relx= .15, rely= .8, anchor= tk.W)

        self.photo_canvas_width = 0.75*self.ww
        self.photo_canvas_height = self.hw
        self.photo_canvas = tk.Canvas(self.master, width = self.photo_canvas_width,
                                      height = self.photo_canvas_height, bg = 'white')

        self.canvas.create_window(0, 0,anchor='nw', window=self.photo_canvas)
        self.photo_canvas.update()  # wait till canvas is created

        self.image_base = Image.open(self.photo_path+'\\'+self.start_photo)  # open image
        # Initial display area of image
        self.img_tx = 0
        self.img_ty = 0
        self.img_bx, self.img_by = self.image_base.size

        # Crop image to fit display
        self.img_r0 = self.image_base.crop((self.img_tx,self.img_ty,self.img_bx,self.img_by))
        self.r0_width,self.r0_height = self.img_r0.size

        self.image_scale = np.min((self.photo_canvas_width/self.r0_width,
                                    self.photo_canvas_height/self.r0_height))


        self.img_r = self.img_r0.resize((int(self.r0_width*self.image_scale),int(self.r0_height*self.image_scale)), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img_r)
        self.displayed_image = self.photo_canvas.create_image(0,0, image=self.img, anchor='nw')


        self.photo_canvas.bind("<ButtonPress-1>", self.add_ref_line)
        self.photo_canvas.bind("<B1-Motion>", self.drag_ref_line)


        self.photo_canvas.bind("<ButtonPress-2>", self.click_zoom)
        self.photo_canvas.bind("<B2-Motion>", self.drag_zoom)
        self.photo_canvas.bind("<ButtonRelease-2>", self.drag_release)
        self.photo_canvas.bind("<ButtonPress-3>", self.click_zoom)
        self.photo_canvas.bind("<B3-Motion>", self.drag_zoom)
        self.photo_canvas.bind("<ButtonRelease-3>", self.drag_release)
        self.update_ref_line()

    def track_area_screen(self):
        # Create canvas and put image on it
        self.track_area_canvas = tk.Canvas(self.master, width = 0.25*self.ww, height = 0.8*self.hw, bg = 'grey')
        self.canvas.create_window(0.75*self.ww,0.1*self.hw,anchor='nw', window=self.track_area_canvas)

        tk.Label(self.track_area_canvas, text= "Click and drag on image to select"+
                 "\narea(s) of image to track before\nclicking next.\n \n"+
                 "Zoom using scroll click/right click and drag.",justify=tk.LEFT).place(relx= .15,rely= .15,
                                                                                                    anchor= tk.W)
        tk.Button(self.track_area_canvas, text= "Clear all tracking areas",
                  command= self.clear_all_rects).place(relx= .15, rely= .7, anchor= 'w')
        tk.Button(self.track_area_canvas, text= "Clear last tracking area",
                  command= self.clear_last_rect).place(relx= .15, rely= .8, anchor= 'w')


        self.photo_canvas_width = 0.75*self.ww
        self.photo_canvas_height = self.hw
        self.photo_canvas = tk.Canvas(self.master, width = self.photo_canvas_width,
                                      height = self.photo_canvas_height, bg = 'white')

        self.canvas.create_window(0, 0,anchor='nw', window=self.photo_canvas)
        self.photo_canvas.update()  # wait till canvas is created

        self.image_base = Image.open(self.photo_path+'\\'+self.start_photo)  # open image
        # Initial display area of image
        self.img_tx = 0
        self.img_ty = 0
        self.img_bx, self.img_by = self.image_base.size

        # Crop image to fit display
        self.img_r0 = self.image_base.crop((self.img_tx,self.img_ty,self.img_bx,self.img_by))
        self.r0_width,self.r0_height = self.img_r0.size

        self.image_scale = np.min((self.photo_canvas_width/self.r0_width,
                                    self.photo_canvas_height/self.r0_height))


        self.img_r = self.img_r0.resize((int(self.r0_width*self.image_scale),int(self.r0_height*self.image_scale)), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img_r)
        self.displayed_image = self.photo_canvas.create_image(0,0, image=self.img, anchor='nw')


        self.photo_canvas.bind("<ButtonPress-1>", self.add_track_area)
        self.photo_canvas.bind("<B1-Motion>", self.drag_track_area)
        self.photo_canvas.bind("<ButtonRelease-1>", self.drag_track_area_release)


        self.photo_canvas.bind("<ButtonPress-2>", self.click_zoom)
        self.photo_canvas.bind("<B2-Motion>", self.drag_zoom)
        self.photo_canvas.bind("<ButtonRelease-2>", self.drag_release)
        self.photo_canvas.bind("<ButtonPress-3>", self.click_zoom)
        self.photo_canvas.bind("<B3-Motion>", self.drag_zoom)
        self.photo_canvas.bind("<ButtonRelease-3>", self.drag_release)
        self.update_track_area()


    def track_margins_screen(self):
        # Create canvas and put image on it
        self.track_margins_canvas = tk.Canvas(self.master, width = 0.25*self.ww, height = 0.8*self.hw, bg = 'grey')
        self.canvas.create_window(0.75*self.ww,0.1*self.hw,anchor='nw', window=self.track_margins_canvas)

        tk.Label(self.track_margins_canvas, text= "Update margins as a fraction of image"+
                 "\ndimensions (if needed) using the boxes"+
                 "\nbelow and the \"Update margins\" button.",justify=tk.LEFT).place(relx= .15,rely= .15,anchor= tk.W)
        self.hmarg =  tk.StringVar(self.master, value=str(self.margin_x))
        self.vmarg =  tk.StringVar(self.master, value=str(self.margin_y))
        tk.Label(self.track_margins_canvas, text= "Vertical search margin [Number of pixels]").place(relx= .15, rely= .5, anchor= tk.W)
        tk.Entry(self.track_margins_canvas, width= 10,textvariable=self.vmarg).place(relx= .15, rely= .55, anchor= tk.W)

        tk.Label(self.track_margins_canvas, text= "Horizontal search margin [Number of pixels]").place(relx= .15, rely= .6, anchor= tk.W)
        tk.Entry(self.track_margins_canvas, width= 10,textvariable=self.hmarg).place(relx= .15, rely= .65, anchor= tk.W)

        tk.Button(self.track_margins_canvas, text= "Update margins", command= self.update_margins).place(relx= .15, rely= .72, anchor= 'w')

        self.photo_canvas_width = 0.75*self.ww
        self.photo_canvas_height = self.hw
        self.photo_canvas = tk.Canvas(self.master, width = self.photo_canvas_width,
                                      height = self.photo_canvas_height, bg = 'white')

        self.canvas.create_window(0, 0,anchor='nw', window=self.photo_canvas)
        self.photo_canvas.update()  # wait till canvas is created

        self.image_base = Image.open(self.photo_path+'\\'+self.start_photo)  # open image
        # Initial display area of image
        self.img_tx = 0
        self.img_ty = 0
        self.img_bx, self.img_by = self.image_base.size

        # Crop image to fit display
        self.img_r0 = self.image_base.crop((self.img_tx,self.img_ty,self.img_bx,self.img_by))
        self.r0_width,self.r0_height = self.img_r0.size

        self.image_scale = np.min((self.photo_canvas_width/self.r0_width,
                                    self.photo_canvas_height/self.r0_height))


        self.img_r = self.img_r0.resize((int(self.r0_width*self.image_scale),int(self.r0_height*self.image_scale)), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img_r)
        self.displayed_image = self.photo_canvas.create_image(0,0, image=self.img, anchor='nw')


        self.photo_canvas.bind("<ButtonPress-2>", self.click_zoom)
        self.photo_canvas.bind("<B2-Motion>", self.drag_zoom)
        self.photo_canvas.bind("<ButtonRelease-2>", self.drag_release)
        self.photo_canvas.bind("<ButtonPress-3>", self.click_zoom)
        self.photo_canvas.bind("<B3-Motion>", self.drag_zoom)
        self.photo_canvas.bind("<ButtonRelease-3>", self.drag_release)
        self.update_margins()

    def clear_all_rects(self):
        if len(self.track_areas)>0:
            for ta in self.track_areas:
                self.photo_canvas.delete(ta)
            for tal in self.track_areas_labels:
                self.photo_canvas.delete(tal)
            self.track_areas = []
            self.track_areas_labels = []
            self.track_areas_co_ords0 = []

    def clear_last_rect(self):
        if len(self.track_areas)>0:
            self.photo_canvas.delete(self.track_areas[-1])
            self.photo_canvas.delete(self.track_areas_labels[-1])
            self.track_areas = self.track_areas[:-1]
            self.track_areas_labels = self.track_areas_labels[:-1]
            self.track_areas_co_ords0 = self.track_areas_co_ords0[:-1]


    def add_track_area(self,e):
        self.ta0_x= (e.x/self.image_scale)+self.img_tx
        self.ta0_y= (e.y/self.image_scale)+self.img_ty
        self.ta1_x= (e.x/self.image_scale)+self.img_tx
        self.ta1_y= (e.y/self.image_scale)+self.img_ty
        self.track_areas.append(self.photo_canvas.create_rectangle((self.ta0_x-self.img_tx)*self.image_scale,
                                                                    (self.ta0_y-self.img_ty)*self.image_scale,
                                                                    (self.ta1_x-self.img_tx)*self.image_scale,
                                                                    (self.ta1_y-self.img_ty)*self.image_scale,
                                                                    width=2,outline='red'))
        self.track_areas_labels.append(self.photo_canvas.create_text((self.ta0_x-self.img_tx)*self.image_scale,
                                                                    (self.ta0_y-self.img_ty)*self.image_scale,
                                                                    text=str(len(self.track_areas)),anchor='se',
                                                                    fill='red',font='bold'))



    def drag_track_area(self,e):
        self.ta1_x= (e.x/self.image_scale)+self.img_tx
        self.ta1_y= (e.y/self.image_scale)+self.img_ty
        self.photo_canvas.coords(self.track_areas[-1],
                                 (self.ta0_x-self.img_tx)*self.image_scale,
                                 (self.ta0_y-self.img_ty)*self.image_scale,
                                 e.x,e.y)
    def drag_track_area_release(self,e):
        if (abs(self.ta1_x-self.ta0_x)>5) and (abs(self.ta1_y-self.ta0_y)>5):
            self.track_areas_co_ords0.append([self.ta0_x,self.ta0_y,self.ta1_x,self.ta1_y])
        else:
            self.photo_canvas.delete(self.track_areas[-1])
            self.photo_canvas.delete(self.track_areas_labels[-1])
            self.track_areas = self.track_areas[:-1]
            self.track_areas_labels = self.track_areas_labels[:-1]

    def update_track_area(self):

        if len(self.track_areas)>=1:

            for ta in self.track_areas:
                self.photo_canvas.delete(ta)
            for tal in self.track_areas_labels:
                self.photo_canvas.delete(tal)
            self.track_areas = []
            self.track_areas_labels = []

            for co in self.track_areas_co_ords0:
                if (abs(co[0]-co[2])>5) and (abs(co[1]-co[3]))>5:

                    self.track_areas.append(self.photo_canvas.create_rectangle((co[0]-self.img_tx)*self.image_scale,
                                                                                (co[1]-self.img_ty)*self.image_scale,
                                                                                (co[2]-self.img_tx)*self.image_scale,
                                                                                (co[3]-self.img_ty)*self.image_scale,
                                                                                width=2,outline='red'))
                    self.track_areas_labels.append(self.photo_canvas.create_text((co[0]-self.img_tx)*self.image_scale,
                                                                                (co[1]-self.img_ty)*self.image_scale,
                                                                                text=str(len(self.track_areas)),anchor='se',
                                                                                fill='red',font='bold'))
                else:
                    self.track_areas_co_ords0.remove(co)

    def update_margins(self):
        self.margin_y = float(self.vmarg.get())
        self.margin_x = float(self.hmarg.get())
        if len(self.track_areas)>=1:

            for ta in self.track_areas:
                self.photo_canvas.delete(ta)
            for tal in self.track_areas_labels:
                self.photo_canvas.delete(tal)
            for ma in self.margin_areas:
                self.photo_canvas.delete(ma)
            self.track_areas = []
            self.track_areas_labels = []
            self.margin_areas = []
            for co in self.track_areas_co_ords0:
                if (abs(co[0]-co[2])>5) and (abs(co[1]-co[3]))>5:

                    self.track_areas.append(self.photo_canvas.create_rectangle((co[0]-self.img_tx)*self.image_scale,
                                                                                (co[1]-self.img_ty)*self.image_scale,
                                                                                (co[2]-self.img_tx)*self.image_scale,
                                                                                (co[3]-self.img_ty)*self.image_scale,
                                                                                width=2,outline='red'))
                    self.track_areas_labels.append(self.photo_canvas.create_text((co[0]-self.img_tx)*self.image_scale,
                                                                                (co[1]-self.img_ty)*self.image_scale,
                                                                                text=str(len(self.track_areas)),anchor='se',
                                                                                fill='red',font='bold'))
                    self.margin_areas.append(self.photo_canvas.create_rectangle((co[0]-self.img_tx-self.margin_x)*self.image_scale,
                                                                                (co[1]-self.img_ty-self.margin_y)*self.image_scale,
                                                                                (co[2]-self.img_tx+self.margin_x)*self.image_scale,
                                                                                (co[3]-self.img_ty+self.margin_y)*self.image_scale,
                                                                                width=2,outline='blue'))
                else:
                    self.track_areas_co_ords0.remove(co)



    def add_ref_line(self,e):
        if self.line_marker_count>1:
            self.photo_canvas.delete(self.marker_start)
            self.photo_canvas.delete(self.marker_end)
            self.photo_canvas.delete(self.ref_line_graphic)
            self.line_marker_count = 0

        if self.line_marker_count==0:
            self.rl0_x= (e.x/self.image_scale)+self.img_tx
            self.rl1_x = self.rl0_x
            self.rl0_y= (e.y/self.image_scale)+self.img_ty
            self.rl1_y = self.rl0_y
            self.marker_start = self.photo_canvas.create_oval(e.x-5,e.y-5,e.x+5,e.y+5,fill='red')
        else:
            self.rl1_x= (e.x/self.image_scale)+self.img_tx
            self.rl1_y= (e.y/self.image_scale)+self.img_ty
            self.marker_end = self.photo_canvas.create_oval(e.x-5,e.y-5,e.x+5,e.y+5,fill='red')
            self.ref_line_graphic = self.photo_canvas.create_line((self.rl0_x-self.img_tx)*self.image_scale,
                                                                  (self.rl0_y-self.img_ty)*self.image_scale,
                                                                  e.x,e.y,width=2,fill='red')
        self.line_marker_count+=1
    def drag_ref_line(self,e):

        if self.line_marker_count==1:
            self.rl0_x= (e.x/self.image_scale)+self.img_tx
            self.rl1_x = self.rl0_x
            self.rl0_y= (e.y/self.image_scale)+self.img_ty
            self.rl1_y = self.rl0_y
            self.photo_canvas.coords(self.marker_start, e.x-5,e.y-5,e.x+5,e.y+5)
        else:
            self.rl1_x= (e.x/self.image_scale)+self.img_tx
            self.rl1_y= (e.y/self.image_scale)+self.img_ty
            self.photo_canvas.coords(self.marker_end, e.x-5,e.y-5,e.x+5,e.y+5)
            self.photo_canvas.coords(self.ref_line_graphic, (self.rl0_x-self.img_tx)*self.image_scale,
                                     (self.rl0_y-self.img_ty)*self.image_scale,e.x,e.y)

    def update_ref_line(self):
        if self.line_marker_count==1:
            self.photo_canvas.delete(self.marker_start)

            self.marker_start = self.photo_canvas.create_oval(((self.rl0_x-self.img_tx)*self.image_scale)-5,
                                                              ((self.rl0_y-self.img_ty)*self.image_scale)-5,
                                                              ((self.rl0_x-self.img_tx)*self.image_scale)+5,
                                                              ((self.rl0_y-self.img_ty)*self.image_scale)+5,fill='red')
        elif self.line_marker_count==2:
            self.photo_canvas.delete(self.marker_start)
            self.photo_canvas.delete(self.marker_end)
            self.photo_canvas.delete(self.ref_line_graphic)

            self.marker_start = self.photo_canvas.create_oval(((self.rl0_x-self.img_tx)*self.image_scale)-5,
                                                              ((self.rl0_y-self.img_ty)*self.image_scale)-5,
                                                              ((self.rl0_x-self.img_tx)*self.image_scale)+5,
                                                              ((self.rl0_y-self.img_ty)*self.image_scale)+5,fill='red')

            self.marker_end = self.photo_canvas.create_oval(((self.rl1_x-self.img_tx)*self.image_scale)-5,
                                                            ((self.rl1_y-self.img_ty)*self.image_scale)-5,
                                                            ((self.rl1_x-self.img_tx)*self.image_scale)+5,
                                                            ((self.rl1_y-self.img_ty)*self.image_scale)+5,fill='red')

            self.ref_line_graphic = self.photo_canvas.create_line(((self.rl0_x-self.img_tx)*self.image_scale),
                                                                  ((self.rl0_y-self.img_ty)*self.image_scale),
                                                                  ((self.rl1_x-self.img_tx)*self.image_scale),
                                                                  ((self.rl1_y-self.img_ty)*self.image_scale),
                                                                  width=2,fill='red')

    def scale_image(self):
        try:
            self.photo_canvas.delete(self.zoom_box)
        except:
            0
        self.image_base = Image.open(self.photo_path+'\\'+self.start_photo)  # open image
        self.img_r0 = self.image_base.crop((self.img_tx,self.img_ty,self.img_bx,self.img_by))


        self.r0_width,self.r0_height = self.img_r0.size
        self.image_scale = np.min((self.photo_canvas_width/self.r0_width,
                                    self.photo_canvas_height/self.r0_height))

        self.img_r = self.img_r0.resize((int(self.r0_width*self.image_scale),int(self.r0_height*self.image_scale)), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img_r)
        self.displayed_image = self.photo_canvas.create_image(0,0, image=self.img, anchor='nw')
        if self.screen_number==1:
            self.update_ref_line()
        elif self.screen_number==2:
            self.update_track_area()
        elif self.screen_number==3:
            self.update_margins()


        self.photo_canvas.update()

    def click_zoom(self,e):
        # define start point for zoom window
        self.zoom_x0 = e.x
        self.zoom_y0 = e.y
        # create a line on this point and store it in the list
        self.zoom_box = self.photo_canvas.create_rectangle(e.x,e.y,e.x,e.y, width=2, outline='black')

    def drag_zoom(self,e):
        self.zoom_x1 = e.x
        self.zoom_y1 = e.y
        # Change the coordinates of the last created line to the new coordinates
        self.photo_canvas.coords(self.zoom_box,self.zoom_x0,self.zoom_y0,e.x,e.y)

    def drag_release(self,e):
        if self.zoom_x1<self.zoom_x0:
            self.img_tx = 0
            self.img_ty = 0
            self.img_bx, self.img_by = self.image_base.size
        else:
            h = self.zoom_y1-self.zoom_y0
            w = self.zoom_x1-self.zoom_x0
            if h>0.75*w:
                image_box_size = h
                self.img_bx = self.img_tx+((self.zoom_x0+image_box_size/0.75)/self.image_scale)
                self.img_tx = self.img_tx+(self.zoom_x0/self.image_scale)

                self.img_by = self.img_ty+((self.zoom_y0+image_box_size)/self.image_scale)
                self.img_ty = self.img_ty+(self.zoom_y0/self.image_scale)
            else:
                image_box_size = w

                self.img_bx = self.img_tx+((self.zoom_x0+image_box_size)/self.image_scale)
                self.img_tx = self.img_tx+(self.zoom_x0/self.image_scale)

                self.img_by = self.img_ty+((self.zoom_y0+0.75*image_box_size)/self.image_scale)
                self.img_ty = self.img_ty+(self.zoom_y0/self.image_scale)


        self.scale_image()




root = tk.Tk()
app = DIC_app(root)
root.mainloop()
