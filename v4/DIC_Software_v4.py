# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:38:14 2022

@author: s1879083
"""
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import ImageTk, Image,ImageFont, ImageDraw
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# Used in real-time data collection from GoPro and load cell
from goprocam import GoProCamera, constants
import ifaddr
import serial
import urllib.request
import os
import cv2

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class DIC_app(ttk.Frame):
    ''' Advanced zoom of the image '''
    def __init__(self, master):
        ''' Initialize the main Frame '''
        ttk.Frame.__init__(self, master=master)
        master.option_add( "*font", "Lucinda 10" )
        master.iconbitmap(default=resource_path("uoe.ico"))
        # master.option_add( "background", "red" )

        self.master_app = master
        self.master_app.title('Digital Image Correlation')

        # Setting the Tkinter window and the canvas in place
        ws = master.winfo_screenwidth()
        hs = master.winfo_screenheight()
        self.ww = int(ws*0.75)
        self.hw = int(hs*0.75)

        # get screen dimension
        screen_width = self.master_app.winfo_screenwidth()
        screen_height = self.master_app.winfo_screenheight()

        # find the center point
        center_x = int(screen_width / 2 - self.ww / 2)
        center_y = int(screen_height / 2 - self.hw / 2)

        # create the screen on window console
        self.master_app.geometry(f'{self.ww}x{self.hw}+{center_x}+{center_y}')
        self.master_canvas = tk.Canvas(self.master_app, width = self.ww, height = self.hw, highlightbackground = '#708090', bg = '#708090')
        self.master_canvas.place(relx=0,rely=0,anchor='nw')

        self.start_up_screen()

    def start_up_screen(self):

        self.analysis_parameters()
        self.canvas = tk.Canvas(self.master_canvas, width = self.ww, height = self.hw, highlightbackground = '#708090', bg = '#708090')
        self.canvas.place(relx=0,rely=0,anchor='nw')

        # Add buttons for analysing existing images or analysing new images
        s = ttk.Style()
        s.configure('my.TButton', font=('Helvetica', 15),justify="center")

        self.GoPro_no_LC_button = ttk.Button(self.canvas, text= "Collect new images\n (GoPro - No load cell)", style='my.TButton',command= lambda : self.GoPro_App_no_LC_start())
        self.GoPro_no_LC_button.place(relx= .33, rely= .25, anchor= tk.CENTER)
        # self.GoPro_button['font']=ttk.font.Font(size=15)

        self.Mobile_no_LC_button = ttk.Button(self.canvas, text= "Collect new images\n (Mobile - No load cell)", style='my.TButton',command= lambda : self.Mobile_App_no_LC_start())
        self.Mobile_no_LC_button.place(relx= .66, rely= .25, anchor= tk.CENTER)

        self.GoPro_button = ttk.Button(self.canvas, text= "Collect new images\nand data (GoPro)", style='my.TButton',command= lambda : self.GoPro_App_start())
        self.GoPro_button.place(relx= .25, rely= .5, anchor= tk.CENTER)
        # self.GoPro_button['font']=ttk.font.Font(size=15)

        self.Mobile_button = ttk.Button(self.canvas, text= "Collect new images\nand data (Mobile)", style='my.TButton',command= lambda : self.Mobile_App_start())
        self.Mobile_button.place(relx= .5, rely= .5, anchor= tk.CENTER)

        self.Existing_button = ttk.Button(self.canvas, text= "Analyse existing\nimages", style='my.TButton',command= lambda : self.Existing_data_App_start())
        self.Existing_button.place(relx= .75, rely= .5, anchor= tk.CENTER)

        self.Calibration_GP_button = ttk.Button(self.canvas, text= "Create new\nGoPro calibration file", style='my.TButton',command= lambda : self.Calibration_GP_App_start())
        self.Calibration_GP_button.place(relx= .33, rely= .75, anchor= tk.CENTER)

        self.Calibration_M_button = ttk.Button(self.canvas, text= "Create new mobile\ncamera calibration file", style='my.TButton',command= lambda : self.Calibration_M_App_start())
        self.Calibration_M_button.place(relx= .66, rely= .75, anchor= tk.CENTER)
        self.master_app.update()

    def analysis_parameters(self):
        self.screen_number = 1
        # Variables used in input information screen
        self.load_path = r'C:\Users\S1879083\OneDrive - University of Edinburgh\RC3 Madagascar\Data from fieldwork Madagascar\Data for DIC-Load analysis\Data from Rafik\Frame test--DIC@5s\Test28----@2-75\@2---OBLB75.csv'
        self.output_path = r'C:\Users\S1879083\OneDrive - University of Edinburgh\Edi_Python\Video_tracking\Video_tracking_app\v3\Output'
        self.output_file = 'Output_data'
        self.calibration_photo_path = r'C:\Users\S1879083\OneDrive - University of Edinburgh\Edi_Python\Video_tracking\Video_tracking_app\v3\Calibration_photos'
        self.photo_path = r'C:\Users\S1879083\OneDrive - University of Edinburgh\RC3 Madagascar\Data from fieldwork Madagascar\Data for DIC-Load analysis\Data from Rafik\Frame test--DIC@5s\Test28----@2-75'
        self.start_photo = 'G0538359.JPG'


        self.error_state = 0
        self.download_track_areas=False
        self.tracked_area_download_option = 0
        self.distortion_removal_option = 0
        self.photo_download_option = 0
        self.download_photos = False
        self.remove_image_distortion = False
        self.cam = None

        self.image_base = None
        self.ref_img = None

        self.line_marker_count = 0
        self.rl0_x = 0
        self.rl0_y = 0
        self.rl1_x = 0
        self.rl1_y = 0
        self.reference_length = 1200.0

        self.distortion_balance=1

        # Variables used in tracking area co-ordinates
        self.track_areas = []
        self.track_areas_labels = []
        self.track_areas_co_ords_initial = []

        # Variables used in margins
        self.margin_x = 200
        self.margin_y = 75
        self.margin_areas = []
        self.subpixel_option = 0
        self.subpixel_tracking=True

        self.analysis_ran = False


    def clear_canvas(self):
        for widget in self.canvas.winfo_children():
            widget.destroy()

    def create_analysis_canvas(self):
        self.screen_number = 1
        self.wc = self.ww
        self.hc = 0.8*self.hw
        self.canvas = tk.Canvas(self.master_canvas, width = self.wc, height = self.hc, highlightbackground = '#708090', bg = '#708090')
        self.canvas.place(relx=0,rely=0.1,anchor='nw')

        # Add master control buttons
        self.close_button = ttk.Button(self.master_canvas, text= "Close", command= self.close_app)
        self.close_button.place(relx= .95, rely= .05, anchor= tk.CENTER)
        self.previous_button = ttk.Button(self.master_canvas, text= "Previous", command= self.previous_screen)
        self.previous_button.place(relx= .85, rely= .95, anchor= tk.CENTER)
        self.next_button = ttk.Button(self.master_canvas, text= "Next", command= self.next_screen)
        self.next_button.place(relx= .92, rely= .95, anchor= tk.CENTER)
        self.input_info_screen()

    def close_app(self):
        self.master.destroy()
        sys.exit("Programme closed")

    def create_error_popup(self):
        self.error_window = tk.Toplevel(root)
        # get screen dimension
        screen_width = self.error_window.winfo_screenwidth()
        screen_height = self.error_window.winfo_screenheight()

        # find the center point
        center_x = int(screen_width / 2 - self.ww / 2)
        center_y = int(screen_height / 2 - self.hw / 2)

        # create the screen on window console
        self.error_window.geometry(f'500x200+{center_x}+{center_y}')

        ttk.Label(self.error_window,
          text =self.error_message).place(relx= .5, rely= .5, anchor= tk.CENTER)

    def update_screen(self):
        self.clear_canvas()
        if self.screen_number==0:
            self.close_button.destroy()
            self.previous_button.destroy()
            self.next_button.destroy()
            self.start_up_screen()

        elif self.screen_number==1:
            self.input_info_screen()

        elif self.screen_number==2:
            self.output_file = self.output_entry_name.get()

            if self.mode=='Existing':
                if (self.start_photo[-4:]!='.jpg') and (self.start_photo[-4:]!='.JPG'):
                    self.start_photo=self.start_photo+'.JPG'

                try:
                    # Check if input folder and photos exist
                    self.flnms = [file for file in glob.glob(self.photo_path+"/*.jpg", recursive = True)]
                    self.error_message = 'Folder with input photos\ndoes not exist!'
                    glob.glob(self.photo_path)[0]

                    self.error_message = 'Start photo does\n not exist!'
                    indx = self.flnms.index(self.photo_path+"\\"+self.start_photo)
                    self.flnms = self.flnms[indx:]

                except:
                    self.create_error_popup()
                    self.screen_number+=-1
                    self.input_info_screen()


            try:
                # Check if output folder exists
                self.error_message = 'Output folder does\nnot exist!'
                glob.glob(self.output_path)[0]
                self.error_state=0
                self.error_message = 'Output file already exists.'
                if os.path.isfile(self.output_path+'//'+self.output_file+'.csv')==True:
                    raise ValueError('Output file already exists!')
                self.reference_length_screen()
            except:
                self.create_error_popup()
                self.screen_number+=-1
                self.input_info_screen()

        elif self.screen_number==3:
            self.reference_length = self.ref_length.get()
            self.balance_updated()
            self.ref_length_updated()
            try:
                self.reference_line_length = (abs(self.rl0_x - self.rl1_x) ** 2 +
                                              abs(self.rl0_y - self.rl1_y) ** 2) ** (1 / 2)
                if self.reference_line_length==0:
                    raise Exception('Reference length equals zero!')
                self.scale_factor = float(self.reference_length)/self.reference_line_length
                self.track_area_screen()
            except:
                self.error_message = 'Please select reference length!'
                self.create_error_popup()
                self.screen_number+=-1
                self.reference_length_screen()

        elif self.screen_number==4:
            if len(self.track_areas_co_ords_initial)>0:
                self.next_button.configure(text='Next',command=self.next_screen)
                self.track_margins_screen()
            else:
                self.error_message = 'Number of tracked areas\nmust be greater than zero!'
                self.create_error_popup()
                self.screen_number+=-1
                self.track_area_screen()

        elif self.screen_number == 5:
            self.outputs_screen()


    def previous_screen(self):
        self.screen_number+=-1
        self.update_screen()

    def next_screen(self):
        self.screen_number+=1
        self.update_screen()

    def take_photo_GP(self):
        self.photo_url = self.gopro.take_photo(0)
        self.ref_img= Image.open(urllib.request.urlopen(self.photo_url))
        print('Photo taken')

    def connect_to_gopro(self):
        for adapter in ifaddr.get_adapters():
            if "GoPro" in adapter.nice_name:
                for ip in adapter.ips:
                    if ip.is_IPv4:
                        addr = ip.ip.split(".")
                        addr[len(addr) - 1] = "51"
                        addr = ".".join(addr)


        try:
            self.gopro = GoProCamera.GoPro(ip_address=addr, camera=constants.gpcontrol)
            self.gopro.mode(constants.Mode.PhotoMode, constants.Mode.SubMode.Photo.Single)
            if self.ref_img is None:
                if self.mode == 'GoPro' or self.mode=='GoPro - No LC':
                    self.take_photo_GP()
        except:
            self.error_message = 'GoPro camera not connected\nor is switched off!'
            self.create_error_popup()
            self.error_state=1

    def connect_to_load_cell(self):
        connected=0
        self.com_no = 0
        # Search for load cell by checking all usb channels
        while (connected==0) and (self.com_no<20):
            try:
                ser = serial.Serial('COM'+str(self.com_no), baudrate=115200,write_timeout=0.1 ,timeout=0.1)
                ser.write("!001:SYS?<CR>\r".encode('ascii'))
                x0 = ser.readline().decode('ascii').strip()
                if len(x0)>0:
                    connected=1
                else:
                    try:
                        ser.close()
                    except:
                        0
                    self.com_no+=1
            except:
                try:
                    ser.close()
                except:
                    0
                self.com_no+=1
        if connected==1:
            print('Load cell detected on COM'+str(self.com_no))
            ser.close()
        else:
            print('Load cell not connected')
            self.error_message = 'Load cell not detected!'
            self.create_error_popup()
            self.error_state=1

    def GoPro_App_start(self):

        self.mode = 'GoPro'
        self.error_state=0

        connection_label = ttk.Label(self.master_canvas, text= "Connecting to GoPro")
        connection_label.place(relx=0.5,rely=0.1,anchor=tk.N)
        self.master_canvas.update()
        self.connect_to_gopro()

        if self.error_state==0:
            connection_label['text']='Connected to GoPro\nConnecting to load cell'
            self.master_canvas.update()
            self.connect_to_load_cell()


            if self.error_state==0:
                connection_label['text']='Connected to GoPro\nConnected to load cell'
                self.image_base = self.ref_img
                self.master_canvas.update()

                connection_label.destroy()
                self.clear_canvas()
                self.canvas.destroy()

                self.create_analysis_canvas()
            else:
                connection_label.destroy()
        else:
            connection_label.destroy()

    def GoPro_App_no_LC_start(self):

        self.mode = 'GoPro - No LC'
        self.error_state=0

        connection_label = ttk.Label(self.master_canvas, text= "Connecting to GoPro")
        connection_label.place(relx=0.5,rely=0.1,anchor=tk.N)
        self.master_canvas.update()
        self.connect_to_gopro()

        if self.error_state==0:
            connection_label['text']='Connected to GoPro'
            self.master_canvas.update()

            self.image_base = self.ref_img
            self.master_canvas.update()

            connection_label.destroy()
            self.clear_canvas()
            self.canvas.destroy()

            self.create_analysis_canvas()
        else:
            connection_label.destroy()

    def take_photo_Mobile(self):
        ret, frame = self.cam.read()
        ret, frame = self.cam.read()
        self.photo_url = None
        self.ref_img= Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print('Photo taken')

    def connect_to_mobile_camera(self):

        try:
            if self.cam == None:
                try:
                    self.cam = cv2.VideoCapture(2)
                except:
                    try:
                        self.cam = cv2.VideoCapture(1)
                    except:
                        self.cam = cv2.VideoCapture(0)

            if self.ref_img is None and self.mode=='Mobile' or self.mode=='Mobile - No LC':
                self.take_photo_Mobile()
        except:
            self.error_message = 'Cellphone camera not connected\nor is switched off!'
            self.create_error_popup()
            self.error_state=1

    def Mobile_App_start(self):

        self.mode = 'Mobile'
        self.error_state=0
        self.ref_img = None


        connection_label = ttk.Label(self.master_canvas, text= "Connecting to cellphone camera")
        connection_label.place(relx=0.5,rely=0.1,anchor=tk.N)
        self.master_canvas.update()
        self.connect_to_mobile_camera()

        if self.error_state==0:
            connection_label['text']='Connected to cellphone camera\nConnecting to load cell'
            self.master_canvas.update()
            self.connect_to_load_cell()


            if self.error_state==0:
                connection_label['text']='Connected to cellphone camera\nConnected to load cell'
                self.image_base = self.ref_img
                self.master_canvas.update()

                connection_label.destroy()
                self.clear_canvas()
                self.canvas.destroy()

                self.create_analysis_canvas()
            else:
                connection_label.destroy()
        else:
            connection_label.destroy()

    def Mobile_App_no_LC_start(self):

        self.mode = 'Mobile - No LC'
        self.error_state=0
        self.ref_img = None


        connection_label = ttk.Label(self.master_canvas, text= "Connecting to cellphone camera")
        connection_label.place(relx=0.5,rely=0.1,anchor=tk.N)
        self.master_canvas.update()
        self.connect_to_mobile_camera()

        if self.error_state==0:
            connection_label['text']='Connected to cellphone camera'
            self.master_canvas.update()

            self.image_base = self.ref_img
            self.master_canvas.update()

            connection_label.destroy()
            self.clear_canvas()
            self.canvas.destroy()

            self.create_analysis_canvas()
        else:
            connection_label.destroy()

    def Existing_data_App_start(self):
        self.clear_canvas()
        self.canvas.destroy()
        self.mode = 'Existing'
        self.image_base = None
        self.create_analysis_canvas()



    def Calibration_GP_App_start(self):
        self.mode = 'GP Calibration'
        self.error_state=0

        connection_label = ttk.Label(self.master_canvas, text= "Connecting to GoPro camera")
        connection_label.place(relx=0.5,rely=0.1,anchor=tk.N)
        self.master_canvas.update()
        self.connect_to_gopro()

        if self.error_state==0:

            connection_label['text']='Connected to GoPro camera'
            self.image_base = self.ref_img
            self.master_canvas.update()

            connection_label.destroy()

            self.create_calibration_window()

        else:
            connection_label.destroy()

    def close_calibration_photo_window(self):
        self.calibration_app.attributes("-topmost", True)
        self.cali_photo_window.destroy()

    def close_calibration_app(self):

        self.calibration_app.destroy()

    def browsefunc_calibration_photo_path(self):
        filename = tk.filedialog.askdirectory()
        self.calibration_photo_path = filename
        self.calibration_photo_path_label.config(text=filename)
    def displayed_calibration_image_changed(self,new_option):
        self.cali_label.set(new_option)
        self.calibration_photo_index = self.displayed_calibration_image_name.index(new_option)
        self.update_calibration_photo()

    def update_calibration_photo(self):
        try:
            self.close_calibration_photo_window()
        except:
            pass
        self.image_base = Image.open(self.calibration_flnms[self.calibration_photo_index])  # open image

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


        self.distort_canvas.create_image(0,0, image=self.img, anchor='nw')

        self.remove_image_distortion_func(self.calibration_flnms[self.calibration_photo_index])
        # img = cv2.cvtColor(self.undistorted_img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)

        self.img_rc = self.image_base.resize((int(self.r0_width*self.image_scale),int(self.r0_height*self.image_scale)), Image.ANTIALIAS)
        self.img_c = ImageTk.PhotoImage(self.img_rc)

        self.corrected_canvas.create_image(0,0, image=self.img_c, anchor='nw')

    def create_calibration_window(self):
        self.calibration_app = tk.Toplevel(root)
        self.calibration_app.attributes("-topmost", True)
        self.calibration_app.title('Fisheye calibration')

        window_width = int(0.95*self.ww)
        window_height = int(0.95*self.hw)

        # get screen dimension
        screen_width = self.calibration_app.winfo_screenwidth()
        screen_height = self.calibration_app.winfo_screenheight()

        # find the center point
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)

        # create the screen on window console
        self.calibration_app.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        self.calibration_canvas = tk.Canvas(self.calibration_app, width = window_width, height = window_height, highlightbackground = '#708090', bg = '#708090')
        self.calibration_canvas.grid()
        self.calibration_app.update()
        ttk.Button(self.calibration_canvas, text= "Close calibration", command= self.close_calibration_app).place(relx= .975, rely= .05, anchor= tk.E)
        self.calibration_app.update()
        # # Create canvas and put images on it (if they exist)
        self.displayed_calibration_canvas = tk.Canvas(self.calibration_app, width = window_width, height = 0.25*window_height, highlightbackground = '#708090', bg = '#708090')
        self.calibration_canvas.create_window(0,0.75*window_height,anchor='nw', window=self.displayed_calibration_canvas)

        self.photo_canvas_width = 0.5*window_width
        self.photo_canvas_height = 0.65*window_height

        self.distort_canvas = tk.Canvas(self.calibration_app, width = self.photo_canvas_width,
                                      height = self.photo_canvas_height, highlightbackground = '#708090', bg = '#708090')

        self.calibration_canvas.create_window(0, 0.1*window_height,anchor='nw', window=self.distort_canvas)
        self.distort_canvas.update()  # wait till canvas is created

        self.corrected_canvas = tk.Canvas(self.calibration_app, width = self.photo_canvas_width,
                                      height = self.photo_canvas_height, highlightbackground = 'grey', bg = '#708090')

        self.calibration_canvas.create_window(0.5*window_width, 0.1*window_height,anchor='nw', window=self.corrected_canvas)
        self.corrected_canvas.update()  # wait till canvas is created

        ttk.Label(self.displayed_calibration_canvas,text='Calibration photos folder path').place(relx= .01, rely= .1, anchor= tk.W)
        ttk.Button(self.displayed_calibration_canvas, text= "Browse",
                  command= self.browsefunc_calibration_photo_path).place(relx= 0.01, rely= .25, anchor= tk.NW)
        self.calibration_photo_path_label = ttk.Label(self.displayed_calibration_canvas,text=self.calibration_photo_path,justify='left',wraplength=int(0.35*window_width))
        self.calibration_photo_path_label.place(relx= .1, rely= .25, anchor= tk.NW)


        self.calibration_flnms =  glob.glob(self.calibration_photo_path+'\*.jpg')
        self.calibration_photo_index = 0

        if len(self.calibration_flnms)>0:
            self.update_calibration_photo()

        ttk.Label(self.displayed_calibration_canvas,text='Current displayed calibration image').place(relx= .5, rely= .1, anchor= tk.W)

        if len(self.calibration_flnms)>0:
            self.displayed_calibration_image_name = [x.split('\\')[-1] for x in self.calibration_flnms]

            self.cali_label = tk.StringVar(self,self.displayed_calibration_image_name)

            self.displayed_calibration_image_menu = ttk.OptionMenu(
                self.displayed_calibration_canvas,

                self.cali_label,
                self.displayed_calibration_image_name[self.calibration_photo_index],
                *self.displayed_calibration_image_name,
                command=self.displayed_calibration_image_changed)
            self.displayed_calibration_image_menu.place(width=0.45*window_width,relx= .75, rely= .25,anchor= tk.N)

        self.calibration_button = ttk.Button(self.displayed_calibration_canvas, text= "Create new calibration file",
                  command= self.create_calibration_file)
        self.calibration_button.place(relx= .55, rely= .5, anchor= tk.NW)

        ttk.Button(self.displayed_calibration_canvas, text= "Take new calibration photos",
                  command= self.take_calibration_photos_window).place(relx= .95, rely= .5, anchor= tk.NE)

        self.calibration_status_label=ttk.Label(self.displayed_calibration_canvas,
                                              text= "")
        self.calibration_status_label.place(relx= .55,rely= .7, anchor= tk.NW)

    def take_calibration_photos_window(self):
        try:
            self.close_calibration_photo_window()
        except:
            pass
        self.cali_photo_window = tk.Toplevel(root)
        self.cali_photo_window.attributes("-topmost", True)
        self.calibration_app.attributes("-topmost", False)
        self.cali_photo_window.title('New Calibration Photos')
        window_width = int(0.75*self.ww)
        window_height = int(0.7*self.hw)

        # get screen dimension
        screen_width = self.cali_photo_window.winfo_screenwidth()
        screen_height = self.cali_photo_window.winfo_screenheight()

        # find the center point
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)

        # create the screen on window console
        self.cali_photo_window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.cali_photo_canvas = tk.Canvas(self.cali_photo_window, width = window_width, height = window_height, highlightbackground = '#708090', bg = '#708090')
        self.cali_photo_canvas.grid()
        self.cali_photo_window.update()
        ttk.Button(self.cali_photo_canvas, text= "Close window", command= self.close_calibration_photo_window).place(relx= .9, rely= .05, anchor= tk.CENTER)
        ttk.Button(self.cali_photo_canvas, text= "Take calibration photo", command= self.take_calibration_photo).place(relx= .9, rely= .95, anchor= tk.CENTER)

        # Create canvas and put images on it (if they exist)
        self.cali_photo_canvas_width = 0.6*self.ww
        self.cali_photo_canvas_height = 0.7*self.hw

        self.cali_distort_canvas = tk.Canvas(self.cali_photo_window, width = self.cali_photo_canvas_width,
                                      height = self.cali_photo_canvas_height, highlightbackground = '#708090', bg = '#708090')

        self.cali_photo_canvas.create_window(0, 0,anchor='nw', window=self.cali_distort_canvas)
        self.cali_distort_canvas.update()  # wait till canvas is created

    def take_calibration_photo(self):
        if self.mode=='GP Calibration':
            self.photo_url = self.gopro.take_photo(0)
            self.gopro.downloadLastMedia(custom_filename=self.calibration_photo_path+'//'+self.photo_url.split('/')[-1])
            new_cali_img= Image.open(urllib.request.urlopen(self.photo_url))
        elif self.mode=='M Calibration':
            ret, frame = self.cam.read()
            ret, frame = self.cam.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            new_cali_img = Image.fromarray(img)
            self.photo_url = 'Calibration_photo_'+str(datetime.datetime.now()).replace(':','.')+'.jpg'
            new_cali_img.save(self.calibration_photo_path+'//'+self.photo_url)
        self.calibration_flnms.append(self.calibration_photo_path+'//'+self.photo_url.split('/')[-1])
        cnt = len(self.calibration_flnms)-1
        try:
            self.displayed_calibration_image_name.append(self.photo_url.split('/')[-1])

            self.displayed_calibration_image_menu["menu"].add_command(
                                                                  label=self.displayed_calibration_image_name[cnt],
                                                                  command= lambda: self.displayed_calibration_image_changed(self.displayed_calibration_image_name[cnt]))
        except:
            self.displayed_calibration_image_name = [x.split('\\')[-1] for x in self.calibration_flnms]

            self.cali_label = tk.StringVar(self,self.displayed_calibration_image_name)

            self.displayed_calibration_image_menu = ttk.OptionMenu(
                self.displayed_calibration_canvas,

                self.cali_label,
                self.displayed_calibration_image_name[self.calibration_photo_index],
                *self.displayed_calibration_image_name,
                command=self.displayed_calibration_image_changed)
            self.displayed_calibration_image_menu.place(width=0.45*int(0.95*self.ww),relx= .75, rely= .25,anchor= tk.N)

        r0_width,r0_height = new_cali_img.size

        image_scale = np.min((self.cali_photo_canvas_width/r0_width,
                                    self.cali_photo_canvas_height/r0_height))


        img_r = new_cali_img.resize((int(r0_width*image_scale),int(r0_height*image_scale)), Image.ANTIALIAS)
        self.image_cali = ImageTk.PhotoImage(img_r)
        self.cali_distort_canvas.create_image(0,0.75*0.25*self.ww, image=self.image_cali, anchor='w')
        self.cali_photo_canvas.update()  # wait till canvas is created




    def cancel_calibration_analysis(self):
        self.calibration_status = False
        self.calibration_status_label['text']='Calibration cancelled.'
        self.calibration_button.configure(text= "Create new calibration file",command= self.create_calibration_file)

    def create_calibration_file(self):

        self.calibration_status = True
        self.calibration_button.configure(text= "Cancel calibration",command= self.cancel_calibration_analysis)

        checkerboard = (6,9)
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, checkerboard[0]*checkerboard[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
        _img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        count=0
        for fname in self.calibration_flnms:
            if self.calibration_status == True:
                print(count/len(self.calibration_flnms))
                self.calibration_status_label['text']=str(np.round(100*count/len(self.calibration_flnms),2))+'% complete.'
                self.calibration_app.update()
                count+=1
                img = cv2.imread(fname)
                if _img_shape == None:
                    _img_shape = img.shape[:2]
                else:
                    assert _img_shape == img.shape[:2], "All images must share the same size."
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, checkerboard,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    objpoints.append(objp)
                    cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                    imgpoints.append(corners)
        if self.calibration_status == True:
            N_OK = len(objpoints)
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            rms, _, _, _, _ = \
                cv2.fisheye.calibrate(
                    objpoints,
                    imgpoints,
                    gray.shape[::-1],
                    K,
                    D,
                    rvecs,
                    tvecs,
                    calibration_flags,
                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                )
            DIM = _img_shape[::-1]
            # Create and save calibration files
            np.savetxt(self.calibration_photo_path+'/Calibration_file_part_1.txt',DIM)
            np.savetxt(self.calibration_photo_path+'/Calibration_file_part_2.txt',K)
            np.savetxt(self.calibration_photo_path+'/Calibration_file_part_3.txt',D)

            print('Calibration file created!')
            self.calibration_status_label['text']='Calibration file saved successfully!'
            self.calibration_status = False

    def Calibration_M_App_start(self):
        self.mode = 'M Calibration'
        self.error_state=0

        connection_label = ttk.Label(self.master_canvas, text= "Connecting to cellphone camera")
        connection_label.place(relx=0.5,rely=0.1,anchor=tk.N)
        self.master_canvas.update()
        self.connect_to_mobile_camera()

        if self.error_state==0:

            connection_label['text']='Connected to cellphone camera'
            self.image_base = self.ref_img
            self.master_canvas.update()

            connection_label.destroy()

            self.create_calibration_window()

        else:
            connection_label.destroy()

    def browsefunc_photo(self):
        filename = tk.filedialog.askopenfilename()
        self.photo_path = '/'.join(filename.split('/')[:-1])
        self.start_photo = filename.split('/')[-1]
        self.photo_label.config(text=filename)

    def browsefunc_calibration_path(self):
        filename = tk.filedialog.askdirectory()
        self.calibration_photo_path = filename
        self.calibration_file_labelpath.config(text=filename)

    def browsefunc_load_cell(self):
        filename = tk.filedialog.askopenfilename()
        self.load_path = filename
        self.lc_label.config(text=filename)

    def browsefunc_output_path(self):
        filename = tk.filedialog.askdirectory()
        self.output_path = filename
        self.output_path_label.config(text=filename)

    def photo_download_option_changed(self,new_option):
        self.photo_download_option = self.photo_download_options.index(new_option)
        if self.photo_download_option==0:
            self.download_photos=False
        else:
            self.download_photos=True

    def tracked_area_download_option_changed(self,new_option):
        self.tracked_area_download_option = self.tracked_area_download_options.index(new_option)
        if self.tracked_area_download_option==1:
            self.download_track_areas=True
        else:
            self.download_track_areas=False
    def distortion_removal_option_changed(self,new_option):
        self.image_base = None
        self.distortion_removal_option = self.distortion_removal_options.index(new_option)
        if self.distortion_removal_option==0:
            self.remove_image_distortion = False
            self.calibration_file_label.destroy()
            self.calibration_file_button.destroy()
            self.calibration_file_labelpath.destroy()
        else:
            self.remove_image_distortion = True
            self.add_distortion_options()

    def add_distortion_options(self):
        if self.distortion_removal_option==1:
            self.calibration_file_label = ttk.Label(self.canvas,text='Select folder containing calibration files')
            self.calibration_file_label.place(relx= .05, rely= .55, anchor= tk.W)
            self.calibration_file_button = ttk.Button(self.canvas, text= "Browse",
                      command= self.browsefunc_calibration_path)
            self.calibration_file_button.place(relx= 0.05, rely= .6, anchor= 'w')
            self.calibration_file_labelpath = ttk.Label(self.canvas,text=self.calibration_photo_path)
            self.calibration_file_labelpath.place(relx= .1, rely= .6, anchor= tk.W)



    def input_info_screen(self):

        ttk.Label(self.canvas, text= "Browse to select the folder where the output data should be saved,"+
                 " add the name of the output file (excluding file extension) and select output options.",
                 justify=tk.LEFT,font='bold').place(relx= .05,rely= .1,anchor= tk.W)

        ttk.Label(self.canvas,text='Output folder path').place(relx= .05, rely= .15, anchor= tk.W)
        ttk.Button(self.canvas, text= "Browse",
                  command= self.browsefunc_output_path).place(relx= 0.05, rely= .2, anchor= 'w')
        self.output_path_label = ttk.Label(self.canvas,text=self.output_path)
        self.output_path_label.place(relx= .125, rely= .2, anchor= tk.W)

        ttk.Label(self.canvas,text='Output filename prefix').place(relx= .05, rely= .25, anchor= tk.W)
        self.output_entry_name = tk.StringVar(self.canvas,self.output_file)
        ttk.Entry(self.canvas, width= 150,textvariable=self.output_entry_name).place(relx= .05, rely= .3, anchor= tk.W)

        ttk.Label(self.canvas,text='Output photos showing tracked areas?').place(relx= .05, rely= .35, anchor= tk.W)

        self.tracked_area_download_options = ['No','Yes']
        self.tracked_area_download_menu = ttk.OptionMenu(
            self.canvas,
            tk.StringVar(self,self.tracked_area_download_options),
            self.tracked_area_download_options[self.tracked_area_download_option],
            *self.tracked_area_download_options,
            command=self.tracked_area_download_option_changed)
        self.tracked_area_download_menu.place(relx= .05, rely= .4, anchor= tk.W)

        ttk.Label(self.canvas,text='Remove image distortion using existing calibration file?').place(relx= .05, rely= .45, anchor= tk.W)

        self.distortion_removal_options = ['No','Yes']
        self.distortion_removal_menu = ttk.OptionMenu(
            self.canvas,
            tk.StringVar(self,self.distortion_removal_options),
            self.distortion_removal_options[self.distortion_removal_option],
            *self.distortion_removal_options,
            command=self.distortion_removal_option_changed)
        self.distortion_removal_menu.place(relx= .05, rely= .5, anchor= tk.W)

        self.add_distortion_options()

        if self.mode == 'Mobile' or self.mode == "GoPro" or self.mode == 'Mobile - No LC' or self.mode == 'GoPro - No LC':
            ttk.Label(self.canvas,text='Download photos from camera after analysis?').place(relx= .35, rely= .35, anchor= tk.W)

            self.photo_download_options = ['No','Yes']
            self.photo_download_menu = ttk.OptionMenu(
                self.canvas,
                tk.StringVar(self,self.photo_download_options),
                self.photo_download_options[self.photo_download_option],
                *self.photo_download_options,
                command=self.photo_download_option_changed)
            self.photo_download_menu.place(relx= .35, rely= .4, anchor= tk.W)

        if self.mode=='Existing':
            ttk.Label(self.canvas, text= "Browse to select first photo to use in analysis and load cell data.",
                     justify=tk.LEFT,font='bold').place(relx= .05,rely= .75,anchor= tk.W)
            ttk.Label(self.canvas,text='Select first photo to use').place(relx= .05, rely= .8, anchor= tk.W)
            ttk.Button(self.canvas, text= "Browse",
                      command= self.browsefunc_photo).place(relx= 0.05, rely= .85, anchor= 'w')
            self.photo_label = ttk.Label(self.canvas,text=self.start_photo)
            self.photo_label.place(relx= .125, rely= .85, anchor= tk.W)

            ttk.Label(self.canvas,text='Select load cell data').place(relx= .05, rely= .9, anchor= tk.W)
            ttk.Button(self.canvas, text= "Browse",
                      command= self.browsefunc_load_cell).place(relx= 0.05, rely= .95, anchor= 'w')
            self.lc_label = ttk.Label(self.canvas,text=self.load_path)
            self.lc_label.place(relx= .125, rely= .95, anchor= tk.W)


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
        self.img_r0 = self.image_base.crop((self.img_tx,self.img_ty,self.img_bx,self.img_by))


        self.r0_width,self.r0_height = self.img_r0.size
        self.image_scale = np.min((self.photo_canvas_width/self.r0_width,
                                    self.photo_canvas_height/self.r0_height))

        self.img_r = self.img_r0.resize((int(self.r0_width*self.image_scale),int(self.r0_height*self.image_scale)), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img_r)
        self.displayed_image = self.photo_canvas.create_image(0,0, image=self.img, anchor='nw')
        if self.screen_number==2:
            self.update_ref_line()
        elif self.screen_number==3:
            self.update_track_area()
        elif self.screen_number==4:
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
        self.photo_canvas.delete(self.zoom_box)
        if self.zoom_x1<self.zoom_x0:
            self.img_tx = 0
            self.img_ty = 0
            self.img_bx, self.img_by = self.image_base.size
        else:
            h = self.zoom_y1-self.zoom_y0
            w = self.zoom_x1-self.zoom_x0
            if h>w:
                image_box_size = h
                self.img_bx = self.img_tx+((self.zoom_x0+image_box_size*(self.photo_canvas_width/self.photo_canvas_height))/self.image_scale)
                self.img_tx = self.img_tx+(self.zoom_x0/self.image_scale)

                self.img_by = self.img_ty+((self.zoom_y0+image_box_size)/self.image_scale)
                self.img_ty = self.img_ty+(self.zoom_y0/self.image_scale)
            else:
                image_box_size = w

                self.img_bx = self.img_tx+((self.zoom_x0+image_box_size)/self.image_scale)
                self.img_tx = self.img_tx+(self.zoom_x0/self.image_scale)

                self.img_by = self.img_ty+((self.zoom_y0+image_box_size/(self.photo_canvas_width/self.photo_canvas_height))/self.image_scale)
                self.img_ty = self.img_ty+(self.zoom_y0/self.image_scale)

        self.scale_image()

    def update_ref_photo(self):
        if self.mode=='GoPro' or self.mode=='GoPro - No LC':
            self.take_photo_GP()
        elif self.mode=='Mobile' or self.mode=='Mobile - No LC':
            self.take_photo_Mobile()
        if self.remove_image_distortion == True:
            self.remove_image_distortion_func(self.photo_url)
        else:
            self.image_base = self.ref_img
        self.scale_image()

    def remove_image_distortion_func(self,img_path):

        # Load calibration file
        DIM = np.loadtxt(self.calibration_photo_path+'/Calibration_file_part_1.txt')
        K=np.loadtxt(self.calibration_photo_path+'/Calibration_file_part_2.txt')
        D=np.loadtxt(self.calibration_photo_path+'/Calibration_file_part_3.txt')
        if self.mode=='Existing' or self.mode=='GP Calibration' or self.mode=='M Calibration':
            img = cv2.imread(img_path)
        elif self.mode=='GoPro' or self.mode=='Mobile' or self.mode=='GoPro - No LC' or self.mode=='Mobile - No LC':
            img = cv2.cvtColor(np.array(self.ref_img), cv2.COLOR_RGB2BGR)
        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim1, np.eye(3), balance=self.distortion_balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2)
        self.undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.undistorted_img = cv2.cvtColor(self.undistorted_img, cv2.COLOR_BGR2RGB)
        self.image_base = Image.fromarray(self.undistorted_img)


    def reference_length_screen(self):
        self.img_tx = 0
        self.img_ty = 0

        # Add reference image to canvas
        self.photo_canvas_width = 0.75*self.wc
        self.photo_canvas_height = self.hc
        self.photo_canvas = tk.Canvas(self.canvas, width = self.photo_canvas_width,
                                      height = self.photo_canvas_height, highlightbackground = '#708090', bg = '#708090')

        self.canvas.create_window(0, 0,anchor='nw', window=self.photo_canvas)
        self.photo_canvas.update()  # wait till canvas is created

        # open image, removing distortion if necessary
        if self.mode=='Existing':
            if self.image_base == None:
                if self.remove_image_distortion == True:
                    self.remove_image_distortion_func(self.photo_path+'\\'+self.start_photo)
                else:
                    self.image_base = Image.open(self.photo_path+'\\'+self.start_photo)
        else:
            if self.remove_image_distortion == True:
                self.remove_image_distortion_func(self.photo_url)
            else:
                self.image_base = self.ref_img


        self.img_bx,self.img_by = self.image_base.size
        self.scale_image()

        # Bind buttons to canvas
        self.photo_canvas.bind("<ButtonPress-1>", self.add_ref_line)
        self.photo_canvas.bind("<B1-Motion>", self.drag_ref_line)

        self.photo_canvas.bind("<ButtonPress-2>", self.click_zoom)
        self.photo_canvas.bind("<B2-Motion>", self.drag_zoom)
        self.photo_canvas.bind("<ButtonRelease-2>", self.drag_release)
        self.photo_canvas.bind("<ButtonPress-3>", self.click_zoom)
        self.photo_canvas.bind("<B3-Motion>", self.drag_zoom)
        self.photo_canvas.bind("<ButtonRelease-3>", self.drag_release)
        self.update_ref_line()



        ttk.Label(self.canvas, text= "Click and drag on image to set known"+
                 "\nlength and update reference length box"+
                 "\nwith length in mm before clicking next.\n \n"+
                 "Zoom using scroll click/right click and drag.",justify=tk.LEFT).place(relx= .77,rely= .15,
                                                                                                    anchor= tk.W)
        if self.mode == 'Mobile' or self.mode == "GoPro" or self.mode=='GoPro - No LC' or self.mode=='Mobile - No LC':
            ttk.Button(self.canvas, text= "Take new reference photo",
                      command= self.update_ref_photo).place(relx= .77, rely= .5, anchor= 'w')

        ttk.Label(self.canvas, text= "Reference length [mm]").place(relx= .77, rely= .65, anchor= tk.W)

        self.ref_length=tk.StringVar(self.master, value=str(self.reference_length))
        self.ref_length_entry = ttk.Entry(self.canvas, width= 10,textvariable=self.ref_length,
                                         validate="focusout", validatecommand=self.ref_length_updated)
        self.ref_length_entry.place(relx= .77, rely= .7, anchor= tk.W)
        ttk.Button(self.canvas, text= "Update reference length", command= self.ref_length_updated).place(relx= .77, rely= .75, anchor= 'w')



        if self.remove_image_distortion == True:
            ttk.Label(self.canvas, text= "Distortion balance").place(relx= .77, rely= .85, anchor= tk.W)
            self.dist_balance=tk.StringVar(self.master, value=str(self.distortion_balance))
            self.balance_entry = ttk.Entry(self.canvas, width= 10,textvariable=self.dist_balance,
                     validate="focusout", validatecommand=self.balance_updated)
            self.balance_entry.place(relx= .77, rely= .9, anchor= tk.W)
            ttk.Button(self.canvas, text= "Update distortion balance", command= self.update_image_distortion_value).place(relx= .77, rely= .95, anchor= 'w')

    def update_image_distortion_value(self):
        self.balance_updated()
        if self.remove_image_distortion == True:
            self.remove_image_distortion_func(self.photo_path+'\\'+self.start_photo)
        else:
            self.image_base = Image.open(self.photo_path+'\\'+self.start_photo)
        self.img_bx,self.img_by = self.image_base.size
        self.scale_image()
        self.update_ref_line()

    def balance_updated(self):
        try:
            if float(self.dist_balance.get())>0:
                self.distortion_balance = float(self.dist_balance.get())
                return True
            else:
                return False
        except:
            return False

    def ref_length_updated(self):
        try:
            if float(self.ref_length.get())>0:
                self.reference_length = float(self.ref_length.get())
                return True
            else:
                return False
        except:
            return False

    def clear_all_rects(self):
        if len(self.track_areas)>0:
            for ta in self.track_areas:
                self.photo_canvas.delete(ta)
            for tal in self.track_areas_labels:
                self.photo_canvas.delete(tal)
            self.track_areas = []
            self.track_areas_labels = []
            self.track_areas_co_ords_initial = []

    def clear_last_rect(self):
        if len(self.track_areas)>0:
            self.photo_canvas.delete(self.track_areas[-1])
            self.photo_canvas.delete(self.track_areas_labels[-1])
            self.track_areas = self.track_areas[:-1]
            self.track_areas_labels = self.track_areas_labels[:-1]
            self.track_areas_co_ords_initial = self.track_areas_co_ords_initial[:-1]

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
            self.track_areas_co_ords_initial.append([(self.ta0_x),(self.ta0_y),(self.ta1_x),(self.ta1_y)])
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

            for co in self.track_areas_co_ords_initial:
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
                    self.track_areas_co_ords_initial.remove(co)

    def update_plotted_track_area(self):

        if len(self.track_areas)>=1:

            for ta in self.track_areas:
                self.photo_canvas.delete(ta)
            for tal in self.track_areas_labels:
                self.photo_canvas.delete(tal)
            self.track_areas = []
            self.track_areas_labels = []

            for co in self.track_areas_co_ords:

                self.track_areas.append(self.photo_canvas.create_rectangle((co[0]-self.img_tx)*self.image_scale,
                                                                            (co[1]-self.img_ty)*self.image_scale,
                                                                            (co[2]-self.img_tx)*self.image_scale,
                                                                            (co[3]-self.img_ty)*self.image_scale,
                                                                            width=2,outline='red'))
                self.track_areas_labels.append(self.photo_canvas.create_text((co[0]-self.img_tx)*self.image_scale,
                                                                            (co[1]-self.img_ty)*self.image_scale,
                                                                            text=str(len(self.track_areas)),anchor='se',
                                                                            fill='red',font='bold'))


    def track_area_screen(self):
        self.img_tx = 0
        self.img_ty = 0

        # Add reference image to canvas
        self.photo_canvas_width = 0.75*self.wc
        self.photo_canvas_height = self.hc
        self.photo_canvas = tk.Canvas(self.canvas, width = self.photo_canvas_width,
                                      height = self.photo_canvas_height, highlightbackground = '#708090', bg = '#708090')

        self.canvas.create_window(0, 0,anchor='nw', window=self.photo_canvas)
        self.photo_canvas.update()  # wait till canvas is created


        self.img_bx,self.img_by = self.image_base.size
        self.scale_image()

        ttk.Label(self.canvas, text= "Click and drag on image to select"+
                 "\narea(s) of image to track before\nclicking next.\n \n"+
                 "Zoom using scroll click/right click and drag.\n\n"+
                 "Tips for selecting tracked areas:\n"+
                          "- Select a high contrast section of the image\n"+
                          "- Leave a gap between the edge of the\n   tracked area and the background\n"
                          "- Make sure the object you want to track\n   is not close to the edge of the image"
                 ,justify=tk.LEFT).place(relx= .77,rely= .1,anchor= tk.NW)


        ttk.Button(self.canvas, text= "Clear all tracking areas",
                  command= self.clear_all_rects).place(relx= .77, rely= .7, anchor= 'w')
        ttk.Button(self.canvas, text= "Clear last tracking area",
                  command= self.clear_last_rect).place(relx= .77, rely= .8, anchor= 'w')


        # Bind buttons to canvas
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

    def subpixel_option_changed(self,new_option):
        self.subpixel_option = self.subpixel_options.index(new_option)
        if self.subpixel_option==0:
            self.subpixel_tracking=True
        else:
            self.subpixel_tracking=False

    def update_margins(self):
        self.margin_y = abs(float(self.vmarg.get()))
        self.margin_x = abs(float(self.hmarg.get()))
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
            for co in self.track_areas_co_ords_initial:
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
                    self.track_areas_co_ords_initial.remove(co)

    def track_margins_screen(self):
        if self.mode=='Existing':
            if self.analysis_ran == True:
                if self.remove_image_distortion == True:
                    self.remove_image_distortion_func(self.photo_path+'\\'+self.start_photo)
                else:
                    self.image_base = Image.open(self.photo_path+'\\'+self.start_photo)
        else:
            if self.analysis_ran==True:
                self.track_areas_co_ords_initial = self.track_areas_co_ords

        self.img_tx = 0
        self.img_ty = 0

        # Add reference image to canvas
        self.photo_canvas_width = 0.75*self.wc
        self.photo_canvas_height = self.hc
        self.photo_canvas = tk.Canvas(self.canvas, width = self.photo_canvas_width,
                                      height = self.photo_canvas_height, highlightbackground = '#708090', bg = '#708090')

        self.canvas.create_window(0, 0,anchor='nw', window=self.photo_canvas)
        self.photo_canvas.update()  # wait till canvas is created


        self.img_bx,self.img_by = self.image_base.size

        ttk.Label(self.canvas, text= "Update margins as a fraction of image"+
                 "\ndimensions (if needed) using the boxes"+
                 "\nbelow and the \"Update margins\" button.",justify=tk.LEFT).place(relx= .77,rely= .15,anchor= tk.W)
        self.hmarg =  tk.StringVar(self.master, value=str(self.margin_x))
        self.vmarg =  tk.StringVar(self.master, value=str(self.margin_y))

        self.subpixel_options = ['Subpixel analysis on','Subpixel analysis off']
        self.subpixel_menu = ttk.OptionMenu(
            self.canvas,
            tk.StringVar(self,self.subpixel_option),
            self.subpixel_options[self.subpixel_option],
            *self.subpixel_options,
            command=self.subpixel_option_changed)
        self.subpixel_menu.place(relx= .77, rely= .4, anchor= tk.W)

        ttk.Label(self.canvas, text= "Vertical search margin [Number of pixels]").place(relx= .77, rely= .5, anchor= tk.W)
        ttk.Entry(self.canvas, width= 10,textvariable=self.vmarg).place(relx= .77, rely= .55, anchor= tk.W)
        ttk.Label(self.canvas, text= "Horizontal search margin [Number of pixels]").place(relx= .77, rely= .6, anchor= tk.W)
        ttk.Entry(self.canvas, width= 10,textvariable=self.hmarg).place(relx= .77, rely= .65, anchor= tk.W)
        ttk.Button(self.canvas, text= "Update margins", command= self.update_margins).place(relx= .77, rely= .72, anchor= 'w')

        self.photo_canvas.bind("<ButtonPress-2>", self.click_zoom)
        self.photo_canvas.bind("<B2-Motion>", self.drag_zoom)
        self.photo_canvas.bind("<ButtonRelease-2>", self.drag_release)
        self.photo_canvas.bind("<ButtonPress-3>", self.click_zoom)
        self.photo_canvas.bind("<B3-Motion>", self.drag_zoom)
        self.photo_canvas.bind("<ButtonRelease-3>", self.drag_release)
        self.scale_image()

        self.update_margins()


    def plot_option_changed(self,new_option):
        self.option_var = self.options.index(new_option)
        self.plotted_data.set_label(self.options[self.option_var])
        # self.ax.legend.remove()
        self.ax.legend(loc=1)
        if len(self.output_data)>0:
            if self.mode=='GoPro' or self.mode=='Mobile' or self.mode=='Existing':
                plt_x_data = np.array(self.output_data)
                plt_x_data = (plt_x_data-plt_x_data[0])*self.scale_factor

                plt_y_data = self.load_data[:len(plt_x_data)]
                try:
                    self.plotted_data.set_data(plt_x_data[:,self.option_var],plt_y_data)
                    self.ax.set_xlim(np.min(plt_x_data[:,self.option_var])-1e-10,np.max(plt_x_data[:,self.option_var]))

                except:
                    self.plotted_data.set_data(plt_x_data[self.option_var],plt_y_data)
                    self.ax.set_xlim(np.min(plt_x_data[self.option_var])-1e-10,np.max(plt_x_data[self.option_var]))

                self.ax.set_ylim(np.min(plt_y_data)-1e-10,np.max(plt_y_data))
            elif self.mode == 'GoPro - No LC' or self.mode=='Mobile - No LC':
                plt_y_data = np.array(self.output_data)
                plt_y_data = (plt_y_data-plt_y_data[0])*self.scale_factor

                plt_x_data = self.time_data[:len(plt_y_data)]
                try:
                    self.plotted_data.set_data(plt_x_data,plt_y_data[:,self.option_var])
                    self.ax.set_ylim(np.min(plt_y_data[:,self.option_var])-1e-10,np.max(plt_y_data[:,self.option_var]))

                except:
                    self.plotted_data.set_data(plt_x_data,plt_y_data[self.option_var])
                    self.ax.set_ylim(np.min(plt_y_data[self.option_var])-1e-10,np.max(plt_y_data[self.option_var]))

                self.ax.set_xlim(plt_x_data[0],plt_x_data[-1])


            self.fig.tight_layout()
        self.plot_canvas.draw()


    def outputs_screen(self):

        self.analysis_complete_label = None
        # Change button to run analysis
        self.next_button.configure(text='Run analysis',command=self.run_analysis)

        # Add reference image to canvas
        self.outputs_canvas_width = self.wc
        self.outputs_canvas_height = self.hc
        self.outputs_canvas =  tk.Canvas(self.canvas, width = self.outputs_canvas_width,
                                      height = self.outputs_canvas_height, highlightbackground = '#708090', bg = '#708090')
        self.canvas.create_window(0, 0,anchor='nw', window=self.outputs_canvas)
        self.outputs_canvas.update()

        self.photo_canvas_width = 0.5*self.wc
        self.photo_canvas_height = self.hc
        self.photo_canvas = tk.Canvas(self.outputs_canvas, width = self.photo_canvas_width,
                                      height = self.photo_canvas_height, highlightbackground = '#708090', bg = '#708090')

        self.outputs_canvas.create_window(0, 0,anchor='nw', window=self.photo_canvas)
        self.photo_canvas.update()  # wait till canvas is created

        self.img_bx,self.img_by = self.image_base.size
        self.scale_image()
        self.update_track_area()

        # Add outputs plot
        px = 1/plt.rcParams['figure.dpi']
        self.fig = plt.Figure(figsize=(self.photo_canvas_width*px,0.9*self.photo_canvas_height*px),dpi=80)
        self.ax = self.fig.add_subplot(111)
        if self.mode=='GoPro' or self.mode=='Mobile' or self.mode=='Existing':
            self.ax.set_xlabel('Displacement [mm]')
            self.ax.set_ylabel('Load [kN]')
        elif self.mode=='GoPro - No LC' or self.mode=='Mobile - No LC':
            self.ax.set_ylabel('Displacement [mm]')
            self.ax.set_xlabel('Date-time [HH:MM:SS]')
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

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
            command=self.plot_option_changed)
        self.option_menu.place(width=0.75*self.photo_canvas_width,relx= .75, rely= .95,anchor= tk.N)

        self.plotted_data, = self.ax.plot(self.load_data,self.output_data,'kx-',label=self.options[0])
        self.ax.legend(loc=1)
        self.fig.tight_layout()

        self.plot_canvas = FigureCanvasTkAgg(self.fig,master=self.outputs_canvas)
        self.plot_canvas.get_tk_widget().place(width=0.5*self.ww,relx= .5, rely= .0,anchor= tk.NW)

    def cancel_analysis(self):
        self.running_analysis = False
        self.next_button.configure(text='Run analysis',command=self.run_analysis)

    def get_load_data_and_timestamps(self):
        ts= []
        self.timestamps = []
        for fl in self.flnms:
            tmp = Image.open(fl)._getexif()[36867]
            tmp = time.mktime(datetime.datetime.strptime(tmp, "%Y:%m:%d %H:%M:%S").timetuple())
            ts.append(tmp)
            self.timestamps.append(datetime.datetime.fromtimestamp(tmp))

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
        self.load_ts = [datetime.datetime.fromtimestamp(i) for i in load_ts]
        self.load_data = np.interp(self.time_data,load_dat['ms']/1000,load_dat['Load_val'])

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

    def update_plots(self):
        self.scale_image()
        self.update_plotted_track_area()


        self.photo_canvas.update()

        # Update plots
        if self.mode=='GoPro' or self.mode=='Mobile' or self.mode=='Existing':
            plt_x_data = np.array(self.output_data)
            plt_x_data = (plt_x_data-plt_x_data[0])*self.scale_factor

            plt_y_data = self.load_data[:len(plt_x_data)]
            try:
                self.plotted_data.set_data(plt_x_data[:,self.option_var],plt_y_data)
                self.ax.set_xlim(np.min(plt_x_data[:,self.option_var])-1e-10,np.max(plt_x_data[:,self.option_var]))

            except:
                self.plotted_data.set_data(plt_x_data[self.option_var],plt_y_data)
                self.ax.set_xlim(np.min(plt_x_data[self.option_var])-1e-10,np.max(plt_x_data[self.option_var]))

            self.ax.set_ylim(np.min(plt_y_data)-1e-10,np.max(plt_y_data))
        elif self.mode == 'GoPro - No LC' or self.mode=='Mobile - No LC':
            plt_y_data = np.array(self.output_data)
            plt_y_data = (plt_y_data-plt_y_data[0])*self.scale_factor

            plt_x_data = self.time_data[:len(plt_y_data)]
            try:
                self.plotted_data.set_data(plt_x_data,plt_y_data[:,self.option_var])
                self.ax.set_ylim(np.min(plt_y_data[:,self.option_var])-1e-10,np.max(plt_y_data[:,self.option_var]))

            except:
                self.plotted_data.set_data(plt_x_data,plt_y_data[self.option_var])
                self.ax.set_ylim(np.min(plt_y_data[self.option_var])-1e-10,np.max(plt_y_data[self.option_var]))

            self.ax.set_xlim(plt_x_data[0],plt_x_data[-1])


        self.fig.tight_layout()

        self.plot_canvas.draw()


    def run_analysis(self):
        self.analysis_ran = True
        if self.analysis_complete_label is not None:
            self.analysis_complete_label.destroy()
            self.analysis_complete_label = None
        self.running_analysis = True
        self.next_button['text'] = 'Stop analysis'
        self.next_button['command'] = self.cancel_analysis
        self.master.update()
        self.track_areas_co_ords = [np.copy(i) for i in self.track_areas_co_ords_initial]
        photo_count = 0



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


        # Collect baseline time and load data
        if self.mode=='Existing':
            self.get_load_data_and_timestamps()
        else:
            if self.mode=='GoPro' or self.mode=='Mobile':
                ser = serial.Serial('COM'+str(self.com_no), baudrate=115200,write_timeout=0.1 ,timeout=0.1)
                self.load_data = []
            # Blank outputs for storing data
            self.output_data = []

            self.time_data = []
            self.photo_names = []
            # Take new image for analysis
            if self.mode=='GoPro' or self.mode=='GoPro - No LC':
                self.take_photo_GP()
            elif self.mode=='Mobile' or self.mode=='Mobile - No LC':
                self.take_photo_Mobile()

            if self.remove_image_distortion == True:
                self.remove_image_distortion_func(self.photo_url)
            else:
                self.image_base = self.ref_img
            if self.mode=='GoPro' or self.mode=='GoPro - No LC':
                self.photo_names.append(self.photo_url.split('/')[-1])
            elif self.mode=='Mobile' or self.mode=='Mobile - No LC':
                self.photo_names.append('Photo_'+str(photo_count).zfill(6))
                photo_count+=1
            # Get load cell reading
            if self.mode=='GoPro' or self.mode=='Mobile':
                try:
                    ser.write("!001:SYS?<CR>\r".encode('ascii'))
                    self.load_data.append(float(ser.readline().decode('ascii').strip()))
                    load_recorded=1
                except:
                    try:
                        ser.close()
                        ser = serial.Serial('COM'+str(self.com_no), baudrate=115200,write_timeout=0.1 ,timeout=0.1)
                    except:
                        ser = serial.Serial('COM'+str(self.com_no), baudrate=115200,write_timeout=0.1 ,timeout=0.1)
            self.time_data.append(datetime.datetime.now())

            if self.download_photos==True and self.mode=='Mobile' or self.mode=='Mobile - No LC':
                photo_output_path = self.output_path+'/Mobile_images_'+self.output_file+'_'+self.time_data[0].strftime("%Y_%m_%d_%H.%M.%S")
                os.makedirs(photo_output_path)
                self.image_base.save(photo_output_path+'//'+self.time_data[0].strftime("%Y_%m_%d_%H.%M.%S")+'.jpg')


        # Load initial image for analysis
        self.previous_image = self.image_base.copy()
        self.base_frame = np.asarray(self.previous_image.convert('L'))
        self.output_data = [[it for lst in self.track_areas_co_ords for it in lst[:2]]]

        draw = ImageDraw.Draw(self.previous_image)
        count= 1
        for it in self.track_areas_co_ords:
            draw.rectangle(((it[0], it[1]), (it[2], it[3])), outline="red",width=5)
            draw.rectangle(((it[0]-self.margin_x, it[1]-self.margin_y), (it[2]+self.margin_x, it[3]+self.margin_y)), outline="blue",width=5)
            draw.text((it[0], it[1]), str(count),anchor='rb',font=ImageFont.truetype("arial.ttf", 100),fill='red')
            count+=1

        if self.mode=='Existing':
            self.previous_image.save(self.output_path+'/Referance_image_'+self.output_file+'.png')
        else:
            self.previous_image.save(self.output_path+'/Referance_image_'+self.output_file+'_'+self.time_data[0].strftime("%Y_%m_%d_%H.%M.%S")+'.png')



        if self.download_track_areas==True:
            if self.mode=='Existing':
                track_areas_output_path = self.output_path+'/Tracked_areas_'+self.output_file
            else:
                track_areas_output_path = self.output_path+'/Tracked_areas_'+self.output_file+'_'+self.time_data[-1].strftime("%Y_%m_%d_%H.%M.%S")
            try:
                os.makedirs(track_areas_output_path)
            except:
                print('Output directory already exists')
            if self.mode=='Existing':
                self.previous_image.save(track_areas_output_path+'\\Tracked_areas_'+self.flnms[0].split('\\')[-1], "JPEG")
            else:
                self.previous_image.save(track_areas_output_path+'\\Tracked_areas_'+self.time_data[-1].strftime("%Y_%m_%d_%H.%M.%S")+'.jpg', "JPEG")


        i = 0

        while self.running_analysis==True:
            if self.mode=='Existing':
                i+=1
                if i>=len(self.flnms)-1:
                    self.running_analysis=False
                # Load new image for analysis
                self.current_picture = self.flnms[i]

                if self.remove_image_distortion == True:
                    self.remove_image_distortion_func(self.current_picture)
                else:
                    self.image_base = Image.open(self.current_picture)
            else:
                if self.mode=='GoPro' or self.mode=='Mobile':
                    load_recorded=0
                    while load_recorded==0:
                        # Get load cell reading
                        try:
                            ser.write("!001:SYS?<CR>\r".encode('ascii'))
                            self.load_data.append(float(ser.readline().decode('ascii').strip()))
                            load_recorded=1
                        except:
                            try:
                                ser.close()
                                ser = serial.Serial('COM'+str(self.com_no), baudrate=115200,write_timeout=0.1 ,timeout=0.1)
                            except:
                                ser = serial.Serial('COM'+str(self.com_no), baudrate=115200,write_timeout=0.1 ,timeout=0.1)
                self.time_data.append(datetime.datetime.now())
                # Take new photo
                if self.mode=='GoPro' or self.mode=='GoPro - No LC':
                    self.take_photo_GP()
                elif self.mode=='Mobile' or self.mode=='Mobile - No LC':
                    self.take_photo_Mobile()


                if self.remove_image_distortion == True:
                    self.remove_image_distortion_func(self.photo_url)
                else:
                    self.image_base = self.ref_img

                if self.mode=='GoPro' or self.mode=='GoPro - No LC':
                    self.photo_names.append(self.photo_url.split('/')[-1])
                elif self.mode=='Mobile' or self.mode=='Mobile - No LC':
                    self.photo_names.append('Photo_'+str(photo_count).zfill(6))
                    photo_count+=1
                    if self.download_photos==True:
                        self.image_base.save(photo_output_path+'//'+self.time_data[-1].strftime("%Y_%m_%d_%H.%M.%S")+'.jpg')

            new_frame = np.asarray(self.image_base.convert('L'))

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

            self.output_data = self.output_data+[[it for lst in self.track_areas_co_ords for it in lst[:2]]]

            if self.download_track_areas==True:
                source_img = self.image_base.copy()
                draw = ImageDraw.Draw(source_img)
                count= 1
                for it in self.track_areas_co_ords:
                    draw.rectangle(((it[0], it[1]), (it[2], it[3])), outline="red",width=5)
                    draw.rectangle(((it[0]-self.margin_x, it[1]-self.margin_y), (it[2]+self.margin_x, it[3]+self.margin_y)), outline="blue",width=5)
                    draw.text((it[0], it[1]), str(count),anchor='rb',font=ImageFont.truetype("arial.ttf", 100),fill='red')
                    count+=1
                if self.mode=='Existing':
                    source_img.save(track_areas_output_path+'\\Tracked_areas_'+self.flnms[i].split('\\')[-1], "JPEG")
                else:
                    source_img.save(track_areas_output_path+'\\Tracked_areas_'+self.time_data[-1].strftime("%Y_%m_%d_%H.%M.%S")+'.jpg', "JPEG")


            self.update_plots()
            # time.sleep(5)
            self.base_frame=new_frame


        if self.mode=='Existing':
            self.save_output_data_existing()
        elif self.mode=='GoPro' or self.mode=='Mobile' or self.mode=='GoPro - No LC' or self.mode=='Mobile - No LC':
            self.save_output_data_new()

        self.analysis_complete_label = ttk.Label(self.outputs_canvas, text= "Analysis has completed and data has been saved successfully to output file!")
        self.analysis_complete_label.place(relx= .5,rely= .01, anchor= 'n')

        self.next_button['text'] = 'Run analysis'
        self.next_button['command'] = self.run_analysis

    def save_output_data_new(self):
        output_dat = np.array(self.output_data)
        output_dat0 = (output_dat-output_dat[0])*self.scale_factor
        if self.mode=='GoPro' or self.mode=='Mobile':
            headers = ['Photo filename','Timestamp','Load cell output [kN]']

            output_data = [self.photo_names,list(self.time_data),list(self.load_data)]
        else:
            headers = ['Photo filename','Timestamp']

            output_data = [self.photo_names,list(self.time_data)]

        headers = headers+self.options+[i[:-17]+'position [px]' for i in self.options]
        for ik in range(len(self.track_areas_co_ords)):
            output_data = output_data+[list(output_dat0[:,(ik*2)]),list(output_dat0[:,(ik*2)+1])]

        # Add raw pixel output
        for ik in range(len(self.track_areas_co_ords)):
            output_data = output_data+[list(output_dat[:,(ik*2)]),list(output_dat[:,(ik*2)+1])]

        output_data = np.array(output_data).T
        with open(self.output_path+'/'+self.output_file+'_'+output_data[0,1].strftime("%Y_%m_%d_%H.%M.%S")+'.csv','w') as f:

            # Write tracked area information
            f.write('Tracking areas co-ordinates on reference image [px],')
            f.write('Tracking area,x0 [px],y0 [px],x1 [px],y1 [px]\n')
            ik = 1
            for sublist in self.track_areas_co_ords_initial:
                f.write(','+str(ik)+',')
                for item in sublist:
                    f.write(str(int(item))+',')
                f.write('\n')
                ik+=1
            f.write('\n')
            f.write('Tracking margins,')
            f.write('Horizontal tracking area margin [px],'+str(self.margin_x)+'\n')
            f.write(',Vertical tracking area margin [px],'+str(self.margin_y)+'\n\n')
            f.write('Reference line co-ordinates,')
            f.write('x0 [px],y0 [px],x1 [px],y1 [px]\n')
            f.write(','+str(int(self.rl0_x))+','+str(int(self.rl0_y))+','+str(int(self.rl1_x))+','+str(int(self.rl1_y))+'\n')
            f.write('\n')
            f.write('Reference length [mm],'+str(self.reference_length)+'\n\n')
            f.write('Scale factor [px to mm],'+str(self.scale_factor)+'\n\n')

            # Write headers to output file
            for item in headers:
                f.write(item + ',')
            f.write('\n')

            # Write output data
            for sublist in output_data:
                for item in sublist:
                    try:
                        f.write(str(item) + ',')
                    except:
                        f.write(item.strftime("%Y-%m-%d %H:%M:%S.%f"))
                f.write('\n')

        if self.download_photos==True and self.mode=='GoPro':
            self.analysis_complete_label=ttk.Label(self.outputs_canvas, text= "Downloading photos from GoPro, please wait...")
            self.analysis_complete_label.place(relx= .5,rely= .01, anchor= 'n')
            photo_output_path = self.output_path+'/GoPro_images_'+self.output_file+'_'+output_data[0,1].strftime("%Y_%m_%d_%H.%M.%S")
            os.makedirs(photo_output_path)

            gp_identifier = self.gopro.getMedia().split('/')[-2]
            [self.gopro.downloadMedia(gp_identifier, fl,photo_output_path+'/'+fl) for fl in self.photo_names]
            self.analysis_complete_label.destroy()

    def save_output_data_existing(self):
        output_dat = np.array(self.output_data)
        output_dat0 = (output_dat-output_dat[0])*self.scale_factor
        headers = ['Photo filename','Load cell timestamp','GoPro timestamp','Load cell output [kN]'
                    ]
        photo_names = [i.split('\\')[-1] for i in self.flnms]
        output_data = [photo_names[:len(output_dat)],list(self.load_ts[:len(output_dat)]),
                                                          list(self.timestamps[:len(output_dat)]),
                                                          list(self.load_data[:len(output_dat)])
                                                        ]

        headers = headers+self.options+[i[:-17]+'position [px]' for i in self.options]
        # Add scaled output
        for ik in range(len(self.track_areas_co_ords)):
            output_data = output_data+[list(output_dat0[:,(ik*2)]),list(output_dat0[:,(ik*2)+1])]

        # Add raw pixel output
        for ik in range(len(self.track_areas_co_ords)):
            output_data = output_data+[list(output_dat[:,(ik*2)]),list(output_dat[:,(ik*2)+1])]

        output_data = np.array(output_data).T
        with open(self.output_path+'/'+self.output_file+'.csv','w') as f:

            # Write tracked area information
            f.write('Tracking areas co-ordinates on reference image [px],')
            f.write('Tracking area,x0 [px],y0 [px],x1 [px],y1 [px]\n')
            ik = 1
            for sublist in self.track_areas_co_ords:
                f.write(','+str(ik)+',')
                for item in sublist:
                    f.write(str(int(item))+',')
                f.write('\n')
                ik+=1
            f.write('\n')
            f.write('Tracking margins,')
            f.write('Horizontal tracking area margin [px],'+str(self.margin_x)+'\n')
            f.write(',Vertical tracking area margin [px],'+str(self.margin_y)+'\n\n')
            f.write('Reference line co-ordinates,')
            f.write('x0 [px],y0 [px],x1 [px],y1 [px]\n')
            f.write(','+str(int(self.rl0_x))+','+str(int(self.rl0_y))+','+str(int(self.rl1_x))+','+str(int(self.rl1_y))+'\n')
            f.write('\n')
            f.write('Reference length [mm],'+str(self.reference_length)+'\n\n')
            f.write('Scale factor [px to mm],'+str(self.scale_factor)+'\n\n')

            # Write headers to output file
            for item in headers:
                f.write(item + ',')
            f.write('\n')

            # Write output data
            for sublist in output_data:
                for item in sublist:
                    try:
                        f.write(str(item) + ',')
                    except:
                        f.write(item.strftime("%Y-%m-%d %H:%M:%S.%f"))
                f.write('\n')








root = tk.Tk()
app = DIC_app(root)
root.mainloop()
