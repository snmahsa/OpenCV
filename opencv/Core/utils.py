#Import library
import cv2
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt


def read_img(path, mode = -1):
    """
    0 : gray scale
    1 : colorfull
    -1 : same
    """
    return cv2.imread(path, mode)

def upload_Image_of_local():

    #get image path of local 
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    #select new path for saving
    dest_directory = filedialog.askdirectory()
    if file_path and dest_directory != 0 :
        #read image
        image = cv2.imread(file_path)
        #save
        file_name = os.path.basename(file_path)
        cv2.imwrite(os.path.join(dest_directory, file_name), image)
        # shutil.copy(file_path,dest_directory)
        # image = cv2.imread(file_path)
        return file_name
    else:
        print("No file selected")

def read_img_from_folder():
    default_folder_path = 'Img/'
    #select file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir=default_folder_path)
    #read image
    image = cv2.imread(file_path)
    #check successfully
    if image is None:
        print("Unable to read image")
        return None
    else:
        return image
    
def show_img_cv2(image):
    """
    q for exist 
    """
    cv2.imshow('img_window',image)
    while True:
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
    cv2.destroywindow('img_window')    

def show_img_plot(image):
    plt.imshow(image, aspect='equal')

def show_in_resizable_window(image):
    #creat window
    cv2.namedWindow('resizable', cv2.WINDOW_NORMAL)
    #show
    cv2.imshow('resizable', image)
    cv2.waitKey(0)
    cv2.destroyWindow('v')

def show_in_selected_resize_Window(image, x, y):
    #creat window
    cv2.namedWindow('resizable', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('resizable', x, y)
    #show
    cv2.imshow('resizable', image)
    cv2.waitKey(0)
    cv2.destroyWindow('resizable')

def resize_img(image, x, y):
    return cv2.resize(image, (x,y))

def ratio_resize_img(image, fx, fy):
    return cv2.resize(image,None, fx=fx, fy=fy)

def get_stack(image1, image2):
    return np.hstack((image1, image2))

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def extraction_color_in_img(image, low:tuple, upper:tuple):
    """
    image must be BGR
    lower and upper must be tuple

    """
    # Check if the image is in BGR format
    if image[0,0,0] == image[0,0,2]:
        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower_range = np.array([low])
        upper_range = np.array([upper])
        binary_mask = creat_mask(hsv_image, lower_range, upper_range)
        return apply_mask(image, binary_mask)
    else:
        hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        lower_range = np.array([low])
        upper_range = np.array([upper])
        binary_mask = creat_mask(hsv_image, lower_range, upper_range)
        return apply_mask(image, binary_mask)
    
def creat_mask(hsv_image, lower_range:tuple, upper_range:tuple):
    return cv2.inRange(hsv_image, lower_range, upper_range)

@staticmethod
def apply_mask(image, binary_mask):
    return cv2.bitwise_and(image, image, mask=binary_mask)

def eriosion(mask, times=1):
    kernel = np.ones(shape = (3,3), dtype = np.uint8)
    mask_erode = cv2.erode(mask, kernel, iterations = times)
    return mask_erode

def dilation(mask_erode, times = 1):
    mask_dilate = cv2.erode(mask_erode, kernel, iterations = items)
    return mask_dilate

def morph_open(mask,times = 1):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    return cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = times)

def morgh_cllose(mask, times =1):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    return cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel, iterations = times)
#######################Video#########################
def upload_Video_of_local():

    #get image path of local 
    root = tk.Tk()
    #hide  tkinter window
    root.withdraw()
    #select file
    file_path = filedialog.askopenfilename()
    #select new path for saving
    dest_directory = filedialog.askdirectory()
    #check
    if file_path and dest_directory != 0 :
        #read video
        cap = cv2.VideoCapture(file_path)
        #save
        file_name = os.path.basename(file_path)
        #
        output_path = dest_directory + "/" + file_name
        #
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #
        fps = cap.get(cv2.CAP_PROP_FPS)
        #
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                    if cv2.waitKey(25) == ord('q'):
                        break
                else:
                    break
        else:
            pass            

        cap.release()
        cv2.destroyAllWindows()   
        # return file_name
    else:
        print("No file selected")

def read_video_from_folder():
    default_folder_path = 'Video/'
    #select file
    root = tk.Tk()
    #hide tkinter window
    root.withdraw()
    #select file
    file_path = filedialog.askopenfilename(initialdir=default_folder_path)
    #check if file is selected
    if not file_path:
        print("No file selected")
        return None
    #open video
    cap = cv2.VideoCapture(file_path)
    #check successfully
    if not cap.isOpened():
        print("Unable to read video")
        return None
    else:
        return cap
    
def read_frames(cap): 

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) == ord('q'):
                    break
            else:
                break
    else:
        pass            

    cap.release()
    cv2.destroyAllWindows()
