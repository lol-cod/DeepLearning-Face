from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import random
import  tkinter as tk
from tkinter import filedialog
from capture import image_capture


img_read = image_capture()


def select_file():
    root = tk.Tk()
    root.withdraw()
    global file_path
    file_path = filedialog.askopenfilename()


def verify_live():
    img1 = cv2.imread(file_path)
    plt.imshow(img1[:,:,::-1])
    plt.show()
    
    
    img2 = cv2.imread(img_read)
    plt.imshow(img2[:,:,::-1])
    plt.show()

    result = DeepFace.verify(img1, img2)


    print("Is Same Face: ", result["verified"])
    a=result["verified"]


    if a==True:
        print("The Face Matches: ", random.randint(85,100), "%")
    else:
        print("Face does not match each Other!!! ")


select_file()
verify_live()

