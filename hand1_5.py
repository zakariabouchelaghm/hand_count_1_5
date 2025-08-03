import tkinter as tk
import customtkinter as ctk
import pandas as pd 
import numpy as np
import pygame
import mediapipe as mp
import cv2 
from PIL import Image, ImageTk

import tensorflow as tf
pygame.mixer.init()
sound1 = pygame.mixer.Sound("1.wav")
sound2 = pygame.mixer.Sound("2.wav")
sound3 = pygame.mixer.Sound("3.wav")
sound4 = pygame.mixer.Sound("4.wav")
sound5 = pygame.mixer.Sound("5.wav")
df=pd.read_csv("hand_number_dataset.csv")

df_columns = df.columns[:-1].tolist()
print(len(df_columns))
window=tk.Tk()
window.geometry("480x700")
window.title("تعليم الأرقام بالأصابع")
ctk.set_appearance_mode("dark")

numberlabel = ctk.CTkLabel(window, height=40,width=120,font=("Arial", 20), text_color="Black",padx=10)
numberlabel.place(x=175, y=10)
numberlabel.configure(text="الرقم")

numberbox = ctk.CTkLabel(window, height=40,width=120,font=("Arial", 20), text_color="Black",fg_color="blue")
numberbox.place(x=175, y=41)
numberbox.configure(text="0")

frame =tk.Frame(width=480, height=480)
frame.place(x=10, y=90)

lmain=tk.Label(frame)
lmain.place(x=0, y=0)

mp_drawing = mp.solutions.drawing_utils
mp_hands= mp.solutions.hands
hands=mp_hands.Hands(min_tracking_confidence=0.5, min_detection_confidence=0.5)

model = tf.keras.models.load_model("hand1_5_v2.h5")
print(model)

cap = cv2.VideoCapture(0)

bodylang_prob=np.array([0,0,0,0,0])
label=''

sound_played1 = False
sound_played2 = False
prev_label = None
def detect():
    global label, bodylang_prob, sound_played1, sound_played2,prev_label
    ret,frame=cap.read()
    image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=hands.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)  
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
     for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS)

  
     try:
      row =np.array( [[res.x, res.y ,res.z] for res in  results.multi_hand_landmarks[0].landmark]).flatten().tolist()
      
      X= pd.DataFrame([row],columns=df_columns)
      
      
      #bodylang_prob = model.predict_proba(X)[0]
      #bodyland_class = model.predict(X)[0]

      #bodylang_prob = model.predict_proba(X)[0]
      bodylang_prob = model.predict(X)[0]
      confidence = max(bodylang_prob)
      if confidence > 0.75:
       print(bodylang_prob)
       label = np.argmax(bodylang_prob)+1
       print(label)
      else:
       label = None 
      
     except Exception as e:
      label=None
    else:
       label=None

    img= image[:,:460,:]
    imagarr=Image.fromarray(img)
    imagtk=ImageTk.PhotoImage(imagarr)
    lmain.imgtk=imagtk
    lmain.configure(image=imagtk)
    lmain.after(80, detect)
    numberbox.configure(text=label)
    

    if label==1 and prev_label != 1:
        sound1.play()
    if label==2 and prev_label != 2:
        sound2.play() 
    if label==3 and prev_label != 3:
        sound3.play()     
    if label==4 and prev_label != 4:
        sound4.play() 
    if label==5 and prev_label != 5:
        sound5.play()                 
    prev_label = label   

    
detect()

window.mainloop()