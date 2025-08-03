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
sound6 = pygame.mixer.Sound("6.wav")
sound7 = pygame.mixer.Sound("7.wav")
sound8 = pygame.mixer.Sound("8.wav")
sound9 = pygame.mixer.Sound("9.wav")
sound10 = pygame.mixer.Sound("10.wav")
df=pd.read_csv("hand_number_v3.csv")

df_columns = df.columns[:-1].tolist()
print(len(df_columns))
window=tk.Tk()
window.geometry("960x720")
window.title("تعليم الأرقام بالأصابع")
ctk.set_appearance_mode("dark")

numberlabel = ctk.CTkLabel(window, height=40,width=120,font=("Arial", 20), text_color="Black",padx=10)
numberlabel.place(x=430, y=10)
numberlabel.configure(text="الرقم")

numberbox = ctk.CTkLabel(window, height=40,width=120,font=("Arial", 20), text_color="Black",fg_color="blue")
numberbox.place(x=430, y=41)
numberbox.configure(text="0")

frame =tk.Frame(width=960, height=720)
frame.place(x=10, y=90)

lmain=tk.Label(frame)
lmain.place(x=0, y=0)

mp_drawing = mp.solutions.drawing_utils
mp_hands= mp.solutions.hands
hands=mp_hands.Hands(min_tracking_confidence=0.5, min_detection_confidence=0.5)

model = tf.keras.models.load_model("hand1_5_v3.h5")
print(model)

cap = cv2.VideoCapture(0)

bodylang_prob=np.array([0,0,0,0,0,0,0,0,0,0])
label=''

sound_played1 = False
sound_played2 = False
frame_count = 0
prev_label = None
def detect():
    global label, bodylang_prob,prev_label,frame_count
    ret,frame=cap.read()
    frame = cv2.flip(frame, 1)
    image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=hands.process(image)
    image.flags.writeable=True
    #image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)  
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
     for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS)

  
     try:
      left_hand = [0] * (21 * 3)
      right_hand = [0] * (21 * 3)
      for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[i].classification[0].label  # 'Left' or 'Right'
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            flat = np.array(landmarks).flatten().tolist()

            if hand_label == 'Left':
                left_hand = flat
            else:
                right_hand = flat
      row = left_hand+right_hand
      X= pd.DataFrame([row],columns=df_columns)
      print(X)
      
      #bodylang_prob = model.predict_proba(X)[0]
      #bodyland_class = model.predict(X)[0]

      #bodylang_prob = model.predict_proba(X)[0]
      frame_count += 1
      if frame_count % 5 == 0: 
       bodylang_prob = model.predict(X)[0]
       confidence = max(bodylang_prob)
       if confidence > 0.75:
        label = np.argmax(bodylang_prob)+1
        
      else:
       label = None 
      
     except Exception as e:
      label=None
    else:
       label=None
    resized_img = cv2.resize(image, (960, 720))
    #img= image[:,:600,:]
    imagarr=Image.fromarray(resized_img)
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
    if label==6 and prev_label != 6:
        sound6.play()   
    if label==7 and prev_label != 7:
        sound7.play()     
    if label==8 and prev_label != 8:
        sound8.play()   
    if label==9 and prev_label != 9:
        sound9.play()   
    if label==10 and prev_label != 10:
        sound10.play()                                
    prev_label = label   

    
detect()

window.mainloop()