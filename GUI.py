# root.mainloop()
from tkinter import *
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

attack_list = ['Normal', 'DOS', 'Probe', 'R2L', 'U2R']
loaded_model1 = load_model('alexmodel.model')
print("Loaded Alexnet model from disk")

json_file = open('model_lstm1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_lstm = tf.keras.models.model_from_json(loaded_model_json)
model_lstm.load_weights("lstm_weight1.h5")
print("Loaded LSTM model from disk")

minivgg_model = load_model('model.model')
print("Loaded MiniVGG model from disk")

def pp(a):
    mylist.insert(END, a)

def predict(val):
    print(val)
    relist = []
    list1 = val.split(",")
    floatlist = [float(x) for x in list1]
    print(list1)
    print(floatlist)
    text = []
    text.append(floatlist)
    featalex = np.array(text)
    alex_scale = pickle.load(open("norm.pkl", "rb"))
    featalex = alex_scale.transform(featalex)
    featalex = np.reshape(featalex, (1, 20, 2, 1))
    preds = loaded_model1.predict(featalex)[0]
    alex_result = np.argmax(preds)
    print("alexnet result==>", alex_result)
    relist.append(alex_result)

    lstm_trans = pickle.load(open("minmaxlstm.pkl", "rb"))
    X_test = lstm_trans.transform(text)
    print(X_test)
    feat = np.array(X_test)
    print(feat.shape)
    feat = np.reshape(feat, (1, 40, 1))
    y = model_lstm.predict(feat)
    print(y)
    lstm_result = round(y[0][0])
    print("LSTM result==>", lstm_result)
    relist.append(lstm_result)

    featvgg = np.array(text)
    vgg_scale = pickle.load(open("norm.pkl", "rb"))
    featvgg = vgg_scale.transform(featvgg)
    featvgg = np.reshape(featvgg, (1, 20, 2, 1))
    preds = minivgg_model.predict(featvgg)[0]
    result_vgg = np.argmax(preds)
    print("Mini VGG result==>", result_vgg)

    relist.append(result_vgg)

    print(relist)
    finalindex = max(relist, key=relist.count)
    print(finalindex)

    print("Intrusion type==>", attack_list[finalindex])

    pp("Input data received")
    pp("Preprocessing started")
    pp("Feature scaling")
    pp("Loaded LSTM, AlexNet, and Mini VGGNet models")
    pp("Prediction using Loaded model")
    pp("Attack type: " + attack_list[finalindex])
    pp("============================")
    shrslt.config(text=attack_list[finalindex], fg="red")
    
  

def check_credentials():
    username = username_entry.get()
    password = password_entry.get()

    # Check if the username and password are valid
    if username == "admin" and password == "password":
        messagebox.showinfo("Login Successful", "Welcome, admin!")
        userHome()
        
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

def userHome():
    global root, mylist, shrslt
    root.destroy()  # Close the login page window

    root = Tk()
    root.geometry("1200x700+0+0")
    root.title("Home Page")

    # Load and display the background image
    image = Image.open("nbg.png")
    image = image.resize((1550, 800), Image.LANCZOS)
    bg_image = ImageTk.PhotoImage(image)
    bg_label = Label(root, image=bg_image)
    bg_label.place(x=0, y=0)

    # -----------------INFO TOP------------
    # lblinfo = Label(root, font=('aria', 20, 'bold'), text="NETWORK INTRUSION DETECTION", fg="white", bg="black",
    #                 bd=10, anchor='w')
    # lblinfo.place(x=350, y=50)

    # lblinfo3 = Label(root, font=('aria', 20), text="Enter Network features", fg="#000955", anchor='w')
    # lblinfo3.place(x=780, y=310)
    # E1 = Entry(root, width=30, font="veranda 20")
    # E1.place(x=650, y=360)

    # lblinfo4 = Label(root, font=('aria', 17), text="Process", fg="white", anchor='w', bg="black")
    # lblinfo4.place(x=180, y=250)
    # mylist = Listbox(root, width=50, height=20, bg="white")
    # mylist.place(x=80, y=300)
    # btntrn = Button(root, padx=16, pady=8, bd=6, fg="white", font=('ariel', 16, 'bold'), width=10, text="Detect",
    #                 bg="green", command=lambda: predict(E1.get()))
    # btntrn.place(x=800, y=420)

    # rslt = Label(root, font=('aria', 20), text="Attack type:", fg="black", bg="white", anchor=W)
    # rslt.place(x=630, y=580)
    # shrslt = Label(root, font=('aria', 20), text="", fg="blue", bg="white", anchor=W)
    # shrslt.place(x=790, y=580)
    lblinfo = Label(root, font=( 'aria' ,20, 'bold' ),text="NETWORK INTRUSION DETECTION",fg="white",bg="black",bd=10,anchor='w')
    lblinfo.place(x=350,y=50)
    
    lblinfo3 = Label(root, font=( 'aria' ,20 ),text="Enter Network features",fg="#000955",anchor='w')
    lblinfo3.place(x=780,y=310)
    E1 = Entry(root,width=30,font="veranda 20")
    E1.place(x=650,y=360)

    lblinfo4 = Label(root, font=( 'aria' ,17 ),text="Process",fg="white",anchor='w',bg="black")
    lblinfo4.place(x=180,y=250)
    mylist = Listbox(root,width=50, height=20,bg="white")

    mylist.place( x = 80, y = 300 )
    btntrn=Button(root,padx=16,pady=8, bd=6 ,fg="white",font=('ariel' ,16,'bold'),width=10, text="Detect", bg="green",command=lambda:predict(E1.get()))
    btntrn.place(x=800, y=420)
    # btnhlp=Button(root,padx=16,pady=8, bd=6 ,fg="white",font=('ariel' ,10,'bold'),width=7, text="Help?", bg="blue",command=lambda:predict(E1.get()))
    # btnhlp.place(x=50, y=450)
    rslt = Label(root, font=( 'aria' ,20, ),text="Attack type :",fg="black",bg="white",anchor=W)
    rslt.place(x=630,y=580)
    shrslt = Label(root, font=( 'aria' ,20, ),text="",fg="blue",bg="white",anchor=W)
    shrslt.place(x=790,y=580)
    root.mainloop()

#root = Tk()
#root.title("Login Page")
#root.geometry("1550x800+0+0")
#root.resizable(False, False)

# ==== Background ====
#bg = Image.open("C:/Instrusion detetion project/Instrusion detetion project/bagd.jpg")   # Keep the space if your file name has one
#bg = bg.resize((1550, 800), Image.LANCZOS)
#background_image = ImageTk.PhotoImage(bg)

#background_label = Label(root, image=background_image)
#background_label.place(x=0, y=0, relwidth=1, relheight=1)

# ==== Centered Form ====
#form_frame = Frame(root, bd=2, relief="solid", bg="white", padx=20, pady=20)
#form_frame.place(relx=0.5, rely=0.5, anchor="center")  # Middle of screen

#label_font = ("Times New Roman", 18)

#lblinfo = Label(root, 
    #font=('aria', 20, 'bold'),
    #text="DRIVEN INTRUSION DETECTION SYSTEM FOR NETWORK SECURITY",
    #fg="white", bg="black", bd=10
#)
#lblinfo.place(relx=0.5, y=80, anchor='n')  # 80 pixels from the top

# Username
#username_label = Label(form_frame, text="Username:", font=label_font, bg="white")
#username_label.pack(pady=10)
#username_entry = Entry(form_frame, font=("Times New Roman", 14), bd=2, relief="solid")
#username_entry.pack(pady=10)

# Password
#password_label = Label(form_frame, text="Password:", font=label_font, bg="white")
#password_label.pack(pady=10)
#password_entry = Entry(form_frame, show="*", font=("Times New Roman", 14), bd=2, relief="solid")
#password_entry.pack(pady=10)

# Login Button
#login_button = Button(form_frame, text="Login", command=check_credentials,
                      #font=("Times New Roman", 14, "bold"), bg="blue", fg="white")
#login_button.pack(pady=15)

#root.mainloop()

# ================== ABOUT PAGE (INSIDE LOGIN WINDOW) ==================
def open_about_page():
    # Hide login form elements temporarily
    form_frame.place_forget()
    lblinfo.place_forget()
    about_top_btn.place_forget()

    # Create about frame on top of login background
    about_frame = Frame(root, bg="#ffffffcc")
    about_frame.place(relx=0.5, rely=0.5, anchor="center", width=1200, height=700)

    # ===== Title =====
    title_label = Label(
        about_frame,
        text="About: Driven Intrusion Detection System for Network Security",
        font=("Times New Roman", 20, "bold"),
        fg="white", bg="black", pady=10
    )
    title_label.pack(fill=X)

    # ===== Description =====
    paragraph = (
        "This Intrusion Detection System (IDS) monitors network traffic and uses AI models such as "
        "AlexNet, LSTM, and MiniVGG to detect and classify attacks.\n\n"
        "It identifies categories like Normal, DoS, Probe, R2L, and U2R attacks using deep learning techniques.\n\n"
        "The system improves network security and reduces false positives, showcasing how Artificial Intelligence "
        "helps defend against cyber threats in real-time."
    )

    paragraph_label = Label(
        about_frame, text=paragraph,
        wraplength=1100, justify="left",
        font=("Times New Roman", 14), bg="#ffffffcc", fg="black", padx=20, pady=20
    )
    paragraph_label.pack(pady=10)

    # ===== Graph Images =====
    img_frame = Frame(about_frame, bg="#ffffffcc")
    img_frame.pack(pady=10)

    graph_paths = [
        "C:/Instrusion detetion project/graphs/graph1.png",
        "C:/Instrusion detetion project/graphs/graph2.png",
        "C:/Instrusion detetion project/graphs/graph3.png"
    ]

    for path in graph_paths:
        try:
            img = Image.open(path)
            img = img.resize((320, 220), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            lbl = Label(img_frame, image=photo, bg="#ffffffcc", bd=2, relief="solid")
            lbl.image = photo
            lbl.pack(side=LEFT, padx=15)
        except Exception as e:
            print(f"Error loading image: {e}")

    # ===== Back Button =====
    back_btn = Button(
        about_frame, text="Back to Login", bg="red", fg="white",
        font=("Times New Roman", 14, "bold"),
        command=lambda: close_about_page(about_frame)
    )
    back_btn.pack(pady=20)

# ================== CLOSE ABOUT PAGE ==================
def close_about_page(frame):
    frame.destroy()  # Remove about frame
    # Restore login page elements
    lblinfo.place(relx=0.5, y=80, anchor='n')
    form_frame.place(relx=0.5, rely=0.5, anchor="center")
    about_top_btn.place(x=30, y=30)

#till now

root = Tk()
root.title("Login Page")
root.geometry("1550x800+0+0")
root.resizable(False, False)

# ==== Background ====
bg = Image.open("C:/Instrusion detetion project/Instrusion detetion project/bagd.jpg")   # Keep the space if your file name has one
bg = bg.resize((1550, 800), Image.LANCZOS)
background_image = ImageTk.PhotoImage(bg)

background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# ==== Centered Form ====
form_frame = Frame(root, bd=2, relief="solid", bg="white", padx=20, pady=20)
form_frame.place(relx=0.5, rely=0.5, anchor="center")  # Middle of screen

label_font = ("Times New Roman", 18)

lblinfo = Label(root, 
    font=('aria', 20, 'bold'),
    text="DRIVEN INTRUSION DETECTION SYSTEM FOR NETWORK SECURITY",
    fg="white", bg="black", bd=10
)
lblinfo.place(relx=0.5, y=80, anchor='n')  # 80 pixels from the top

# Username
username_label = Label(form_frame, text="Username:", font=label_font, bg="white")
username_label.pack(pady=10)
username_entry = Entry(form_frame, font=("Times New Roman", 14), bd=2, relief="solid")
username_entry.pack(pady=10)

# Password
password_label = Label(form_frame, text="Password:", font=label_font, bg="white")
password_label.pack(pady=10)
password_entry = Entry(form_frame, show="*", font=("Times New Roman", 14), bd=2, relief="solid")
password_entry.pack(pady=10)

# Login Button
login_button = Button(form_frame, text="Login", command=check_credentials,
                      font=("Times New Roman", 14, "bold"), bg="blue", fg="white")
login_button.pack(pady=15)

root.mainloop()










