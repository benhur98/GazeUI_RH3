import tkinter
from tkinter import *
import serial
#error occurs if arduino not present in COM6 
#ard=serial.Serial('COM6',9600)
state=False
prog="You typed:"
def led_onf():
    global state
    if not state:
        state=True
        ard.write(str(chr(1)).encode())
    else:
        state=False
        ard.write(str(chr(2)).encode())
def left():
    global prog
    prog=prog+'H'
    label_text=tkinter.Label(window,compound=CENTER,text=prog)
    label_text.config(font=("Rouge", 30))
    label_text.config(foreground="#636161")
    label_text.config(background="#dddbdb")
    label_text.place(x=400,y=100)
    pass
def right():
    global prog
    prog=prog+'i'
    label_text=tkinter.Label(window,compound=CENTER,text=prog)
    label_text.config(font=("Rouge", 30))
    label_text.config(foreground="#636161")
    label_text.config(background="#dddbdb")
    label_text.place(x=400,y=100)

    pass

class Option(tkinter.Toplevel):
    def __init__(self,window):
        tkinter.Toplevel.__init__(self)
        self.title("select mode")
        self.geometry("250x100")
        self.resizable(width=False, height=False)
        self.configure(background='#dddbdb')
        def ioT():
            
            b_iot=Button(window,text="Turn LED ON/OFF",command=led_onf
                   ,activebackground="#9a9ea5",padx=100,pady=100,relief=GROOVE)
            b_iot.place(x=100,y=150)
            

            
            self.destroy()
            window.deiconify()
        def gaze():
            global prog
            label_text=tkinter.Label(window,compound=CENTER,text=prog)
            
            b_i1=Button(window,text="H",command=left
           ,activebackground="#9a9ea5",padx=100,pady=100,relief=GROOVE)
        
            b_i2=Button(window,text="i",command=right
           ,activebackground="#9a9ea5",padx=100,pady=100,relief=GROOVE)
            b_i1.place(x=90,y=150)
            b_i2.place(x=1100,y=150)
            label_text.config(font=("Rouge", 30))
            label_text.config(foreground="#636161")
            label_text.config(background="#dddbdb")
            label_text.place(x=400,y=100)

            self.destroy()
            window.deiconify()
            
        b_m1=Button(self,text="IoT mode",command=ioT
           ,activebackground="#9a9ea5",padx=4,relief=GROOVE)
        
        b_m2=Button(self,text="Gaze[TYPE] demo",command=gaze
           ,activebackground="#9a9ea5",padx=4,relief=GROOVE)
        b_m1.place(x=80,y=10)
        b_m2.place(x=80,y=60)
        self.update()
       
window=tkinter.Tk()
window.withdraw()
splash=Option(window)


window.wm_title("GAZE UI")
window.resizable(width=False,height=False)
window.geometry('1366x400')
window.configure(background='#dddbdb')
label=tkinter.Label(window,compound=CENTER,text="GAZE UI CONTROL UNIT")
label.config(font=("Rouge", 30))
label.config(foreground="#636161")
label.config(background="#dddbdb")
label.place(x=400,y=10)

##print(mode)
##if(mode is 1):
##    state=False
##    def led_onf():
##        if not state:
##            state=True
##            ard.write(str(chr(1)).encode())
##        else:
##            state=False
##            ard.write(str(chr(2)).encode())
##            
##            
##        
##    b_iot=Button(self,text="Turn LED ON/OFF",command=led_onf
##           ,activebackground="#9a9ea5",padx=10,pady=10,relief=GROOVE)
##    b_iot.place(x=400,y=100)
##    
##    pass
##if(mode is 2):
##    pass










window.mainloop()
