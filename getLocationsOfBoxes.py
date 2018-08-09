import Tkinter as tk
from Tkinter import *
import tkMessageBox
from tkFileDialog import askopenfilename
from PIL import ImageTk, Image
import os
import json
from collections import OrderedDict

event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))

class ExampleApp():
    def __init__(self):
        #tk.Tk.__init__(self)
	self.root = tk.Tk()
        self.x = self.y = 0
	self.frame = Frame(bd=2, relief=SUNKEN)
	self.frame.grid_rowconfigure(0, weight=1)
	self.frame.grid_columnconfigure(0, weight=1)
	self.xscroll = Scrollbar(self.frame, orient=HORIZONTAL)
	self.xscroll.grid(row=1, column=0, sticky=E+W)
	self.yscroll = Scrollbar(self.frame)
	self.yscroll.grid(row=0, column=1, sticky=N+S)
	self.canvas = Canvas(self.frame, width=1024, height=1024, bd=0, xscrollcommand=self.xscroll.set, yscrollcommand=self.yscroll.set)
	self.canvas.grid(row=0, column=0, sticky=N+S+E+W)
	self.xscroll.config(command=self.canvas.xview)
	self.yscroll.config(command=self.canvas.yview)
	self.frame.pack(fill=BOTH,expand=1)
        #self.canvas = tk.Canvas(self, width=512, height=512, cursor="cross")
        #self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

	self.menu = Menu(self.frame)
	self.root.config(menu=self.menu)
	self.filemenu = Menu(self.menu)
	self.menu.add_cascade(label="File", menu=self.filemenu)
	self.filemenu.add_command(label="New Image", command= self.new_image)
	#self.filemenu.add_command(label="Save Rotated/Flipped Images", command= self.flip_rotate_images)
	#self.filemenu.add_command(label="Load Template", command= self.flip_rotate_images)
	self.filemenu.add_command(label="Exit", command=self.root.quit)

        self.rect = None

        self.start_x = None
        self.start_y = None
	self.rects = []
	self.objClass = []
	self.json_images = []
	self.E1 = None
	#self.start_x_save = None
	#self.start_y_save = None
	#self.end_x_save = None
	#self.end_y_save = None
	

	self.curFile = os.getcwd()
        self._draw_image()
	
	self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
	json_image = OrderedDict([("image_path",self.curFile),("rects",self.rects),("object class",self.objClass)])
    	self.json_images.append(json_image)
	print self.json_images
	with open("Output.txt", "a+") as text_file:
    		text_file.write(json.dumps(self.json_images, indent =1))
	if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
		self.root.destroy()

    
    def _draw_image(self):
	print (self.curFile)
	dir_path = os.path.dirname(os.path.realpath(self.curFile))
	self.File = askopenfilename(initialdir=dir_path ,title='Choose an image.', filetypes = [("Image Files", ("*.jpg"))])
	
	self.curFile = self.File
	print (self.curFile)
	self.img = ImageTk.PhotoImage(file=self.File)
	self.canvas.create_image(0,0,image=self.img,anchor="nw")
	self.canvas.config(scrollregion=self.canvas.bbox(ALL))

	tFile = os.path.relpath(''.join(self.File), '/home/david/tensorbox/data/upload')

    def new_image(self):
	print (self.curFile)
	tempRect = list(self.rects)
	tempObj = list(self.objClass)
	json_image = OrderedDict([("image_path",self.curFile),("rects",tempRect),("object class",tempObj)])
    	self.json_images.append(json_image)
	print self.json_images
	#with open("Output.txt", "a+") as text_file:
    	#	text_file.write(json.dumps(self.json_images, indent =1))
	del self.rects[:]
	del self.objClass[:]
	dir_path = os.path.dirname(os.path.realpath(self.curFile))
	self.File = askopenfilename(initialdir=dir_path ,title='Choose an image.', filetypes = [("Image Files", ("*.jpg"))])
	
	self.curFile = self.File
	print (self.curFile)
	self.img = ImageTk.PhotoImage(file=self.File)
	self.canvas.create_image(0,0,image=self.img,anchor="nw")
	self.canvas.config(scrollregion=self.canvas.bbox(ALL))

	tFile = os.path.relpath(''.join(self.File), '/home/david/tensorbox/data/upload')

	#print self.json_images

    
	

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x, self.start_y = event2canvas(event, self.canvas)

        # create rectangle if not yet exist
        #if not self.rect:
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x+1, self.start_y+1, fill='', outline='red')
	

    def on_move_press(self, event):
        curX, curY = event2canvas(event, self.canvas)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    
    def clickConfirm(self):
		print self.json_images
		obj = self.E1.get()
		bbox = dict([("x1",self.start_x),("y1",self.start_y),("x2",self.endX),("y2",self.endY)])
        	self.rects.append(bbox)
		self.objClass.append(obj)

    def on_button_release(self, event):
	
	self.endX, self.endY = event2canvas(event, self.canvas)
	popup = tk.Tk()
        popup.wm_title("!")
        label = tk.Label(popup, text="Confirm Square")
        label.pack(side="top", fill="x", pady=10)
	self.E1 = tk.Entry(popup, text="Class of Object")
	self.E1.pack()
        B1 = tk.Button(popup, text="Okay", command = lambda: self.clickConfirm() or popup.destroy())
        B1.pack()
	B2 = tk.Button(popup, text="Cancel", command = popup.destroy)
        B2.pack()
        popup.mainloop()
	#print ("(%d, %d) / (%d, %d)" % (event.x,event.y,cx,cy))
	
        pass

    

if __name__ == "__main__":
    text_file = open("Output.txt", "a+")
  
    app = ExampleApp()
    app.root.mainloop()
    
