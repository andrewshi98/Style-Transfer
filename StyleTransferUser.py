import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from StyleTransfer import StyleTransfer, ImgUtil

from tkinter import *
from tkinter.ttk import *
import threading
import time

root = Tk()
root.title("Style Transfer Kit")
root.geometry("300x150")
progress = Progressbar(root, orient = HORIZONTAL, 
              length = 200, mode = 'determinate', maximum = 100)

ImageSize = 128

def display_image(img):
    img = transforms.ToPILImage()(img.cpu().clone().squeeze(0))
    plt.imshow(img)
    plt.show()

def chooseStyleImage():
    root.styleImage = filedialog.askopenfilename()

def chooseContentImage():
    root.contentImage = filedialog.askopenfilename()

class JobWorker(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        style_transfer = StyleTransfer(root.styleImage, root.contentImage, IMG_SIZE=ImageSize)
        result = style_transfer.style_transfer(ImgUtil.load_image(root.contentImage, ImageSize), progress=progress)
        display_image(result)
        plt.ioff()
        plt.show()

class RootUpdater(threading.Thread):
    def __init__(self, threadID, name, counter, root): 
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.root = root
    def run(self):
        while True:
            progress.update()
            time.sleep(0.5)

def executeStyleTransfer():
    if root.styleImage != None and root.contentImage != None:
        thread = JobWorker(1, "Thread", 1)
        thread.start()


topFrame = Frame(root)
topFrame.pack()
Button(topFrame, text="Choose Style", command = chooseStyleImage).pack(side=LEFT)
Button(topFrame, text="Choose Content", command = chooseContentImage).pack(side=LEFT)
Button(root, text="Style Transfer", command = executeStyleTransfer).pack(side=BOTTOM)
progress.pack(side=BOTTOM)

updater = RootUpdater(2, "RootUpdater", 2, root)
updater.start()

mainloop()
