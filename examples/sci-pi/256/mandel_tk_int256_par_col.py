#!/usr/bin/python3

# Mandelbrot set using tkinter, multiprocessing and a C extension,
# along with matplotlib for colour maps
# MJR 7/2022  http://www.sci-pi.org.uk/maths/mandel_fixed256.html

# Licence: GPL v3
#
# Add multiprocesing
#
# Fork as fixed point integer version
#
# Version 1.11, Feb 2021
#   -- add some extra colour maps
#
# Version 1.1, Feb 2021
#   -- sqrt colour map checkbox added
#   -- more sig figs for coords
#   -- show position of next image when using back button
#
# Version 1.0, Jan 2021


import numpy as np
from matplotlib import cm, colormaps
from matplotlib.colors import ListedColormap

import tkinter as tk
import tkinter.filedialog as filedialog
import threading
import queue
import zlib
import re
import platform
import gmpy2
from time import time
from multiprocessing import Pool,RawArray
from functools import partial
import int256_mandel


# Defaults conveniently placed at the start
xmin=-2
xmax=0.5
ymin=-1.25
ymax=1.25
maxiter=255
size=600

bshift=17
xmin=int(xmin*2**bshift)
xmax=int(xmax*2**bshift)
ymin=int(ymin*2**bshift)
ymax=int(ymax*2**bshift)


def mandel_write_column(i,xmin,xmax,ymin,ymax,size,maxiter):
    global MS_data,bshift
    MS=np.frombuffer(MS_data,dtype=np.float64).reshape(size,size)
    x=xmin+int((i*(xmax-xmin))/size)
    MS[:,i]=int256_mandel.mandel_column(x,ymin,ymax,size,maxiter,bshift)

def init_mandel_write_column(MS_store,bshift_in):
    global MS_data,bshift
    MS_data=MS_store
    bshift=bshift_in

def mandel_calc():
    global M,calculating,q,maxiter,cmap,abort_calc,nproc,MS_store,bshift
    
    chunk=nproc*int(10/nproc)
    if (chunk==0): chunk=nproc
    if (chunk>size): chunk=size
    t=time()

    ppm_col_header=b"P6\n%i %i\n255\n"%(chunk,size)

    with Pool(processes=nproc, initializer=init_mandel_write_column,initargs=(MS_store,bshift)) as pool:
        for i in range (0,size-chunk+1,chunk):
            if abort_calc: break
            res=pool.map(partial(mandel_write_column,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,size=size,maxiter=maxiter),range(i,i+chunk),1)
            if (i>0):
                sblock=np.array(255*(cmap(M[:,i+1-chunk:i+1])[:,:,0:3]),
                                dtype=np.uint8)
                ppm_col_bytes=ppm_col_header+sblock.tobytes()
                tim=time()-t
                q.put(lambda i=i, bytes=ppm_col_bytes, tim=tim:
                      col_draw(i,bytes,tim))

        remainder=size%chunk
        if ((remainder>0)and(not abort_calc)):
            i+=chunk
            pool.map(partial(mandel_write_column,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,size=size,maxiter=maxiter),range(i,i+remainder))
        

    t=time()-t
    q.put(lambda t=t,maxiter=maxiter, size=size:
                 finished_calc(t,maxiter,size))
    
def col_draw(i,bytes,t):
    global canvas,legend,zoom,saved_cols
    img=tk.PhotoImage(data=bytes).zoom(zoom)
    saved_cols.append(img)
    canvas.create_image((i-9)*zoom,0,anchor="nw",image=img)
    legend.config(text="Calculating: %.1fs"%(t))
    
def finished_calc(t,maxiter,size):
    global history,legend,M,M_img,calculating,saved_cols,img_bytes,cmap
    global toggle_widgets,back_btn,recalc_btn,imgArea,img_array,sqrtmap
    global nproc
    for w in toggle_widgets:
        w.configure(state=tk.NORMAL)
    if len(history)<=1: back_btn.configure(state=tk.DISABLED)
    abort_calc=False
    recalc_btn.configure(text="Recompute")
    recalc_btn.configure(command=recalculate)
    legend.config(text="Finished in %.2fs, maxiter=%d, res=%d"
                  " nproc=%d bshift=%d"%
                  (t,maxiter,size,nproc,bshift))
    ppm_header=b"P6\n%i %i\n255\n"%(size,size)
    if (False):
        img_array=np.array(255*((cmap(sqrt(M)))[:,:,0:3]),dtype=np.uint8)
    else:
        img_array=np.array(255*((cmap(M))[:,:,0:3]),dtype=np.uint8)
    img_bytes=ppm_header+img_array.tobytes()
    M_img=tk.PhotoImage(data=img_bytes).zoom(zoom)
    canvas.itemconfig(imgArea,image=M_img)
    saved_cols=[]
    calculating=False
    
def recalculate(old_coords=[]):
    global r,coords_label,white_ppm_bytes,ppm_header
    global calculating,M,abort_calc,MS_store
    global xmin,xmax,ymin,ymax,bshift
    global history,size,canvas,imgArea,zoom,toggle_widgets
    if (calculating): return
    calculating=True
    # Fields might have been changed without enter being pressed
    input_res(0)
    input_maxiter(0)
    input_nproc(0)
    input_bshift(0)
    if (new_xmax-new_xmin)>0:
        xmin=new_xmin
        xmax=new_xmax
        ymin=new_ymin
        ymax=new_ymax
    history.append((xmin,xmax,ymin,ymax,size,maxiter,bshift))
    (old_size,junk)=M.shape

    MS_store=RawArray('d',size*size)
    M=np.frombuffer(MS_store,dtype=np.float64).reshape(size,size)
    np.copyto(M,255*np.ones((size,size)))

    if (size!=old_size):    
        white=255*np.ones((size,size,3),dtype=np.uint8)
        ppm_header=b"P6\n%i %i\n255\n"%(size,size)
        white_ppm_bytes=ppm_header+white.tobytes()

    M_img=tk.PhotoImage(data=white_ppm_bytes).zoom(zoom)
    canvas.itemconfig(imgArea,image=M_img)
    if (size!=old_size):
        calculating=False
        canvas.config(width=size*zoom,height=size*zoom)
        canvas_resize(junk)
        calculating=True
#    coords_label.configure(text=("(%.10f"%(xmin)).rstrip("0")+
#                           ("%+.10f"%(ymin)).rstrip("0")+
#                           ("i) to (%.10f"%(xmax)).rstrip("0")+
#                           ("%+.10f"%(ymax)).rstrip("0")+"i)")
#    coords_string=(("(%.12f"%(xmin/2**bshift)).rstrip("0")+
#                        ("%+.12f"%(ymin/2**bshift)).rstrip("0")+
#                        ("i) to (%.12f"%(xmax/2**bshift)).rstrip("0")+
#                        ("%+.12f"%(ymax/2**bshift)).rstrip("0")+"i)")
    coords_string=("("+fixedtostr(xmin)+fixedtostr(ymin,sign=1)+"i) to ("+
                   fixedtostr(xmax)+fixedtostr(ymax,sign=1)+"i)")
    coords_label.configure(state="normal")
    coords_label.delete(0,last=tk.END)
    coords_label.insert(tk.END,coords_string)
    coords_label.configure(state="readonly")
    coords_label.configure(width=len(coords_string)+1)

    r.reset()
    if (len(old_coords)==4):
        r.set(old_coords)
        old_coords=[]
    for w in toggle_widgets:
        w.configure(state=tk.DISABLED)
    abort_calc=False
    recalc_btn.configure(text="Abort")
    recalc_btn.configure(command=set_abort_calc)
    thread=threading.Thread(target=mandel_calc)
    thread.daemon=True
    thread.start()
    run_queue()

def set_abort_calc():
    global abort_calc
    abort_calc=True
    
def run_queue():
    global calculating,q,window
    while True:
        try:
            task=q.get(block=False)
        except queue.Empty:
            break
        else:
            window.after_idle(task)
            q.task_done()
    if calculating:
        window.after(100,run_queue)

def cursor_update(event):
    global cursor,canvas,size,xmin,xmax,ymin,ymax,zoom
    x,y=canvas.canvasx(event.x),canvas.canvasy(event.y)
    cursor.config(text="% .10f%+.10fi "%
                    ((xmin+x*((xmax-xmin)/(zoom*size)))/2**bshift,
                     (ymax-y*((ymax-ymin)/(zoom*size)))/2**bshift))


class Rubberband():
    
    def __init__(self,widget):
        global size
        self.active=False
        self.visible=False
        self.dragging=False
        self.stx=0
        self.sty=0
        self.x=0
        self.y=0
        self.dragx=0
        self.dragy=0
        self.band=None
        self.oldzoom=zoom
        widget.bind("<1>",self.press,add='+')
        widget.bind("<ButtonRelease-1>",self.release,add='+')
        widget.bind("<Motion>",self.motion,add='+')
        widget.bind("<Configure>",self.resize,add='+')
        self.widget=widget

    def clear(self):
        if (self.visible):
           self.widget.delete(self.band)

    def draw_band(self,x,y,w,h):
        self.band=self.widget.create_rectangle(x-2,y-2,(x+w)+2,(y+h)+2,
                                               width=1,outline='#777')
        
    def reset(self):
        self.clear()
        self.active=False
        self.visible=False
        self.oldzoom=zoom

    def set(self,coords):
        global new_xmin,new_xmax,new_ymin,new_ymax
        if self.visible:
            self.reset()
        try:
            xsize=int(self.widget['scrollregion'].split()[2])
            ysize=xsize
        except:
            xsize=int(self.widget['width'])
            ysize=xsize
        self.stx=int(round((coords[0]-xmin)*(xsize/(xmax-xmin))))
        self.x=int(round((coords[1]-xmin)*(xsize/(xmax-xmin))))
        self.sty=int(round((ymax-coords[3])*(ysize/(ymax-ymin))))
        self.y=int(round((ymax-coords[2])*(ysize/(ymax-ymin))))
        self.visible=True
        self.draw_band(self.stx,self.sty,
                                   self.x-self.stx,self.y-self.sty)
        [new_xmin,new_xmax,new_ymin,new_ymax]=coords
        
    def resize(self,event):
        global zoom
        if (zoom!=self.oldzoom):
            self.stx=zoom*(self.stx/self.oldzoom)
            self.sty=zoom*(self.sty/self.oldzoom)
            self.x=zoom*(self.x/self.oldzoom)
            self.y=zoom*(self.y/self.oldzoom)
            self.oldzoom=zoom
            if self.visible:
                self.widget.delete(self.band)
                self.draw_band(self.stx,self.sty,
                                   self.x-self.stx,self.y-self.sty)
        
    def press(self,event):
        x,y=self.widget.canvasx(event.x),self.widget.canvasy(event.y)
        if (self.visible):
            if ((x>min(self.stx,self.x))and
                (x<max(self.stx,self.x))and
               (y>min(self.sty,self.y))and
                (y<max(self.sty,self.y))):
                self.dragging=True
                self.active=True
                self.dragx=x
                self.dragy=y
                return True
        self.clear()
        self.stx,self.sty=x,y
        self.x,self.y=self.stx+1,self.sty+1
        self.active=True
        self.visible=True
        self.draw_band(self.stx,self.sty,1,1)

    def motion(self, event):
        if not self.active: return
        self.clear()

        if (self.dragging):
            x,y=self.widget.canvasx(event.x),self.widget.canvasy(event.y)
            dx=x-self.dragx
            dy=y-self.dragy
            self.stx+=dx
            self.x+=dx
            self.sty+=dy
            self.y+=dy
            self.dragx,self.dragy=x,y
            self.draw_band(self.stx,self.sty,self.x-self.stx,self.y-self.sty)
            return
         
        self.x,self.y=self.widget.canvasx(event.x),self.widget.canvasy(event.y)

# Constrain to square        
        w=abs(self.stx-self.x)
        h=abs(self.sty-self.y)
        w=min(w,h)
        h=w
        if (self.x>self.stx):
            self.x=self.stx+w
        else:
            self.x=self.stx-w
        if (self.y>self.sty):
            self.y=self.sty+h
        else:
            self.y=self.sty-h
        x=min(self.stx,self.x)
        w=abs(self.stx-self.x)
        y=min(self.sty,self.y)
        h=abs(self.sty-self.y)
        w=min(w,h)
        h=w

        if (self.band):
            self.widget.delete(self.band)
        self.draw_band(x,y,w,h)

    def release(self,event):
        global new_xmin,new_xmax,new_ymin,new_ymax

        self.active=False

        if self.dragging:
            self.dragging=False
        else:
            self.x=self.widget.canvasx(event.x)
            self.y=self.widget.canvasy(event.y)
# Constrain to square        
            w=abs(self.stx-self.x)
            h=abs(self.sty-self.y)
            w=min(w,h)
            h=w
            if (self.x>self.stx):
                self.x=self.stx+w
            else:
                self.x=self.stx-w
            if (self.y>self.sty):
                self.y=self.sty+h
            else:
                self.y=self.sty-h
            
        x=min(self.stx,self.x)
        w=abs(self.stx-self.x)
        y=min(self.sty,self.y)
        h=abs(self.sty-self.y)
        w=min(w,h)
        h=w
        self.widget.delete(self.band)
        self.draw_band(x,y,w,h)

        # Do not allow tiny selections
        if (w>5):
# Need full width of canvas, assumed square
# scrollregion is a string of four ints if in use, empty string otherwise
# could simply use size*zoom, but the below is more general
            try:
                xsize=int(self.widget['scrollregion'].split()[2])
                ysize=xsize
            except:
                xsize=int(self.widget['width'])
                ysize=xsize
            new_xmin=xmin+int((xmax-xmin)*x/xsize)
            new_ymin=ymax-int((ymax-ymin)*(y+h)/ysize)
            new_xmax=xmin+int((xmax-xmin)*(x+w)/xsize)
            new_ymax=ymax-int((ymax-ymin)*y/ysize)
        else:
            self.widget.delete(self.band)
            self.visible=False
            
def back():
    global new_xmin,new_xmax,new_ymin,new_ymax
    global maxiter,maxiter_entry,bshift,bshift_entry,size
    global history

    if len(history)<=1: return

    history.pop()
    (new_xmin,new_xmax,new_ymin,new_ymax,size,maxiter,new_bshift)=history.pop()
    maxiter_entry.delete(0,last=tk.END)
    maxiter_entry.insert(tk.END,"%d"%(maxiter))
    if ((new_bshift==bshift) and
        ((new_xmin<=xmin) and (new_ymin<=ymin) and
        (new_xmax>=xmax) and (new_ymax>=ymax)) and
        ((new_xmin!=xmin) or (new_ymin!=ymin) or
         (new_xmax!=xmax) or (new_ymax!=ymax))):
        old_coords=[xmin,xmax,ymin,ymax]
    else:
        old_coords=[]
    bshift=new_bshift;
    bshift_entry.delete(0,last=tk.END)
    bshift_entry.insert(tk.END,"%d"%(bshift))
    recalculate(old_coords)

def input_maxiter(event):
    global maxiter,maxiter_entry
    i=0
    try:
        i=int(maxiter_entry.get())
    except:
        pass
    if (i>0)and(i<100000): maxiter=i
    maxiter_entry.delete(0,last=tk.END)
    maxiter_entry.insert(tk.END,"%d"%(maxiter))

def input_nproc(event):
    global nproc,nproc_entry
    i=0
    try:
        i=int(nproc_entry.get())
    except:
        pass
    if (i>0)and(i<200): nproc=i
    nproc_entry.delete(0,last=tk.END)
    nproc_entry.insert(tk.END,"%d"%(nproc))

def input_bshift(event):
    global bshift,bshift_entry
    global xmin,xmax,ymin,ymax,new_xmin,new_xmax,new_ymin,new_ymax
    i=0
    try:
        i=int(bshift_entry.get())
    except:
        pass
    if (i>0)and(i<1000):
        if (i>bshift):
            xmin=xmin<<(i-bshift)
            xmax=xmax<<(i-bshift)
            ymin=ymin<<(i-bshift)
            ymax=ymax<<(i-bshift)
            new_xmin=xmin
            new_xmax=xmax
            new_ymin=ymin
            new_ymax=ymax
        elif (bshift>i):
            xmin=xmin>>(bshift-i)
            xmax=xmax>>(bshift-i)
            ymin=ymin>>(bshift-i)
            ymax=ymax>>(bshift-i)
            new_xmin=xmin
            new_xmax=xmax
            new_ymin=ymin
            new_ymax=ymax
        bshift=i
    bshift_entry.delete(0,last=tk.END)
    bshift_entry.insert(tk.END,"%d"%(bshift))

def input_res(entry):
    global size,canvas,res_entry,frame
    i=0
    try:
        i=int(res_entry.get())
    except:
        pass
    if i<1:
        i=min(frame.winfo_width(),frame.winfo_height())-4
    if (i>0)and(i<10000): size=i
    res_entry.delete(0,last=tk.END)
    res_entry.insert(tk.END,"%d"%(size))

    
def crc_init_table():
    global crc_table
    crc_table = [0] * 256
    for n in range(256):
        c = n
        for k in range(8):
            if c & 1:
                c = 0xedb88320 ^ (c >> 1)
            else:
                c = c >> 1
        crc_table[n] = c

def calc_crc(crc, buf):
    global crc_table
    c = crc
    for byte in buf:
        c = crc_table[int((c ^ byte) & 0xff)] ^ (c >> 8)
    return c

# Write a PNG file.
# Array should be np.uint8 and [:,:] (greyscale or paletted),
#                              [:,:,3] (RGB) or [:,:,4] (RGBA)
#
# palette, if present, should be [n:3] or [3n] as np.uint8
def save_png(array,palette=None):
    global xmin,ymin,xmax,ymax,maxiter

    if (array.dtype!=np.dtype('u1')):
        raise TypeError

    d=1
    try:
        (w,h,d)=array.shape
    except:
        try:
            (w,h)=array.shape
        except:
            raise IndexError("Unsupported array shape")
        
    if ((d!=1)and(d!=3)and(d!=4)):
        raise IndexError("Unsupported image depth")
    file=filedialog.asksaveasfile(mode="wb",filetypes=[('PNG','*.png')])
    if file==None:
        return

    if not (palette is None):
        if d!=1:
            raise IndexError("Palette present with 3D array")
        if len(palette.tobytes())>768:
            raise IndexError("Palette too long")
    
# Init CRC
    crc_init_table()
# Write PNG magic
    file.write(b"\x89PNG\r\n\x1A\n")
# IHDR
    buff=np.zeros((21),dtype=np.uint8)
    buff[3]=13
    buff[4]=ord('I')
    buff[5]=ord('H')
    buff[6]=ord('D')
    buff[7]=ord('R')

    buff[8]=(w&0xff000000)>>24
    buff[9]=(w&0xff0000)>>16
    buff[10]=(w&0xff00)>>8
    buff[11]=w&0xff

    buff[12]=(h&0xff000000)>>24
    buff[13]=(h&0xff0000)>>16
    buff[14]=(h&0xff00)>>8
    buff[15]=h&0xff

    buff[16]=8
    if (d==1):
        if palette is None:
            buff[17]=0  # greyscale
        else:
            buff[17]=3
    elif (d==3):
        buff[17]=2  # RGB
    elif (d==4):
        buff[17]=6  # RGBA
        
    file.write(buff.tobytes());

    ccrc=calc_crc(0xffffffff,buff[4:21].tobytes())
    ccrc=ccrc^0xffffffff
    file.write(ccrc.to_bytes(4,byteorder='big'))

    if not (palette is None):
        plen=len(palette.tobytes())
        buff=np.zeros((8),dtype=np.uint8)
        buff[2]=plen>>8
        buff[3]=plen&0xff
        buff[4]=ord('P')
        buff[5]=ord('L')
        buff[6]=ord('T')
        buff[7]=ord('E')
        file.write(buff.tobytes())
        ccrc=calc_crc(0xffffffff,buff[4:8])
        file.write(palette.tobytes())
        ccrc=calc_crc(ccrc,palette.tobytes())
        ccrc=ccrc^0xffffffff
        file.write(ccrc.to_bytes(4,byteorder='big'))
        
    
# optional tEXt
#    tEXt=b'Comment\x00(%.12f%+.12fi) to (%.12f%+.12fi) maxiter=%d'%(xmin,ymin,
#                                                        xmax,ymax,maxiter)
    tEXt=b'Comment\x00('+bytes(fixedtostr(xmin)+fixedtostr(ymin,sign=1)+
                               'i) to ('+fixedtostr(xmax)+
                               fixedtostr(ymax,sign=1)+
                               'i) maxiter=%d bshift=%d'%(maxiter,bshift),
                               'utf-8')
    buff2=b'tEXt'
    ccrc=calc_crc(0xffffffff,buff2)
    file.write(len(tEXt).to_bytes(4,byteorder='big'))
    file.write(buff2)
    ccrc=calc_crc(ccrc,tEXt)
    file.write(tEXt)
    ccrc=ccrc^0xffffffff
    file.write(ccrc.to_bytes(4,byteorder='big'))
    
# IDAT
    dat=np.zeros((size*(d*size+1)),dtype=np.uint8)
    if (d>1):
        for i in range (size):
            dat[i*(d*size+1)+1:(i+1)*(d*size+1)]=array[i,:,:].flatten()
    else:
        for i in range (size):
            dat[i*(size+1)+1:(i+1)*(size+1)]=array[i,:].flatten()

    zdat=zlib.compress(dat.tobytes(),level=8)
    zlen=len(zdat)
        
    buff2=b'IDAT'
    ccrc=calc_crc(0xffffffff,buff2)
    file.write(zlen.to_bytes(4,byteorder='big'))
    file.write(buff2)

    ccrc=calc_crc(ccrc,zdat)
    file.write(zdat)
    ccrc=ccrc^0xffffffff
    file.write(ccrc.to_bytes(4,byteorder='big'))
    
# IEND                 
    buff=np.zeros((8),dtype=np.uint8)
    buff[4]=ord('I')
    buff[5]=ord('E')
    buff[6]=ord('N')
    buff[7]=ord('D')
    file.write(buff.tobytes())
    ccrc=calc_crc(0xffffffff,buff[4:8].tobytes())
    ccrc=ccrc^0xffffffff
    file.write(ccrc.to_bytes(4,byteorder='big'))
    file.close()

class New_coords():

    def __init__(self,main):
        global xmin,ymin,xmax,ymax,toggle_widgets,recalc_btn
        self.dialog=dialog=tk.Toplevel(main)
        dialog.title("Coordinates")
        
        self.nxmn,self.nymn,self.nxmx,self.nymx=xmin,ymin,xmax,ymax

        x_frame=tk.Frame(dialog)
        x_frame.pack(padx=10,pady=10)

        dialog.bind('<Destroy>',self.clean_up)
        
        for w in toggle_widgets:
            w.configure(state=tk.DISABLED)
        recalc_btn.configure(state=tk.DISABLED)
            
        xmin_frame=tk.Frame(x_frame)
        xmin_frame.pack(side=tk.LEFT)
        xmin_label=tk.Label(xmin_frame,text="xmin: ")
        xmin_label.pack(side=tk.LEFT)
        xmin_entry=tk.Entry(xmin_frame,width=17)
        xmin_entry.insert(tk.END,("%.14f"%(xmin/2**bshift)).rstrip("0"))
        xmin_entry.bind("<Return>", self.input_coord)
        xmin_entry.bind("<FocusOut>", self.input_coord)
        xmin_entry.pack(side=tk.LEFT)

        xmax_frame=tk.Frame(x_frame)
        xmax_frame.pack(side=tk.LEFT)
        xmax_label=tk.Label(xmin_frame,text="xmax: ")
        xmax_label.pack(side=tk.LEFT,padx=(10,0))
        xmax_entry=tk.Entry(xmax_frame,width=17)
        xmax_entry.insert(tk.END,("%.14f"%(xmax/2**bshift)).rstrip("0"))
        xmax_entry.bind("<Return>", self.input_coord)
        xmax_entry.bind("<FocusOut>", self.input_coord)
        xmax_entry.pack(side=tk.LEFT)

        y_frame=tk.Frame(dialog)
        y_frame.pack(padx=10,pady=10)
        ymin_frame=tk.Frame(y_frame)
        ymin_frame.pack(side=tk.LEFT)
        ymin_label=tk.Label(ymin_frame,text="ymin: ")
        ymin_label.pack(side=tk.LEFT)
        ymin_entry=tk.Entry(ymin_frame,width=17)
        ymin_entry.insert(tk.END,("%.14f"%(ymin/2**bshift)).rstrip("0"))
        ymin_entry.bind("<Return>", self.input_coord)
        ymin_entry.bind("<FocusOut>", self.input_coord)
        ymin_entry.pack(side=tk.LEFT)

        ymax_frame=tk.Frame(y_frame)
        ymax_frame.pack(side=tk.LEFT)
        ymax_label=tk.Label(ymin_frame,text="ymax: ")
        ymax_label.pack(side=tk.LEFT,padx=(10,0))
        ymax_entry=tk.Entry(ymax_frame,width=17)
        ymax_entry.insert(tk.END,("%.14f"%(ymax/2**bshift)).rstrip("0"))
        ymax_entry.bind("<Return>", self.input_coord)
        ymax_entry.bind("<FocusOut>", self.input_coord)
        ymax_entry.pack(side=tk.LEFT)

        self.entries=[xmin_entry,ymin_entry,xmax_entry,ymax_entry]

        str_frame=tk.Frame(dialog)
        str_frame.pack(padx=10,pady=10)
        str_label=tk.Label(str_frame,text="coord string: ")
        str_label.pack(side=tk.LEFT)
        str_entry=tk.Entry(str_frame,width=64)
        str_entry.pack(side=tk.LEFT)
        str_entry.bind("<Return>", self.str_input_coord)

        self.str_entry=str_entry
        
        button_frame=tk.Frame(dialog)
        button_frame.pack(side=tk.BOTTOM,fill=tk.X,pady=10)
        check_btn=tk.Button(button_frame,text="Check",command=self.check_coord)
        check_btn.pack(side=tk.LEFT,expand=1)
        cancel_btn=tk.Button(button_frame,text="Cancel",command=dialog.destroy)
        cancel_btn.pack(side=tk.LEFT,expand=1)
        ok_btn=tk.Button(button_frame,text="OK",command=self.apply)
        ok_btn.pack(side=tk.LEFT,expand=1)
        self.ok_btn=ok_btn

        dialog.update()
        dialog.transient(main)
        dialog.resizable(False,False)


    def clean_up(self,event):
        global toggle_widgets,recalc_btn
        for w in toggle_widgets:
            w.configure(state=tk.NORMAL)
        recalc_btn.configure(state=tk.NORMAL)
        
    def apply(self):
        global new_xmin,new_ymin,new_xmax,new_ymax
        new_xmin,new_ymin=self.nxmn,self.nymn
        new_xmax,new_ymax=self.nxmx,self.nymx
        self.dialog.destroy()
        recalculate()

    def str_input_coord(self,event):
        entry=self.str_entry
        coord_str=entry.get()
        if (coord_str==""): return
        # find first four numbers in string. Yuk.
        rex="([+-]?[0-9.]+).*?([+-]?[0-9.]+).*?([+-]?[0-9.]+).*?([+-]?[0-9.]+)"
        match=re.search(rex,coord_str,flags=re.DOTALL)
        entry.delete(0,last=tk.END)
        if not match:
            print ("match failed")
            return
        c=[99,99,99,99]
        for i in range(4):
            try:
                c[i]=float(match.group(i+1))
            except:
                pass

        for i in range(4):
            if abs(c[i])>2.5:
                return

        x_range=c[2]-c[0]
        if x_range<0: return
        y_range=c[3]-c[1]
        if y_range<0: return

        d=c
        for i in range(4):
            d[i]=int(d[i]*2**bshift)
        [self.nxmn,self.nymn,self.nxmx,self.nymx]=d
        for i in range(4):
            self.entries[i].delete(0,last=tk.END)
            self.entries[i].insert(tk.END,("%.14f"%c[i]).rstrip("0"))
        
        entry.insert(tk.END,("(%.14f"%(c[0])).rstrip("0")+
                           ("%+.14f"%(c[1])).rstrip("0")+
                           ("i) to (%.14f"%(c[2])).rstrip("0")+
                           ("%+.14f"%(c[3])).rstrip("0")+"i)")

        self.ok_btn.configure(state=tk.DISABLED)
        

    def check_coord(self):
        global xmax,xmin,ymax,ymin

        # if nothing has happened, check for string entry
        if ((self.nxmx==xmax) and (self.nxmn==xmin) and
            (self.nymx==ymax) and (self.nymn==ymin)):
            self.str_input_coord(None)
        
        x_range=self.nxmx-self.nxmn
        y_range=self.nymx-self.nymn

        if (self.nymx==ymax):
            self.nymx=self.nymn+x_range
        elif (self.nxmx==xmax):
            self.nxmx=self.nxmn+y_range
        elif (self.nymn==ymin):
            self.nymn=self.nymx-x_range
        elif (self.nxmn==xmin):
            self.nxmn=self.nxmx-y_range
        else:
            self.nymx=self.nymn+x_range

        c=[self.nxmn,self.nymn,self.nxmx,self.nymx]

        for i in range(4):
            c[i]/=2**bshift
            self.entries[i].delete(0,last=tk.END)
            self.entries[i].insert(tk.END,("%.14f"%c[i]).rstrip("0"))

        self.str_entry.delete(0,last=tk.END)
        self.str_entry.insert(tk.END,("(%.14f"%(c[0])).rstrip("0")+
                           ("%+.14f"%(c[1])).rstrip("0")+
                           ("i) to (%.14f"%(c[2])).rstrip("0")+
                           ("%+.14f"%(c[3])).rstrip("0")+"i)")

        self.ok_btn.configure(state=tk.NORMAL)
        
    def input_coord(self,event):
        x=99
        entry=event.widget
        index=self.entries.index(entry)
        try:
            x=float(entry.get())
        except:
            pass
        
        if abs(x)<2.5:
            c=[self.nxmn,self.nymn,self.nxmx,self.nymx]
            c[index]=int(x*2**bshift)
            [self.nxmn,self.nymn,self.nxmx,self.nymx]=c
            self.ok_btn.configure(state=tk.DISABLED)
            self.entries[index].delete(0,last=tk.END)
            self.entries[index].insert(tk.END,("%.14f"%x).rstrip("0"))

def canvas_resize(event):
    global zoom,M,M_img,size,frame,imgArea,vbar,hbar,scrollbars,img_bytes
    f_size=min(frame.winfo_width(),frame.winfo_height())-4
    if (f_size>=size) and scrollbars:
        canvas.config(width=size,height=size)
        canvas.config(xscrollcommand='', yscrollcommand='')
        canvas.config(scrollregion='')
        vbar.destroy()
        hbar.destroy()
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)
        canvas.pack(expand=1,fill=tk.NONE)
        scrollbars=False
    if (f_size<size) and not scrollbars:
        canvas.pack_forget()
        hbar=tk.Scrollbar(frame,orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM,fill=tk.X)
        hbar.config(command=canvas.xview)
        vbar=tk.Scrollbar(frame,orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT,fill=tk.Y)
        vbar.config(command=canvas.yview)
        canvas.config(scrollregion=(0,0,size,size),width=f_size,height=f_size)
        canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.pack(side=tk.LEFT,expand=1,fill=tk.BOTH)
        scrollbars=True
        zoom=1
        return
    if (calculating): return
    new_zoom=max(1,int(f_size/size))
    if (zoom!=new_zoom):
        zoom=new_zoom
        M_img=tk.PhotoImage(data=img_bytes).zoom(zoom)
        canvas.configure(width=size*zoom,height=size*zoom)
        canvas.itemconfig(imgArea,image=M_img)
        
def input_map(event):
    global colours_list,cmap,M,img_bytes,img_array,M_img,ppm_header,zoom
    global sqrtmap
    try:
        new_map=colormaps.get_cmap(colours_list.get(colours_list.curselection()))
    except:
        print("Ooops")
        return
    cmap=new_map
    if (sqrtmap.get()):
        scale=np.linspace(0,1,1024)
        scale=np.square(scale)
        newmap=cmap(scale)
        cmap=ListedColormap(newmap)
        
    img_array=np.array(255*((cmap(M))[:,:,0:3]),dtype=np.uint8)
    img_bytes=ppm_header+img_array.tobytes()
    M_img=tk.PhotoImage(data=img_bytes).zoom(zoom)
    canvas.itemconfig(imgArea,image=M_img)

# fixed point conversion routines

def strtofixed(s):
    global bshift
    sign=1
    if (s[0]=='-'):
        sign=-1
        s=s[1:]
    pt=s.find('.')
    if (pt==-1): pt=len(s)
#    print("sign: ",sign)
#    print("int: ",s[:pt])
#    print("frac: ",s[pt+1:])
    result=int(s[:pt])<<bshift
    if (pt+1<len(s)):
        result+=int(0.5+(int(s[pt+1:])<<bshift)/(10**len(s[pt+1:])))
    result*=sign
#    print(result)
    return result

def fixedtostr(f,prec=-1,sign=0):
    global bshift
    if (prec==-1): prec=int(1+bshift/3.3)
    result=''
    if (sign==1): result='+'
    if(f<0):
        result='-'
        f*=-1
# round to nearest
    f+=int((1<<bshift)/(2*10**prec))
    i=f>>bshift
    f-=i<<bshift
    result+=str(i)
    result+='.'
    f=(f*10**bshift)>>bshift
    zt=10**(bshift-1)
    zeros=''
    while((f<zt)and(zt>1)):
        zeros+='0'
        prec-=1
        zt/=10
    result+=zeros[:prec]
    if len(zeros)>=prec: return result
    sf=str(f)
    result+=sf[:prec-len(zeros)]
    return result
      
# Start of main

if __name__ == '__main__':
    q=queue.Queue()
    new_xmin=xmin
    new_xmax=xmax
    new_ymin=ymin
    new_ymax=ymax
    abort_calc=False
    saved_cols=[]
    history=[]
    window=tk.Tk()
    window.title("Mandelbrot Set")
    window.minsize(520,450)
    nproc=2

    MS_store=RawArray('d',size*size)
    M=np.frombuffer(MS_store,dtype=np.float64).reshape(size,size)
    np.copyto(M,255*np.ones((size,size)))
    
    img_array=255*np.ones((size,size,3),dtype=np.uint8)
    ppm_header=b"P6\n%i %i\n255\n"%(size,size)
    zoom=1
    scale=np.linspace(0.0,1.0,num=256)

# If we use a read-only Entry, not a Label, the text displayed
# can be copied with the mouse
#coords_label=tk.Label(text="(%f%+fi) to (%f%+fi)"%(xmin,ymin,xmax,ymax))
    coords_label=tk.Entry(window,borderwidth=0,justify="center")
    coords_string=""
    coords_label.insert(tk.END,"(%f%+fi) to (%f%+fi)"%(xmin,ymin,xmax,ymax))
    coords_label.configure(state="readonly")
    coords_label.pack()
    legend=tk.Label(text="Idle")

    buttonframe1=tk.Frame(window,height=25)

# Specify width to prevent it changing when it becomes the Abort button
    recalc_btn=tk.Button(buttonframe1, text="Recompute", width=9,
                         command=recalculate)
    recalc_btn.pack(side=tk.LEFT,expand=1)

    coloursframe=tk.Frame(buttonframe1)
    coloursframe.pack(side=tk.LEFT,expand=1)
    colours_label=tk.Label(coloursframe,text="Colours:")
    colours_label.pack(side=tk.LEFT)
    colours_list=tk.Listbox(coloursframe,height=1,width=12,selectmode=tk.SINGLE)
    colours_list.config(exportselection=False)
    colours_list.pack(side=tk.LEFT,fill=tk.BOTH)
    sqrtmap=tk.IntVar()
    colours_btn=tk.Checkbutton(coloursframe,variable=sqrtmap,text="sq",
                               command=lambda: input_map(None))
    colours_btn.pack(side=tk.RIGHT)
    if (platform.system() != "Darwin"):
        colours_scroll=tk.Scrollbar(coloursframe,orient=tk.VERTICAL)
        colours_scroll.pack(side=tk.RIGHT,fill=tk.Y)
        colours_list.config(yscrollcommand=colours_scroll.set)
        colours_scroll.config(command=colours_list.yview)
    colours_list.insert(tk.END,"flag_r")
    colours_list.insert(tk.END,"gist_ncar")
    colours_list.insert(tk.END,"gist_rainbow_r+")
    colours_list.insert(tk.END,"gnuplot2")
    colours_list.insert(tk.END,"gray")
    colours_list.insert(tk.END,"hot")
    colours_list.insert(tk.END,"jet")
    colours_list.insert(tk.END,"magma")
    colours_list.insert(tk.END,"nipy_spectral")
    colours_list.insert(tk.END,"prism+")
    colours_list.insert(tk.END,"rainbow+")
    colours_list.insert(tk.END,"seismic")
# Set first colour of some maps to black
    for m in ["rainbow","gist_rainbow_r","prism"]:
        mdata=colormaps.get_cmap(m)(range(256))
        mdata[0]=[0,0,0,1]
        colormaps.register(name=m+"+",cmap=ListedColormap(mdata))
# Select magma and scroll listbox to show it
    magma_index=list(colours_list.get(0, tk.END)).index("magma")
    colours_list.select_set(magma_index)
    colours_list.yview_moveto(magma_index/colours_list.size())
    cmap=cm.magma
    colours_list.bind("<<ListboxSelect>>",input_map)

    resframe=tk.Frame(buttonframe1)
    resframe.pack(side=tk.LEFT,expand=1)
    res_label=tk.Label(resframe,text="Res:")
    res_label.pack(side=tk.LEFT)
    res_entry=tk.Entry(resframe,width=5)
    res_entry.insert(tk.END,"%d"%(size))
    res_entry.bind('<Return>',input_res)
    res_entry.pack(side=tk.LEFT)


    maxiterframe=tk.Frame(buttonframe1)
    maxiterframe.pack(side=tk.LEFT,expand=1)
    maxiter_label=tk.Label(maxiterframe,text="Max iter:")
    maxiter_label.pack(side=tk.LEFT)
    maxiter_entry=tk.Entry(maxiterframe,width=5)
    maxiter_entry.insert(tk.END,"%d"%(maxiter))
    maxiter_entry.bind('<Return>',input_maxiter)
    maxiter_entry.pack(side=tk.LEFT)

    buttonframe2=tk.Frame(window,height=25)

    save_btn=tk.Button(buttonframe2, text="Save",
                       command=lambda: save_png(np.array(np.rint(255*M),
                                                         dtype=np.uint8),
                                                np.array(255*cmap(scale),
                                                         dtype=np.uint8)[:,0:3]))
    save_btn.configure(state=tk.DISABLED)
    save_btn.pack(side=tk.LEFT,expand=1)

    set_btn=tk.Button(buttonframe2, text="Set",
                      command=lambda: New_coords(window))
    set_btn.configure(state=tk.DISABLED)
    set_btn.pack(side=tk.LEFT,expand=1)

    back_btn=tk.Button(buttonframe2, text="Back", command=back)
    back_btn.configure(state=tk.DISABLED)
    back_btn.pack(side=tk.LEFT,expand=1)

    bshiftframe=tk.Frame(buttonframe2)
    bshiftframe.pack(side=tk.LEFT,expand=1)
    bshift_label=tk.Label(bshiftframe,text="Bits:")
    bshift_label.pack(side=tk.LEFT)
    bshift_entry=tk.Entry(bshiftframe,width=4)
    bshift_entry.insert(tk.END,"%d"%(bshift))
    bshift_entry.bind('<Return>',input_bshift)
    bshift_entry.pack(side=tk.LEFT)

    nprocframe=tk.Frame(buttonframe2)
    nprocframe.pack(side=tk.LEFT,expand=1)
    nproc_label=tk.Label(nprocframe,text="nproc:")
    nproc_label.pack(side=tk.LEFT)
    nproc_entry=tk.Entry(nprocframe,width=3)
    nproc_entry.insert(tk.END,"%d"%(nproc))
    nproc_entry.bind('<Return>',input_nproc)
    nproc_entry.pack(side=tk.LEFT)

    cursor=tk.Label(buttonframe2,
                    text=" 0.0000000000+0.0000000000i ",font="TkFixedFont")
    cursor.pack(side=tk.RIGHT,expand=1)

    frame=tk.Frame(window,bd=0)
    canvas=tk.Canvas(frame,width=size,height=size,bd=1)
    canvas.pack(expand=1,fill=tk.NONE)
    zoom=1
    scrollbars=False
    vbar=hbar=None

    white=255*np.ones((size,size,3),dtype=np.uint8)
    img_array=white
    white_ppm_bytes=ppm_header+white.tobytes()
    img_bytes=white_ppm_bytes
    M_img=tk.PhotoImage(data=white_ppm_bytes)

    imgArea=canvas.create_image(0,0,anchor="nw",image=M_img)
    canvas.bind('<Motion>',cursor_update)
    r=Rubberband(canvas)

    legend.pack()
    buttonframe2.pack(side=tk.BOTTOM,fill=tk.X,pady=10)
    buttonframe1.pack(side=tk.BOTTOM,fill=tk.X,pady=5)
    frame.pack(expand=1,fill=tk.BOTH)
    frame.bind("<Configure>",canvas_resize)

    toggle_widgets=[back_btn, save_btn, set_btn, colours_list,
                    res_entry, maxiter_entry]

    calculating=False
    recalculate()

    window.mainloop()
