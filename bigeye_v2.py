#  bigeye.py: program for eyeball inspection of images
#  original version: Neal Jackson, 01.02.17
#  
#  for each picture you think is a lens, hit 1,2,3 or 4 depending on confidence
#  hit 5 to go to the next screen.
#
import numpy as np,scipy as sp, matplotlib,glob,os,sys,time
from scipy import ndimage
import astropy; from astropy.io.fits import getdata
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
global img,s, img_bw
from time import sleep
#  Change the following things if needed:
PLOTSIZE = (32.5,25)    # size of plot on screen (32.5x25 is a big screen)
CUT = [25,175]
NX,NY = 8,4
subplots=[0.01,1.0,0.05,0.99]
plt.rcParams['image.origin']='lower'
plt.rcParams['image.interpolation']='nearest'
# ==============================================================
event_col = ['white','yellow','orange','red','purple']

def replace_plots (thisfile):
    try:
        f = open('bigeye_events')
    except:
        return
    for line in f:
        obj = line.split()[0].split('/')[-1]
        marker = int(line.split()[1])
        for yidx in range(NY):
            for xidx in range(NX):
                print (thisfile[yidx,xidx],obj)
                if thisfile[yidx,xidx].split('/')[-1]==obj:
                    plt.plot([xidx*npix+10],[(NY-yidx-1)*npix+10],marker='o',color=event_col[marker])
        

def onkeyclick (event):
    global thisfile, npix, img, img_bw, s, subplots
    t = thisfile[0] if thisfile.ndim==3 else thisfile
    if event.xdata==None:
        print('Did not read button, try moving the mouse first')
        return
    elif event.key not in ['1','2','3','4','5',';','#']:
        print ('Allowed key options are 1,2,3,4,5,;,#')
        return
    elif event.key=='5':
        plt.close('all')
    elif event.key==';':
        plt.clf()
        plt.subplot(111,xticks=[],yticks=[])
        plt.subplots_adjust(left=subplots[0],right=subplots[1],\
                bottom=subplots[2],top=subplots[3])
        plt.imshow(img_bw,vmin=0.0,vmax=1.0)
        replace_plots (thisfile)
        plt.plot([0.0],[0.0],'wx')
        plt.draw()
    elif event.key=='#':
        plt.clf()
        plt.subplot(111,xticks=[],yticks=[])
        plt.subplots_adjust(left=subplots[0],right=subplots[1],\
                bottom=subplots[2],top=subplots[3])
        plt.imshow(img,vmin=0.0,vmax=1.0)
        replace_plots (thisfile)
        plt.plot([0.0],[0.0],'wx')
        plt.draw()
    else:
        try:
            xidx,yidx = int(event.xdata/npix),int(event.ydata/npix)
            print(event.xdata,event.ydata)
            idfile = t[t.shape[0]-yidx-1,xidx]
        except:
            print ('No image at position requested')
            return
        f=open('bigeye_events','a')
        f.write('%s %s\n'%(idfile,event.key))
        f.close()    
        sys.stdout.write('%s %s\n'%(idfile.split('/')[-1],event.key))
        plt.plot([xidx*npix+10],[yidx*npix+10],marker='o',color=event_col[int(event.key)])
        plt.draw()

def update(val):
    global img,img_bw,s,subplots
#    sys.stdout.write ('Adjusting colour scale, please wait...')
    plt.subplot(111,xticks=[],yticks=[])
    plt.subplots_adjust(left=subplots[0],right=subplots[1],\
                bottom=subplots[2],top=subplots[3])
#    print('value:',s.val)
    plt.imshow(img,vmin=0.,vmax=s.val)
    plt.draw()


def array_interp (a, newshape, order=3):   # newshape is (y,x) i.e. (nrow,ncol)
    ax,ay = float(a.shape[1]),float(a.shape[0])
    nx,ny = float(newshape[1]),float(newshape[0])
    xc = np.tile(np.arange(0.0,ax-0.5*ax/nx,ax/nx)+0.5*ax/nx,int(ny))
    yc = np.repeat(np.arange(0.0,ay-0.5*ay/ny,ay/ny)+0.5*ay/ny,nx)
    allc = np.vstack((yc,xc))
    return sp.ndimage.map_coordinates(a,allc,order=order).reshape(newshape)


def npy_addtext(texts):
    global npix
    for i in range(NY):
        for j in range(NX):
            px,py = j*npix, NY*npix-npix*(i+1)
            plt.text(px,py,texts[i,j],color='white',size=18,\
                     bbox=dict(facecolor='black',alpha=1.0))
    

def npy_examine(imgfile,img_bwfile,listfile,slide_init=0.94,\
                slide=[0.1,0.01,0.65,0.02],plotsize=PLOTSIZE):
    global npix, thisfile, subplots, img, img_bw, s
    img,thisfile = np.load(imgfile), np.load(listfile)
    img_bw = np.load(img_bwfile)
    npix = img.shape[0]/thisfile.shape[-2] # guarantees the first size
    fig = plt.figure(figsize=plotsize)
    fig.add_subplot(111,xticks=[],yticks=[])
    plt.subplots_adjust(left=subplots[0],right=subplots[1],\
                    bottom=subplots[2],top=subplots[3])
    plt.imshow(img,vmin=0.0,vmax=1.0)
    if thisfile.ndim==3:
        npy_addtext (thisfile[1])
        plt.text(0.00,-20.0,thisfile[0,0,0].split('/')[-1])
    else:
        plt.text(0.00,-20.0,thisfile[0,0].split('/')[-1])
    cid = fig.canvas.mpl_connect('key_press_event',onkeyclick)
#    axc='lightgoldenrodyellow'
#    axs = plt.axes([slide[0],slide[1],slide[2],slide[3]],axisbg=axc)
#    axs = plt.axes([slide[0],slide[1],slide[2],slide[3]],facecolor=axc)
#    s = Slider(axs, 'Brightness',0.0,2.,valinit=slide_init)
#    update(1.0)
#    s.on_changed(update)
    plt.show()

# Process the events file to take the LAST evaluation for each object (so that
# if a mistake has been made, the second click counts)
def proc_bigeye_events():
    a = np.loadtxt('bigeye_events',dtype='str')
    f = open('bigeye_events_new','w')
    for i in np.unique(a[:,0]):
        iwhere = (np.ravel(np.argwhere(a[:,0]==i)))[-1]
        f.write('%s %s\n'%(i,a[iwhere,1]))
    f.close()
    os.system('mv bigeye_events_new bigeye_events')

def bigeye_examine(npystart=0,npyend=1000000,prefix='./'):
    time1=time.time()
    for i in range(npystart,npyend):
        try:
            enew = np.ravel(np.load(prefix+'F%04d.npy'%i))
        except:    # leave if no file present
            break
#        try:
#            elist = np.append(elist,enew)
#        except:
#            elist = np.copy(enew)
        npy_examine(prefix+'%04d.npy'%i,prefix+'G%04d.npy'%i,prefix+'F%04d.npy'%i)
        print('Elapsed time %d sec'%(int(time.time()-time1)))
        proc_bigeye_events()


bigeye_examine()
