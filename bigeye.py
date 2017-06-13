#  bigeye.py: program for eyeball inspection of images
#  Neal Jackson, 01.02.17
#  
#  Program has 2 modes:
#     testspace() for one-colour images
#     testground() for two-colour images
#  for each picture you think is a lens, hit 1,2,3 or 4 depending on confidence
#  hit the x at the top right to go to the next screen
#  the program prints out the ROC area at the end. 0.9-0.95 is doing well.
#
import numpy as np,scipy as sp, matplotlib,glob,os,sys,img_scale,time
#import pyfits;from pyfits import getdata
from scipy import ndimage
import astropy; from astropy.io.fits import getdata
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
global img,s
#  Change the following things if needed:
PLOTSIZE = (32.5,25)    # size of plot on screen (32.5x25 is a big screen)
SSCREEN = 0         # screenful to start with
NSCREEN = 100             # number of screenfuls (each takes about a minute)
NX,NY = 8,4            # images per screen
CHEAT = False            # tells you which are the lenses if True
LIST0 = 100000          # ID no of first image
subplots=[0.01,1.0,0.05,0.99]
plt.rcParams['image.origin']='lower'
plt.rcParams['image.interpolation']='nearest'
ROOT = '/home/njj/projects/euclid/challenge/'
SPUBLIC = ROOT+'SpaceBasedTraining/data/public/'
SPRIVATE = ROOT+'SpaceBasedTraining/data/private/'
SCLASSIFICATIONS = ROOT+'SpaceBasedTraining/classifications.csv'
RGB = False
GPUBLIC = ROOT+'results/ground_fpos_data_nj/'
GPRIVATE = ROOT+'GroundBasedTraining/data/private/'
GCLASSIFICATIONS = ROOT+'GroundBasedTraining/classifications.csv'
CSCALE = matplotlib.cm.gray_r
# ==============================================================

def onkeyclick (event):
    global thisfile, npix
    if event.xdata==None:
        print 'Did not read button, try moving the mouse first'
        return
    if event.key=='5':
        plt.close('all')
    else:
        try:
            xidx,yidx = int(event.xdata/npix),int(event.ydata/npix)
            idfile = thisfile[thisfile.shape[0]-yidx-1,xidx]
            f=open('bigeye_events','a')
            f.write('%s %s\n'%(idfile,event.key))
            f.close()    
            sys.stdout.write('%s %s\n'%(idfile,event.key))
            plt.plot([xidx*npix+10],[yidx*npix+10],'yo')
            plt.draw()
        except:
            print 'No image at position requested'

def update(val):
    global img,s,subplots
    sys.stdout.write ('Adjusting colour scale, please wait...')
    plt.subplot(111,xticks=[],yticks=[])
    plt.subplots_adjust(left=subplots[0],right=subplots[1],\
                bottom=subplots[2],top=subplots[3])
    if img.ndim == 2:
        plt.imshow(img,vmin=0.,vmax=s.val,cmap=CSCALE)
    else:
        img1 = np.asarray(img,dtype='float')/s.val
        np.putmask(img1,img1>1.0,1.0)
        plt.imshow(img1,vmin=0.,vmax=s.val,cmap=CSCALE)
    plt.draw()

def getimg (inimg,noise,pc,cut,smooth):
    if smooth:
        inimg = ndimage.gaussian_filter(inimg,smooth)
    if cut!=[]:
        inimg = inimg[cut[0]:cut[1],cut[0]:cut[1]]
    inimg[:,0]=inimg[:,-1]=inimg[0,:]=inimg[-1,:] = inimg.max()
    isiz = inimg.shape[0]
    b = np.sort(np.ravel(inimg[isiz/3:2*isiz/3,isiz/3:2*isiz/3]))
    black = b[min(pc*len(b)/100,len(b)-1)]
    black = max(black,noise)
    inimg = img_scale.sqrt(inimg,scale_min=0,scale_max=black)
    if RGB:
        inimg /= inimg.max()
    return inimg

def list2npy (rlist,glist='',blist='',nx=NX,ny=NY,cut=[],smooth=1.0,noise=0.001,pc=95,prefix='bgz_',llist='',rgbcut=1.0):
    print 'got',nx,ny
    npix = getdata(rlist[0]).shape[0] if cut==[] else cut[1]-cut[0]
    rgb = (len(glist)>0 or len(blist)>0)
    if rgb:
        glist = blist if not len(glist) else glist
        blist = glist if not len(blist) else blist
    imn = 0
    paneln = 0
    while True:
        imfile = []
        img = np.zeros((ny*npix,nx*npix,3)) if rgb else np.zeros((ny*npix,nx*npix))
        for i in range(ny):
            for j in range(nx):
                px,py = j*npix, ny*npix-npix*(i+1)
                if rgb:
                    dr,dg,db = getdata(rlist[imn]),getdata(glist[imn]),getdata(blist[imn])
                    # deal with artefacts
                    np.putmask(dr,dr>rgbcut,2.*dg-db)
                    np.putmask(dg,dg>rgbcut,0.5*(dr+db))
                    np.putmask(db,db>rgbcut,2.*dg-dr)
                    img[py:py+npix,px:px+npix,0] = getimg(dr,noise,pc,cut,smooth)
                    img[py:py+npix,px:px+npix,1] = getimg(dg,noise,pc,cut,smooth)
                    img[py:py+npix,px:px+npix,2] = getimg(db,noise,pc,cut,smooth)
                else:
                    d = getdata(rlist[imn])
                    img[py:py+npix,px:px+npix] = getimg(d,noise,pc,cut,smooth)
                imfile.append(rlist[imn])
                if llist != '' and llist[imn]:
                    img[py:py+npix/20,px:px+npix/20] = img.max()
                imn+=1
                if imn>=len(rlist):
                    break
            if imn>=len(rlist):
                break
        if len(imfile)<nx*ny:
            iextra = nx*ny-len(imfile)
            for i in range(iextra):
                imfile.append('DUMMY')
        print 'saving',imfile[0],'...',imfile[-1],'to',prefix+'%04d'%paneln
        limfile = np.asarray(imfile).reshape(ny,nx)
        np.save (prefix+'F%04d'%paneln,limfile)
        np.save (prefix+'%04d'%paneln,img)
        paneln+=1
        if imn>=len(rlist):
            break
    return npix
        
def npy_examine(imgfile,listfile,slide_init=0.94,\
                slide=[0.1,0.01,0.65,0.02],plotsize=PLOTSIZE):
    global npix, thisfile, subplots, img, s
    img,thisfile = np.load(imgfile), np.load(listfile)
    npix = img.shape[0]/thisfile.shape[0]
    fig = plt.figure(figsize=plotsize)
    fig.add_subplot(111,xticks=[],yticks=[])
    plt.subplots_adjust(left=subplots[0],right=subplots[1],\
                    bottom=subplots[2],top=subplots[3])
    plt.imshow(img,vmin=0.0,vmax=1.0,cmap=CSCALE)
    plt.text(0.00,-20.0,thisfile[0,0].split('/')[-1])
    cid = fig.canvas.mpl_connect('key_press_event',onkeyclick)
    axc='lightgoldenrodyellow'
    axs = plt.axes([slide[0],slide[1],slide[2],slide[3]],axisbg=axc)
    s = Slider(axs, 'Brightness',0.0,2.,valinit=slide_init)
    update(1.0)
    s.on_changed(update)
    plt.show()
    

def testspace():
    os.system('mkdir npy')#; os.system('rm bigeye_events')
    rlist = np.sort(glob.glob(SPUBLIC+'i*fits'))
    llist = np.loadtxt(SCLASSIFICATIONS, delimiter=',')[:,1]
    t1,t2 = max(0,SSCREEN*NX*NY), min(len(rlist),(SSCREEN+NSCREEN)*NX*NY)
    print 'Using objects',t1,'to',t2
    if CHEAT:
        list2npy (rlist[t1:t2], nx=NX, ny=NY, prefix='npy/space',\
                  smooth=0.7,noise=1.e-11,pc=99,llist=llist[t1:t2])
    else:
        list2npy (rlist[t1:t2], nx=NX, ny=NY, prefix='npy/space',\
                  smooth=0.7,noise=1.e-11,pc=99)
    for i in range(SSCREEN,SSCREEN+NSCREEN):
        print time.ctime()
        npy_examine('npy/space%04d.npy'%i,'npy/spaceF%04d.npy'%i)
    evaluate_roc (SCLASSIFICATIONS,min_event=t1+LIST0,max_event=t2+LIST0)
        
def testground():
    os.system('mkdir npy'); os.system('rm bigeye_events')
    rlist = np.sort(glob.glob(GPUBLIC+'*I*.fits'))
    glist = np.sort(glob.glob(GPUBLIC+'*R*.fits'))
    blist = np.sort(glob.glob(GPUBLIC+'*G*.fits'))
    t1,t2 = max(0,SSCREEN*NX*NY), min(len(rlist),(SSCREEN+NSCREEN)*NX*NY)
    llist = np.loadtxt(GCLASSIFICATIONS, delimiter=',')[:,1] # assumes numeric order
    print 'Using objects',t1,'to',t2
    if CHEAT:
        list2npy (rlist[t1:t2],glist[t1:t2],blist[t1:t2],nx=NX,ny=NY,\
                  noise=2e-11,prefix='npy/ground',llist=llist[t1:t2],pc=99)
    else:
        list2npy (rlist[t1:t2],glist[t1:t2],blist[t1:t2],nx=NX,ny=NY,\
                  noise=2e-11,prefix='npy/ground',pc=99)
    for i in range(SSCREEN,SSCREEN+NSCREEN):
        print time.ctime()
        npy_examine('npy/ground%04d.npy'%i,'npy/groundF%04d.npy'%i)
    evaluate_roc (GCLASSIFICATIONS,min_event=t1+LIST0,max_event=t2+LIST0)

def evaluate_roc(classifications, min_event=-1.0,max_event=1.0E9,pre='-',post='.fits'):
    llist = np.loadtxt(classifications, delimiter=',')
    events = np.loadtxt('bigeye_events',dtype='S')
    for l in events:
        enew = np.array([int(l[0].split(pre)[1].split(post)[0]),int(l[1])])
        try:
            e = np.vstack((e,enew))
        except:
            e = np.copy(enew)
        
    if max_event==1.0E9:
        max_event = e[:,0].max()
    islens = llist[:,1]
    llist = np.asarray(llist[:,0],dtype='int')
    islens = islens[(min_event<=llist)&(llist<=max_event)]
    islens = np.asarray(islens,dtype='bool')
    llist = llist[(min_event<=llist)&(llist<=max_event)]
    tpr,fpr = np.array([0.0]), np.array([0.0])
    print 'Thr        TP       FN       TN       FP      TPR      FPR'
    for t in np.sort(np.unique(e[:,1]))[::-1]:
        false_pos = false_neg = true_pos = true_neg = 0
        ethis = e[e[:,1]>=t]
        for i in llist[islens]:
            if i in ethis[:,0]:
                true_pos+=1
            else:
                false_neg+=1
        for i in llist[~islens]:
            if i in ethis[:,0]:
                false_pos+=1
            else:
                true_neg+=1
        thistpr = float(true_pos)/(true_pos+false_neg)
        thisfpr = float(false_pos)/(false_pos+true_neg)
        print '%4d %8d %8d %8d %8d %8f %8f' % \
             (t,true_pos,false_neg,true_neg,false_pos,thistpr,thisfpr)
        tpr = np.append(tpr, thistpr)
        fpr = np.append(fpr, thisfpr)
    tpr = np.append(tpr,1.0)
    fpr = np.append(fpr,1.0)
    print 'Area under ROC curve (0.5=random, 1=perfect):', np.trapz(tpr,fpr)
    plt.plot(fpr,tpr);plt.plot(fpr,tpr,'bo')
    plt.xlabel('False positive rate'); plt.ylabel ('True positive rate')
    plt.savefig('bigeye_roc.png')    
