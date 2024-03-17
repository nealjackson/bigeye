# getimg - creates the npy files for examination by bigeye.py
# Usage: edit the following at the end of the preamble below:
#   directory maindir (a few lines below) with VIS, Y, J, H subdirectories
#   CPIX which gives the number of pixels in VIS image
#        (IR images re-interpolated to this if needed)
#   NX, NY just after this (number of images on a mosaic)
import os,sys,glob,numpy as np,scipy
import matplotlib; from matplotlib import pyplot as plt
import astropy; from astropy.io import fits
from astropy.io.fits import getdata, getheader
from astropy.stats import sigma_clipped_stats
from scipy import ndimage
from scipy.optimize import fmin
from astropy import visualization
import reproject; from reproject import reproject_interp
matplotlib.use('Qt5Agg')
plt.rcParams['image.origin']='lower'
plt.rcParams['image.interpolation']='nearest'
#
# EDIT THESE THINGS HERE
maindir = '../../Qt-stamp-visualizer/team_1/'
NX, NY = 6,6
CPIX = 99
#  probably leave the rest alone
dirVIS, dirY, dirJ, dirH = maindir+'VIS/',maindir+'Y/',maindir+'J/',maindir+'H/'

def pixi (pi,burn=10):
    vx,vy = np.asarray(np.asarray(pi.shape)/2,dtype='int')
    b=pi[vy-5:vy+5,vx-5:vx+5]
    b=np.sort(np.ravel(b[~np.isnan(b)]))
    vmax = np.sqrt(b[int(len(b))-burn])
    vout = np.sqrt(pi)
    np.putmask(vout,np.isnan(vout),0.0)
    np.putmask(vout,vout>vmax,vmax)
    return vout / vmax

def imgproc(a,interp=None,filt=None,shift=None,\
            normsum=False,cut=None):
    if filt:
        a = ndimage.gaussian_filter(a,filt)
    if normsum:
        a0,amed,astd = getmedstd(a)
        a -= amed
        a /= np.nansum(a)
        astd /= np.nansum(a)
    if cut:
        a = a[cut[0]:cut[1],cut[0]:cut[1]]
    return a

def getmedstd(a):
    return sigma_clipped_stats(a,sigma=3.0)

def eprint(array,n=4):
    array = np.asarray(array,dtype='int')
    form = '%%%dd '%n
    for iy in range(array.shape[0]-1,-1,-1):
        for ix in range(array.shape[0]):
            sys.stdout.write(form % (array[iy,ix]))
        sys.stdout.write('\n')

#
# This is the basic subroutine which takes four filenames (Vis+3*IR) and
# produces a numpy array in RGBI format (with G empty)
#
def mk3im(vfile,yfile,hfile,jfile,vsmooth,cut,angle):
    sys.stdout.write(vfile.split('/')[-1]+'\n');sys.stdout.flush()
    vhdu, yhdu, hhdu, jhdu = fits.open(vfile)[0], fits.open(yfile)[0], \
        fits.open(hfile)[0], fits.open(jfile)[0]
    y,footprint = reproject_interp (yhdu,vhdu.header)
    h,footprint = reproject_interp (hhdu,vhdu.header)
    j,footprint = reproject_interp (jhdu,vhdu.header)
    v = imgproc(vhdu.data,filt=vsmooth,shift=None,normsum=True,cut=cut)
    y = imgproc(y,interp=None,normsum=True,cut=cut)
    h = imgproc(h,interp=None,normsum=True,cut=cut)
    j = imgproc(j,interp=None,normsum=True,cut=cut)
    ir = (y+h+j)/3.0
    vsum = vhdu.data.sum()
    v0,vmed,vstd = getmedstd(v);v-=vmed
    ir0,irmed,irstd = getmedstd(ir);ir-=irmed
    bluebits = np.array( (v>3.*vstd)&(((ir>3*irstd)&(v>1.05*ir))\
           | (v>3.*irstd)&(ir<3.*irstd)) ,dtype='float')
    redbits = np.array( (v>3.*vstd)&(((ir>3*irstd)&(ir>1.05*v))\
           | (v<3.*irstd)&(ir>3.*irstd)) ,dtype='float')
    vpixi = pixi(v,burn=5)
    theta = np.deg2rad(angle)
    blueplane = bluebits*(1.0-vpixi)*np.sin(theta)
    redplane = redbits*(1.0-vpixi)*np.sin(theta)
    np.putmask(vpixi,bluebits!=0.0,1.0-(1.0-vpixi)*np.cos(theta))
    np.putmask(vpixi,redbits!=0.0,1.0-(1.0-vpixi)*np.cos(theta))
    vall=np.rollaxis(np.stack([redplane,np.zeros_like(vpixi),\
                               blueplane,vpixi]),0,3)
    return vall

def mk3im_bw(vfile):
    vhdu = fits.open(vfile)[0]; a=vhdu.data
    vx,vy = np.asarray(np.asarray(a.shape)/2,dtype='int')
    b=a[vy-5:vy+5,vx-5:vx+5]
    b=np.sort(np.ravel(b[~np.isnan(b)]))
    vmax = np.sqrt(b[int(len(b))-10])
    vout = np.sqrt(a)
    np.putmask(vout,np.isnan(vout),0.0)
    np.putmask(vout,vout>vmax,vmax)
    return 1 - vout / vmax

# Arguments to list2npy_new:
#   fl: n x 4 array of strings which are the VIS,Y,J,H names (including path)
#   prefix: Prefix (including subdirectory) for npy files which are produced.
#           Must be the same as in npy_examine in bigeye.
#   vsmooth: Smoothing applied to VIS band for display (otherwise v. pixellated)
#   angle: Colour table rotation for red and blue parts. 0 means greyscale,
#         90 means fully coloured. 20-25 works best.
def list2npy_new(fl,nx,ny,prefix,cpix=50,cut=None,vsmooth=0.5,\
                 angle=25.0,nstart=None,nend=None):
    imn=0; paneln=0
    while imn<len(fl):
        if paneln>10:
            break
        imfile = []
        img = np.zeros((ny*cpix,nx*cpix*2,4))
        for i in range(ny):
            for j in range(nx):
                px,py = j*2*cpix,ny*cpix-cpix*(i+1)
                img[py:py+cpix,px+cpix:px+2*cpix] = mk3im (fl[imn,0],fl[imn,1],fl[imn,2],fl[imn,3],\
                            vsmooth,cut,angle)
                img[py:py+cpix,px:px+cpix,3] = mk3im_bw (fl[imn,0])
                imfile.append(fl[imn,0])
                imn+=1
                if imn==len(fl):
                    for k in range(nx*ny - len(imfile)%(nx*ny)):
                        imfile.append('DUMMY')
                    break
            if imn==len(fl):
                break
        print('saving',imfile[0],'...',imfile[-1],'to',prefix+'%04d'%paneln)
        print(imfile,len(imfile))
        limfile = np.asarray(imfile).reshape(ny,nx)
        np.save (prefix+'F%04d'%paneln,limfile)
        np.save (prefix+'%04d'%paneln,img)
        paneln+=1

# You should have been able to edit maindir above to get dirVIS etc here
# Otherwise you have to provide a routine which returns an nx4 array of
#    filenames, with paths.
def euclid_get_fl():
    os.system('ls -1 %s >col1'%dirVIS)
    os.system('ls -1 %s >col2'%dirY)
    os.system('ls -1 %s >col3'%dirJ)
    os.system('ls -1 %s >col4'%dirH)
    c1 = np.loadtxt('col1',dtype='str')
    c2 = np.loadtxt('col2',dtype='str')
    c3 = np.loadtxt('col3',dtype='str')
    c4 = np.loadtxt('col4',dtype='str')
    os.system('rm col1;rm col2;rm col3;rm col4')
    c1 = np.char.add(len(c1)*[dirVIS],c1)
    c2 = np.char.add(len(c2)*[dirY],c2)
    c3 = np.char.add(len(c3)*[dirJ],c3)
    c4 = np.char.add(len(c4)*[dirH],c4)
    return np.array([c1,c2,c3,c4]).T


# Generate the npy files and put them in the current directory, 32 plots per page
fl = euclid_get_fl()
list2npy_new (fl, NX,NY, './', cpix=CPIX)
