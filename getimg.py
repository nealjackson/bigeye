# getimg - creates the npy files for examination by bigeye.py
# Usage: edit the routine euclid_get_fl at the bottom and run
import os,sys,glob,numpy as np,img_scale,scipy
import matplotlib; from matplotlib import pyplot as plt
import astropy; from astropy.io import fits
from astropy.io.fits import getdata, getheader
from astropy.stats import sigma_clipped_stats
#import photutils; from photutils import segmentation  # has been broken
#from photutils.segmentation import SegmentationImage
from scipy import ndimage
from scipy.optimize import fmin
from astropy import visualization
import reproject; from reproject import reproject_interp
matplotlib.use('Qt5Agg')
plt.rcParams['image.origin']='lower'
plt.rcParams['image.interpolation']='nearest'

def pixi (pi,pimin=5.8e-13,pc=99.9):
    pinan = pi[~np.isnan(pi)]
    pisort = np.sort(np.ravel(pinan))
    pimax = pisort[int(len(pisort)*pc/100.)]
    pi = np.clip(pi,pimin,pimax)
    return np.sqrt((pi-pimin)/(pimax-pimin))

def imgproc(a,interp=None,filt=None,shift=None,\
            normsum=False,colscale=None,cut=None,pc=99.9):
    if filt:
        a = ndimage.gaussian_filter(a,filt)
    if normsum:
        a0,amed,astd = getmedstd(a)
        a -= amed
        a /= np.nansum(a)
        astd /= np.nansum(a)
    if colscale:
        a0,amed,astd = getmedstd(a)
        colscale = astd if np.isnan(colscale) else colscale
        a = pixi(a,colscale,pc=pc)
    if cut:
        a = a[cut[0]:cut[1],cut[0]:cut[1]]
    return a

#  do a histogram with successively more bins until the maximum occurs in bin BINNO
#  (this assumes the background is more uniform than the foreground) and use the
#  pixels from 0 to 2*BINNO to do statistics.
def getmedstd(a):
    BINNO = 6
    found_bkg = False
    np.save('save',a)
    ar = np.sort(np.ravel(a)); ar=ar[~np.isnan(ar)]
    while not found_bkg:
        for i in range(1,len(ar)):
            ah = np.histogram(ar,bins=i)[0]
            if np.argwhere(ah==ah.max())[0][0]>=BINNO:
                npix_back = np.sum(ah[:2*BINNO])
                found_bkg = True
                break
        if (found_bkg):
            break
        ar = ar [:int(len(ar)/2)]
    return np.mean(ar[:npix_back]),np.median(ar[:npix_back]),np.std(ar[:npix_back])


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
def mk3im(vfile,yfile,hfile,jfile,pc,pcclip,pcslope,vsmooth,cut,angle):
    sys.stdout.write(vfile.split('/')[-1]+'\n');sys.stdout.flush()
    vhdu, yhdu, hhdu, jhdu = fits.open(vfile)[0], fits.open(yfile)[0], fits.open(hfile)[0],\
                             fits.open(jfile)[0]
    y,footprint = reproject_interp (yhdu,vhdu.header)
    h,footprint = reproject_interp (hhdu,vhdu.header)
    j,footprint = reproject_interp (jhdu,vhdu.header)
    v = imgproc(vhdu.data,filt=vsmooth,shift=None,normsum=True,cut=cut)
    y = imgproc(y,interp=None,normsum=True,cut=cut)
    h = imgproc(h,interp=None,normsum=True,cut=cut)
    j = imgproc(j,interp=None,normsum=True,cut=cut)
    ir = (y+h+j)/3.0
    vsum = vhdu.data.sum()
    thispc = pc
    if vsum >= pcclip:
        thispc = pcslope*(np.log10(vsum)-np.log10(pcclip))
    v0,vmed,vstd = getmedstd(v);v-=vmed
    ir0,irmed,irstd = getmedstd(ir);ir-=irmed
    bluebits = np.array( (v>3.*vstd)&(((ir>3*irstd)&(v>1.05*ir))\
           | (v>3.*irstd)&(ir<3.*irstd)) ,dtype='float')
    redbits = np.array( (v>3.*vstd)&(((ir>3*irstd)&(ir>1.05*v))\
           | (v<3.*irstd)&(ir>3.*irstd)) ,dtype='float')
    vpixi = pixi(v,vstd,pc=thispc)
    theta = np.deg2rad(angle)
    blueplane = bluebits*(1.0-vpixi)*np.sin(theta)
    redplane = redbits*(1.0-vpixi)*np.sin(theta)
    np.putmask(vpixi,bluebits!=0.0,1.0-(1.0-vpixi)*np.cos(theta))
    np.putmask(vpixi,redbits!=0.0,1.0-(1.0-vpixi)*np.cos(theta))
    vall=np.rollaxis(np.stack([redplane,np.zeros_like(vpixi),blueplane,vpixi]),0,3)
    return vall

# Arguments to list2npy_new:
#   fl: n x 4 array of strings which are the VIS,Y,J,H names (including path)
#   prefix: Prefix (including subdirectory) for npy files which are produced.
#           Must be the same as in npy_examine in bigeye.
#   vsmooth: Smoothing applied to VIS band for display (otherwise v. pixellated)
#   angle: Colour table rotation for red and blue parts. 0 means greyscale,
#         90 means fully coloured. 20-25 works best.
#   pc: Cutoff for maximum (black-level) in the sqrt scaling. 99.9 means use
#         pixels at the 99.9th percentile.
#   pcclip, pcslope: Tapers the black level so that very bright sources (with sum
#         of counts in V >pcclip) are more burned out (to see faint detail).
#         Taper is fiercer if pcslope is higher. No effect if pcslope=0.0.
def list2npy_new(fl,nx,ny,prefix,cpix=50,cut=None,vsmooth=0.5,pcclip=5.0e6,pcslope=1.3,\
                 angle=25.0,pc=99.9,nstart=None,nend=None):
    imn=0; paneln=0
    while imn<len(fl):
        imfile = []
        img = np.zeros((ny*cpix,nx*cpix,4))
        for i in range(ny):
            for j in range(nx):
                px,py = j*cpix,ny*cpix-cpix*(i+1)
                img[py:py+cpix,px:px+cpix] = mk3im (fl[imn,0],fl[imn,1],fl[imn,2],fl[imn,3],\
                            pc,pcclip,pcslope,vsmooth,cut,angle)
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

# Installation-dependent routine to get the filelist array
# If you have four directories, one for each band, all with the same number of files, and for which
# 'ls -1' gets the files in each case in the right order, then all you need to do is to change the
# first line of this routine to get the correct directories (including the trailing slashes).
# Otherwise you have to provide a routine which returns an nx4 array of filenames, with paths.
def euclid_get_fl():
    dirVIS, dirY, dirJ, dirH = '../ERO/LSB/VIS/','../ERO/LSB/Y/','../ERO/LSB/J/','../ERO/LSB/H/'
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
list2npy_new (fl, 8, 4, './')
