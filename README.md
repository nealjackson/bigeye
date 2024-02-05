# bigeye
Software for automatic examination of many images

Two Python scripts:

getimg.py - makes a set of npy files containing the mosaics of colour stamps from VIS/J/H/Y images. You need to modify the subroutine at
            the end to tell it the directory where the files are. Also makes a set of npy files beginning F... which index the mosaics.
            If you don't like the colour scale, see the comments for the parameters to change it.
            
bigeye.py - allows you to inspect the mosaic stamps. Hit 1, 2, 3 or 4 in each object you consider to be possible, probable, likely or
            certain, and hit 5 to continue to the next mosaic. If you make a mistake, just go back and re-click and your previous
            judgement is overridden. Writes out a file called bigeye_events with the VIS fits file and the key-click (nb this is always
            open for append, so you need to manually delete it if starting again).

Reference: see the accompanying pdf file.
