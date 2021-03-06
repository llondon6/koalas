
'''

Example script for intpdiff

spxll'16

'''

#
from os import system
system('clear')

#
from positive import *
from matplotlib.pyplot import *
from numpy import *

#
t = linspace(0,12*pi,1e3)

#
y0 = cos(t)

#
y1 = intrp_diff( t, y0 )
y2 = intrp_diff( t, y0, n = 2 )

#
pks,locs = findpeaks(y0)

#
figure()

plot( t, y0, color='0.8', linewidth=5 )
plot( t, y1, 'r--' )
plot( t,-y2, '--b' )
plot( t[locs], pks, 'or' )

xlim( lim(t) )

show()
