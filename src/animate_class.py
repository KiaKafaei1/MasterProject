import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Ani():
    def __init__(self,omegas, nsteps, line):
        self.nsteps = nsteps
        self.omegas = omegas
        self.line=line
        self.step = 0
        self.i = 0
    
    def getdata(self,j):
        t = np.arange(0,j)/float(self.nsteps)*2*np.pi
        x = np.sin(self.omegas[self.step]*t)
        return t,x
        
    def gen(self):
        for i in range(len(self.omegas)):
            tit = u"animated sin(${:g}\cdot t$)".format(self.omegas[self.step])
            self.line.axes.set_title(tit)
            for j in range(self.nsteps):
                yield j
            self.step += 1
            
    def animate(self,j):
        x,y = self.getdata(j)
        self.line.set_data(x,y)
