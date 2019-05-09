import os
import pickle
import numpy as np

from API import *


from a2c import *
from randomSolver import *
from pointerNetwork import *
from programGraph import *
from SMC import *
from ForwardSample import *
from MCTS import MCTS
from beamSearch import *
from CNN import *

import traceback
import sys
import os
import time
import random


RESOLUTION = 32

import torch
import torch.nn as nn
import torch.nn.functional as F

class BadCSG(Exception): pass

def discrete(c,cs):
    "return the coordinate in cs which is closest to the true coordinate c"
    return min(cs,key=lambda z: abs(z - c))

def voxels2dm(vs):
    r = vs.shape[0]
    span = np.arange(0.,1.,1./r)

    x, y, z = np.indices((r,r,r))/float(r)

    maps = []

    # collapsing Z (front)
    ds = np.zeros((r,r,r))
    ds[:,:,:] = span

    maps.append(np.amax((vs*z), axis=2))

    # collapsing Z (behind)
    ds[:,:,:] = np.flip(span)
    maps.append(np.amax((vs*np.flip(z,2)), axis=2))

    return [m
            for n,i in enumerate(np.indices((r,r,r))/float(r))
            for m in [np.amax(vs * i, axis=n),
                      np.amax(vs * np.flip(i,n), axis=n)] ]

class CSG(Program):
    def __init__(self):
        self._rendering = None

    def clearRendering(self):
        self._rendering = None
        for c in self.children(): c.clearRendering()

    def simplifications(self):
        """removes a single thing from the tree"""
        if True:
            return 
        yield 

    def heatMapTarget(self):
        hm = np.zeros((RESOLUTION,RESOLUTION,3)) > 1.
        for c in self.children():
            hm = np.logical_or(hm,c.heatMapTarget())
        return hm

    def scad(self,fn=None):
        cylindrical = """
function vectorLength(v1,v2) = sqrt(
    (v2[0]-v1[0])*(v2[0]-v1[0])+
    (v2[1]-v1[1])*(v2[1]-v1[1])+
    (v2[2]-v1[2])*(v2[2]-v1[2]));

function lookAt(v1, v2) =
    let(v = v2-v1)
    [
       0,
       acos(v[2]/vectorLength(v1,v2)),
       atan2(v[1], v[0])
    ];

module cylindrical(p1,p2,radius)
{
    translate(p1)
    rotate(lookAt(p1,p2))
    cylinder(vectorLength(p1,p2),radius,radius,$fn=60);
}
"""
        source = f"translate([{-RESOLUTION//2},{-RESOLUTION//2},{-RESOLUTION//2}])"
        source = "%s {\n%s\n}\n"%(source,self._scad())
        source = f"{cylindrical}\n{source}"
        if fn is None: return source
        if fn.endswith("scad"):
            with open(fn,"w") as handle:
                handle.write(source)
        elif fn.endswith("png"):
            with open(fn + ".scad","w") as handle:
                handle.write(source)
            os.system(f"openscad --camera=0,0,0,60,0,30,140 --autocenter -o {fn} {fn}.scad")
            #os.system(f"rm {fn}.scad")
            
        return source

    def __repr__(self):
        return str(self)

    def simplify(self): return self

    def __eq__(self,o):
        return isinstance(o,CSG) and self.serialize() == o.serialize()

    def __ne__(self, o): return not (self == o)

    def __add__(self,o): return Union(self,o)

    def __sub__(self,o): return Difference(self,o)

    def __hash__(self): return hash(self.serialize())

    @property
    def dimensionality(self):
        cs = self.children()
        if len(cs) == 0: assert False
        assert all( c.dimensionality == cs[0].dimensionality for c in cs  )
        return cs[0].dimensionality
        

    def execute(self):
        if self.dimensionality == 2:
            return self.render(64)
        elif self.dimensionality == 3:
            return self.render(32)
        assert False
        

    def render(self,r=None):
        resolution = r or RESOLUTION
        if self._rendering is None or self._rendering[0] != resolution:
            self._rendering = (resolution,self._render(resolution))
        return self._rendering[1]

    def IoU(self, other):
        if isinstance(other, CSG): other = other.execute()
        return (self.execute()*other).sum()/(self.execute() + other - self.execute()*other).sum()
    
    def removeDeadCode(self): return self

    def depthMaps(self, r=None):
        """Returns six depth maps"""
        r = r or RESOLUTION
        return voxels2dm(self.render(r=r))
    
    def show(self, resolution=None, export=None):
        """Open up a new window and show the CSG"""
        import matplotlib.pyplot as plot
        from mpl_toolkits.mplot3d import Axes3D
        columns = 2
        f = plot.figure()

        maps = self.depthMaps(resolution)
        

        a = f.add_subplot(1 + int(export is None), 1 + len(maps) if export is None else 1, 1,
                          projection='3d')        
            
        a.voxels(self.render(RESOLUTION), edgecolor='k')

        if export is not None:
            plot.savefig(f"{export}_v.png")
            f = plot.figure()

        for i,m in enumerate(maps):
            a = f.add_subplot(1,
                              len(maps) + (int(export is None)), 1 + i + int(export is None))
            a.imshow(1 - m, cmap='Greys',
                     vmin=0., vmax=1.)
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
        
        if export is None:
            plot.show()
        else:
            return 
            #plot.savefig(f"{export}_d.png")

    def export(self,fn,resolution):
        pixels = self.render(resolution)
        assert len(pixels.shape) == 2
        saveMatrixAsImage(np.flip(pixels.T,0),fn)
        

    def removeCodeNotChangingProjections(self):
        while True:
            current = np.array(voxels2dm(self.render()))
            bad = False
            for child in self.simplifications():
                new = np.array(voxels2dm(child.render()))
                if np.sum(np.abs(new - current)) <= 0.0001:
                    bad = True
                    self = child
                    break
            if not bad: return self
                
    def removeCodeNotChangingRender(self):
        while True:
            current = self.render()
            bad = False
            for child in self.simplifications():
                new = child.render()
                if np.sum(np.abs(new - current)) <= 0.0001:
                    bad = True
                    self = child
                    break
            if not bad: return self
                
            
        
        
        
                
# The type of CSG's
tCSG = BaseType(CSG)

class Cylinder(CSG):
    token = 'cylinder'
    type = arrow(integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 tCSG)

    def __init__(self, r,
                 x0, y0, z0,
                 x1, y1, z1):
        super(Cylinder, self).__init__()
        self.r = r
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.p0 = np.array([x0,y0,z0])
        self.p1 = np.array([x1,y1,z1])

    @property
    def dimensionality(self): return 3

    def _scad(self):
        return f"cylindrical([{self.x0}, {self.y0}, {self.z0}], [{self.x1}, {self.y1}, {self.z1}], {self.r});"

    def __str__(self):
        return f"(cylinder r={self.r} p0={self.p0} p1={self.p1})"

    def rotate(self,rx,ry,rz):
        R = rotationMatrix(rx,ry,rz)
        p0 = R@self.p0
        p1 = R@self.p1
        return Cylinder(*([self.r] + list(p0) + list(p1)))        
        
    def children(self): return []
    
    def translate(self, x,y,z):
        return Cylinder(self.r,
                        self.x0 + x,self.y0 + y,self.z0 + z,
                        self.x1 + x,self.y1 + y,self.z1 + z)

    def extent(self):
        return np.stack([self.p0,self.p1]).min(0),np.stack([self.p0,self.p1]).max(0)
    def scale(self,s):
        return Cylinder(self.r*s,
                        self.x0*s,self.y0*s,self.z0*s,
                        self.x1*s,self.y1*s,self.z1*s)
    def discrete(self, coordinates):
        return Cylinder(discrete(self.r,coordinates),
                        discrete(self.x0,coordinates),
                        discrete(self.y0,coordinates),
                        discrete(self.z0,coordinates),
                        discrete(self.x1,coordinates),
                        discrete(self.y1,coordinates),
                        discrete(self.z1,coordinates))

    def toTrace(self): return [self]

    def serialize(self):
        return (self.__class__.token,self.r,
                self.x0,self.y0,self.z0,
                self.x1,self.y1,self.z1)

    def _render(self,r=None):
        r = r or RESOLUTION
        X,Y,Z = np.indices((r,r,r))/r
        p0 = self.p0/RESOLUTION
        p1 = self.p1/RESOLUTION
        v = p1 - p0
        radius = self.r/RESOLUTION

        # (q - p0).(p1 - p0) >= 0
        # (q - p0).v >= 0
        # q.v - p0.v >= 0
        # q.v >= p0.v
        qv = X*v[0] + Y*v[1] + Z*v[2]
        firstCheck = qv >= np.dot(p0,v)
        # (q - p1)*v <= 0
        # q.v - p1.v <= 0
        # q.v <= p1.v
        secondCheck = qv <= np.dot(p1,v)

        # qxv - p0xv
        p0_x_v = np.cross(p0,v)
        crossX = Y*v[2] - Z*v[1] - p0_x_v[0]
        crossY = Z*v[0] - X*v[2] - p0_x_v[1]
        crossZ = X*v[1] - Y*v[0] - p0_x_v[2]
        normCross = crossX*crossX + crossY*crossY + crossZ*crossZ
        thirdCheck = normCross < radius*radius*np.dot(v,v)
        return 1.*(firstCheck&secondCheck&thirdCheck)

    def __contains__(self, q):
        if np.dot(q - self.p0, self.p1 - self.p0) >= 0 and np.dot(q - self.p1,self.p1 - self.p0) <= 0:
            v1 = np.cross(q - self.p0, self.p1 - self.p0)
            v2 = self.p1 - self.p0
            return np.linalg.norm(v1)/np.linalg.norm(v2) < self.r
        return False

    def flipX(self):
        return Cylinder(self.r,
                        RESOLUTION - self.x0,
                        self.y0,self.z0,
                        RESOLUTION - self.x1,
                        self.y1,self.z1)
    def flipY(self):
        return Cylinder(self.r,
                        self.x0,
                        RESOLUTION - self.y0,self.z0,
                        self.x1,
                        RESOLUTION - self.y1,self.z1)
    def flipZ(self):
        return Cylinder(self.r,
                        self.x0,
                        self.y0,RESOLUTION - self.z0,
                        self.x1,
                        self.y1,RESOLUTION - self.z1)

class Cuboid(CSG):
    token = 'cuboid'
    type = arrow(integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 tCSG)
    
    def __init__(self, x0, y0, z0, x1, y1, z1):
        super(Cuboid, self).__init__()
        if x1 < x0: raise ParseFailure()
        if y1 < y0: raise ParseFailure()
        if z1 < z0: raise ParseFailure()
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1

    @property
    def dimensionality(self): return 3        

    def removeDeadCode(self):
        if self.x1 <= self.x0: return None
        if self.y1 <= self.y0: return None
        if self.z1 <= self.z0: return None
        return self

    def translate(self,x,y,z):
        return Cuboid(
            self.x0 + x,self.y0 + y,self.z0 + z,
            self.x1 + x,self.y1 + y,self.z1 + z)

    def rotate(self,rx,ry,rz):
        print("WARNING: ignoring rotation of cuboid by {(rx,ry,rz)}")
        return self

    def _scad(self):
        s = f"translate([{self.x0}, {self.y0}, {self.z0}])"
        s += f" cube(size=[{self.x1 - self.x0}, {self.y1 - self.y0}, {self.z1 - self.z0}]);"
        return s
        

    def extent(self):
        return np.array([[self.x0,self.y0,self.z0],[self.x1,self.y1,self.z1]]).min(0), \
            np.array([[self.x0,self.y0,self.z0],[self.x1,self.y1,self.z1]]).max(0)
    def scale(self,s):
        return Cuboid(self.x0*s,self.y0*s,self.z0*s,self.x1*s,self.y1*s,self.z1*s)
    def discrete(self, coordinates):
        return Cuboid(discrete(self.x0,coordinates),
                      discrete(self.y0,coordinates),
                      discrete(self.z0,coordinates),
                      discrete(self.x1,coordinates),
                      discrete(self.y1,coordinates),
                      discrete(self.z1,coordinates))
        

    def toTrace(self): return [self]

    def _render(self,r=None):
        r = r or RESOLUTION

        a = np.zeros((r,r,r))
        a[int(self.x0*r/RESOLUTION):int(self.x1*r/RESOLUTION),
          int(self.y0*r/RESOLUTION):int(self.y1*r/RESOLUTION),
          int(self.z0*r/RESOLUTION):int(self.z1*r/RESOLUTION)] = 1

        return a
        

    def __str__(self):
        return f"(cuboid {self.x0} {self.y0} {self.z0} ; {self.x1} {self.y1} {self.z1})"

    def children(self): return []

    def serialize(self):
        return (self.__class__.token, self.x0, self.y0, self.z0, self.x1, self.y1, self.z1)

    def __contains__(self, p):
        return p[0] >= self.x0 and p[1] >= self.y0 and p[2] >= self.z0 and \
            p[0] < self.x1 and p[1] < self.y1 and p[2] < self.z1

    def flipX(self):
        return Cuboid(RESOLUTION - self.x1, self.y0, self.z0,
                      RESOLUTION - self.x0, self.y1, self.z1)

    def flipY(self):
        return Cuboid(self.x0, RESOLUTION - self.y1, self.z0,
                      self.x1, RESOLUTION - self.y0, self.z1)

    def flipZ(self):
        return Cuboid(self.x0, self.y1, RESOLUTION - self.z0,
                      self.x1, self.y0, RESOLUTION - self.z1)

class Sphere(CSG):
    token = 'sphere'
    type = arrow(integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1),
                 tCSG)
    
    def __init__(self, x,y,z,r):
        super(Sphere, self).__init__()
        self.r = r
        self.x = x
        self.y = y
        self.z = z
    @property
    def dimensionality(self): return 3
    def toTrace(self): return [self]

    def _scad(self):
        return f"translate([{self.x}, {self.y}, {self.z}])" + f" sphere(r={self.r + 1}, $fn = 60);"

    def translate(self,x,y,z):
        return Sphere(self.x + x,self.y + y,self.z + z,self.r)

    def rotate(self,rx,ry,rz):
        return self

    def _render(self,r=None):
        r = r or RESOLUTION
        x, y, z = np.indices((r,r,r))/float(r)

        dx = (x - self.x/RESOLUTION)
        dy = (y - self.y/RESOLUTION)
        dz = (z - self.z/RESOLUTION)
        distance2 = dx*dx + dy*dy + dz*dz

        return 1.*(distance2 <= (self.r/RESOLUTION)*(self.r/RESOLUTION))

    def extent(self):
        return np.array([self.x,self.y,self.z]) - self.r,\
            np.array([self.x,self.y,self.z]) + self.r
    def scale(self,s):
        return Sphere(self.x*s,self.y*s,self.z*s,self.r*s)
    def discrete(self,coordinates):
        return Sphere(discrete(self.r,coordinates),
                      discrete(self.x,coordinates),
                      discrete(self.y,coordinates),
                      discrete(self.z,coordinates))
        
    def __str__(self):
        return f"(sphere ({self.x}, {self.y}, {self.z}) {self.r})"

    def children(self): return []

    def serialize(self):
        return (self.__class__.token, self.x,self.y,self.z,self.r)

    def __contains__(self, p):
        return (p[0]-self.x)*(p[0]-self.x) + (p[1] - self.y)*(p[1] - self.y) + (p[2] - self.z)*(p[2] - self.z)\
            <= self.r*self.r

    def flipX(self):
        return Sphere(RESOLUTION - self.x, self.y, self.z, self.r)

    def flipY(self):
        return Sphere(self.x, RESOLUTION - self.y, self.z, self.r)

    def flipZ(self):
        return Sphere(self.x, self.y, RESOLUTION - self.z, self.r)

class Rectangle(CSG):
    token = 'tr'
    type = arrow(integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 tCSG)
    
    def __init__(self, x0, y0, x1, y1,
                 x2, y2, x3, y3):
        super(Rectangle, self).__init__()
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3

    def extent(self):
        xs = [self.x0,self.x1,self.x2,self.x3]
        ys = [self.y0,self.y1,self.y2,self.y3]
        return np.array([min(xs),min(ys)]),np.array([max(xs),max(ys)])

    @property
    def dimensionality(self): return 2        

    def toTrace(self): return [self]

    def __str__(self):
        return f"(tr ({self.x0}, {self.y0}) ({self.x1}, {self.y1}) ({self.x2}, {self.y2}) ({self.x3}, {self.y3}))"

    def children(self): return []

    def __eq__(self, o):
        return isinstance(o, Rectangle) and self.serialize() == o.serialize()

    def __hash__(self):
        return hash(self.serialize())

    def serialize(self):
        return (self.__class__.token,
                self.x0, self.y0, self.x1, self.y1,
                self.x2, self.y2, self.x3, self.y3)

    def _render(self,r=None):
        r = r or RESOLUTION
        x, y = np.indices((r,r))/float(r)
        p0 = np.array([self.x0,self.y0])/RESOLUTION
        p1 = np.array([self.x1,self.y1])/RESOLUTION
        p2 = np.array([self.x2,self.y2])/RESOLUTION
        p3 = np.array([self.x3,self.y3])/RESOLUTION

        def halfPlane(p,q):
            nonlocal x, y
            return (x - p[0])*(p[1] - q[1]) - (y - p[1])*(p[0] - q[0])
        return 1.*((halfPlane(p0,p1) <= 0.)&(halfPlane(p1,p2) <= 0)&(halfPlane(p2,p3) <= 0)&(halfPlane(p3,p0) <= 0))

    # def flipX(self):
    #     return Rectangle(RESOLUTION - self.x1, self.y0,
    #                       RESOLUTION - self.x0, self.y1)

    # def flipY(self):
    #     return Rectangle(self.x0, RESOLUTION - self.y1,
    #     			   self.x1, RESOLUTION - self.y0)

class Circle(CSG):
    token = 'tc'
    type = arrow(integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1),
                 tCSG)
    
    def __init__(self, x,y,d):
        super(Circle, self).__init__()
        self.d = d
        self.x = x
        self.y = y

    def toTrace(self): return [self]

    def extent(self):
        r = self.d//2 + 1
        return np.array([self.x - r,self.y - r]), np.array([self.x + r,self.y + r])

    @property
    def dimensionality(self): return 2

    def heatMapTarget(self):
        hm = np.zeros((RESOLUTION,RESOLUTION,3)) > 1.
        hm[self.x,self.y,2] = 1
        return hm
        
    def __str__(self):
        return f"(tc ({self.x}, {self.y}) d={self.d})"

    def children(self): return []

    def __eq__(self, o):
        return isinstance(o, Circle) and self.serialize() == o.serialize()
    def __hash__(self):
        return hash(self.serialize())

    def serialize(self):
        return (self.__class__.token, self.x,self.y,self.d)

    def __contains__(self, p):
        r = self.d/2
        return (p[0]-self.x)*(p[0]-self.x) + (p[1] - self.y)*(p[1] - self.y) <= r*r

    def flipX(self):
        return Circle(RESOLUTION - self.x, self.y, self.d)

    def flipY(self):
        return Circle(self.x, RESOLUTION - self.y, self.d)

    def _render(self,r=None):
        r = r or RESOLUTION
        x, y = np.indices((r,r))/float(r)

        dx = (x - self.x/RESOLUTION)
        dy = (y - self.y/RESOLUTION)
        distance2 = dx*dx + dy*dy

        r = self.d/2

        return 1.*(distance2 <= (r/RESOLUTION)*(r/RESOLUTION))

class Union(CSG):
    token = '+'
    type = arrow(tCSG, tCSG, tCSG)
    
    def __init__(self, a, b):
        super(Union, self).__init__()
        self.elements = [a,b]

    def translate(self,*d):
        return Union(self.elements[0].translate(*d),self.elements[1].translate(*d))

    def simplifications(self):
        for a in self.elements[0].simplifications():
            yield Union(a,self.elements[1])
        for b in self.elements[1].simplifications():
            yield Union(self.elements[0],b)
        yield self.elements[0]
        yield self.elements[1]

    def toTrace(self):
        return self.elements[0].toTrace() + self.elements[1].toTrace() + [self]

    def __str__(self):
        return f"(+ {str(self.elements[0])} {str(self.elements[1])})"

    def _scad(self):
        return "union() {\n" + self.elements[0]._scad() + "\n" + self.elements[1]._scad() + "\n}"

    def extent(self):
        a = self.elements[0]
        b = self.elements[1]
        a1,a2 = a.extent()
        b1,b2 = b.extent()
        return np.stack([a1,b1]).min(0), np.stack([a2,b2]).max(0)
    def scale(self,s):
        return Union(*[e.scale(s) for e in self.elements])
    def discrete(self,cs):
        return Union(*[e.discrete(cs) for e in self.elements])

    def children(self): return self.elements

    def serialize(self):
        return ('+',self.elements[0],self.elements[1])

    def __eq__(self, o):
        return isinstance(o, Union) and tuple(o.elements) == tuple(self.elements)

    def __hash__(self):
        return hash(('u', tuple(hash(e) for e in self.elements )))

    def __contains__(self, p):
        return any( p in e for e in self.elements )

    def removeDeadCode(self):
        a = self.elements[0].removeDeadCode()
        b = self.elements[1].removeDeadCode()

        if a is None: return b
        if b is None: return a

        self = Union(a,b)

        me = self.render()
        l = a.render()
        r = b.render()
        if np.all(me == l): return a
        if np.all(me == r): return b
        return self

    def flipX(self): return Union(self.elements[0].flipX(),
                                  self.elements[1].flipX())
    def flipY(self): return Union(self.elements[0].flipY(),
                                  self.elements[1].flipY())
    def flipZ(self): return Union(self.elements[0].flipZ(),
                                  self.elements[1].flipZ())

    def _render(self,r=None):
        r = r or RESOLUTION
        a = 1.*(self.elements[0].render(r))
        b = 1.*(self.elements[1].render(r))
        return a + b - a*b        

class Difference(CSG):
    token = '-'
    type = arrow(tCSG, tCSG, tCSG)
    
    def __init__(self, a, b):
        super(Difference, self).__init__()
        self.a, self.b = a, b
    def simplifications(self):
        for a in self.a.simplifications():
            yield Difference(a,self.b)
        for b in self.b.simplifications():
            yield Difference(self.a,b)
        yield self.a
        
    def toTrace(self):
        return self.a.toTrace() + self.b.toTrace() + [self]

    def _scad(self):
        return "difference() {\n" + self.a._scad() + "\n" + self.b._scad() + "\n}"

    def translate(self,*d):
        return Difference(self.a.translate(*d),self.b.translate(*d))

    def extent(self):
        a = self.a
        b = self.b
        a1,a2 = a.extent()
        b1,b2 = b.extent()
        return np.stack([a1,b1]).min(0), np.stack([a2,b2]).max(0)
    def scale(self,s):
        return Difference(self.a.scale(s), self.b.scale(s))
    def discrete(self,cs):
        return Difference(self.a.discrete(cs), self.b.discrete(cs))

    def __str__(self):
        return f"(- {self.a} {self.b})"
        
    def children(self): return [self.a, self.b]

    def serialize(self):
        return ('-',self.a,self.b)

    def __eq__(self, o):
        return isinstance(o, Difference) and self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash(('-', hash(self.a), hash(self.b)))
    
    def __contains__(self, p):
        return p in self.a and (not (p in self.b))

    def removeDeadCode(self):
        a = self.a.removeDeadCode()
        b = self.b.removeDeadCode()

        if a is None: return None
        if b is None: return a

        self = Difference(a,b)

        me = self.render()
        l = a.render()
        r = b.render()
        if np.all(l < r): return None
        if np.all(me == l): return a
        if np.all(me == r): return b
        return self

    def flipY(self): return Difference(self.a.flipY(),
                                       self.b.flipY())
    def flipX(self): return Difference(self.a.flipX(),
                                       self.b.flipX())
    def flipZ(self): return Difference(self.a.flipZ(),
                                       self.b.flipZ())

    def _render(self,r=None):
        a = 1.*self.a.render(r)
        b = 1.*self.b.render(r)
        return np.clip(a - b,
                       0., 1.)

class Intersection(CSG):
    token = '*'
    type = arrow(tCSG, tCSG, tCSG)
    
    def __init__(self, a, b):
        super(Intersection, self).__init__()
        self.a, self.b = a, b
    def simplifications(self):
        yield self.a
        yield self.b
        for a in self.a.simplifications():
            yield Intersection(a,self.b)
        for b in self.b.simplifications():
            yield Intersection(self.a,b)
        

    def toTrace(self):
        return self.a.toTrace() + self.b.toTrace() + [self]

    def __str__(self):
        return f"(* {self.a} {self.b})"

    def _scad(self):
        return "intersection() {\n" + self.a._scad() + "\n" + self.b._scad() + "\n}"
        
    def children(self): return [self.a, self.b]

    def translate(self,*d):
        return Intersection(self.a.translate(*d),self.b.translate(*d))

    def extent(self):
        a = self.a
        b = self.b
        a1,a2 = a.extent()
        b1,b2 = b.extent()
        return np.stack([a1,b1]).min(0), np.stack([a2,b2]).max(0)
    def scale(self,s):
        return Intersection(self.a.scale(s), self.b.scale(s))
    def discrete(self,cs):
        return Intersection(self.a.discrete(cs), self.b.discrete(cs))

    def serialize(self):
        return ('*',self.a,self.b)

    def __eq__(self, o):
        return isinstance(o, Intersection) and self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash(('*', hash(self.a), hash(self.b)))
    
    def __contains__(self, p):
        return (p in self.a) and (p in self.b)

    def removeDeadCode(self):
        a = self.a.removeDeadCode()
        b = self.b.removeDeadCode()

        if a is None: return b
        if b is None: return a

        self = Intersection(a,b)

        me = self.render()
        l = a.render()
        r = b.render()
        
        if np.all(me <= 0.): return None
        if np.all(me == l): return a
        if np.all(me == r): return b
        return self

    def flipY(self): return Intersection(self.a.flipY(),
                                         self.b.flipY())
    def flipX(self): return Intersection(self.a.flipX(),
                                         self.b.flipX())
    def flipZ(self): return Intersection(self.a.flipZ(),
                                         self.b.flipZ())

    def _render(self,r=None):
        a = 1.*self.a.render(r)
        b = 1.*self.b.render(r)
        return np.clip(a*b,
                       0., 1.)

        
class Loop2(CSG):
    token = 'for2'
    type = arrow(tCSG,
                 integer(2,5), # repetition count
                 integer(0, RESOLUTION - 1), # dx
                 integer(0, RESOLUTION - 1), # dy
                 tCSG)
    
    def __init__(self, child, n, dx, dy):
        super(Loop2, self).__init__()
        self.child,self.n,self.dx,self.dy = child,n,dx,dy

    def translate(self,*d):
        return Loop2(self.child.translate(*d),self.n,self.dx,self.dy)

    def simplifications(self):
        yield self.child
        for i in range(2,self.n):
            yield Loop2(self.child,i,self.dx,self.dy)
        for child in self.child.simplifications():
            yield Loop2(child,self.n,self.dx,self.dy)
    def toTrace(self):
        return self.child.toTrace() + [ai]

    def __str__(self):
        return f"(for {str(self.child)} n={self.n} dx={self.dx} dy={self.dy})"

    def extent(self):
        a,b = self.child.extent()
        displacement = np.array([self.dx,self.dy])
        ps = np.stack(
            [a + displacement*n for n in range(self.n) ] + [b + displacement*n for n in range(self.n) ])
        return ps.min(0),ps.max(0)
    def scale(self,s):
        return Loop2(self.child.scale(s),self.n,
                     self.dx*s,self.dy*s)

    def children(self): return [self.child]

    def serialize(self):
        return (self.__class__.token,self.child,self.n,self.dx,self.dy)

    def removeDeadCode(self):
        child = self.child.removeDeadCode()
        return Loop2(child,self.n,self.dx,self.dy)
    
    def flipX(self): return Union(self.elements[0].flipX(),
                                  self.elements[1].flipX())
    def flipY(self): return Union(self.elements[0].flipY(),
                                  self.elements[1].flipY())
    def flipZ(self): return Union(self.elements[0].flipZ(),
                                  self.elements[1].flipZ())

    def _render(self,r=None):
        r = r or RESOLUTION
        dx = self.dx*r//RESOLUTION
        dy = self.dy*r//RESOLUTION
        reference = self.child.render(r)
        accumulator = reference
        for i in range(self.n-1):
            reference = np.copy(np.roll(reference, (dy,dx)))
            # if dx > 0: reference[:dx] = 0
            # elif dx < 0: reference[dx:] = 0
            # if dy > 0: reference[:dy] = 0
            # elif dy < 0: reference[dy:] = 0                
            accumulator = accumulator + reference
        return np.clip(accumulator,0.,1.)

dsl_3d = DSL([Union, Difference, Intersection, Cuboid, Sphere, Cylinder])
dsl_2d = DSL([Union, Difference, Intersection, Loop2, Rectangle, Circle])


def loadScad(path):
    import re
    
    with open(path, "r") as handle:
        source = handle.readlines()

    def parse():
        nonlocal source

        while all( c.isspace() for c in source[0] ):
            source.pop(0)

        translate = re.match(r"\s*translate\(\[([^,]+), ([^,]+), ([^,]+)\]\)", source[0])
        if translate:
            displacement = tuple(float(translate.group(j)) for j in range(1,4) )
            source[0] = source[0][translate.span()[1]:]
            return parse().translate(*displacement)

        rotate = re.match(r"\s*rotate\(\[([^,]+), ([^,]+), ([^,]+)\]\)", source[0])
        if rotate:
            displacement = tuple(float(rotate.group(j)) for j in range(1,4) )
            source[0] = source[0][rotate.span()[1]:]
            o = parse()
            return o.rotate(*placement)
            #o = ("rotate", displacement, parse())
            
            

        cuboid = re.match(r"\s*cube\(size = \[([^,]+), ([^,]+), ([^,]+)\], center = (true|false)\);", source[0])
        if cuboid:
            displacement = tuple(float(cuboid.group(j)) for j in range(1,4) )
            source[0] = source[0][cuboid.span()[1]:]
            if cuboid.group(4) == "true": # centering
                return Cuboid(-displacement[0]/2, -displacement[1]/2, -displacement[2]/2,
                              displacement[0]/2, displacement[1]/2, displacement[2]/2)
            else:
                return Cuboid(0.,0.,0.,*displacement)
                

        sphere = re.match(r"\s*sphere\(r = ([^,]+), [^)]+\);", source[0])
        if sphere:
            r = float(sphere.group(1))
            source[0] = source[0][sphere.span()[1]:]
            return Sphere(r,0.,0.,0.)

        cylinder = re.match(r"\s*cylinder\(h = ([^,]+), r1 = ([^,]+), r2 = ([^,]+), center = (true|false)[^)]+\);", source[0])
        if cylinder:
            h = float(cylinder.group(1))
            r1 = float(cylinder.group(2))
            r2 = float(cylinder.group(3))
            assert r1 == r2
            source[0] = source[0][cylinder.span()[1]:]
            if cylinder.group(4) == "true": # center
                print("centered cylinder")
                return Cylinder(r1,
                                0.,0.,-h/2,
                                0.,0., h/2)
            else:
                print("not centered cylinder")
                return Cylinder(r1,
                                0.,0.,0.,
                                0.,0.,h)

        union = re.match(r"\s*union\(\)\s*\{$", source[0])
        if union:
            source.pop(0)

            elements = []
            while True:
                elements.append(parse())

                while all( c.isspace() for c in source[0] ):
                    source.pop(0)

                if re.match(r"\s*\}$", source[0]):
                    source.pop(0)
                    break

            o = elements[0]
            for e in elements[1:]:
                o = Union(o,e)
            return o
        
        difference = re.match(r"\s*difference\(\)\s*\{$", source[0])
        if difference:
            source.pop(0)

            elements = []
            while True:
                elements.append(parse())

                while all( c.isspace() for c in source[0] ):
                    source.pop(0)

                if re.match(r"\s*\}$", source[0]):
                    source.pop(0)
                    break

            assert len(elements) == 2
            return Difference(*elements)

        intersection = re.match(r"\s*intersection\(\)\s*\{$", source[0])
        if intersection:
            source.pop(0)

            elements = []
            while True:
                elements.append(parse())

                while all( c.isspace() for c in source[0] ):
                    source.pop(0)

                if re.match(r"\s*\}$", source[0]):
                    source.pop(0)
                    break

            o = elements[0]
            for e in elements[1:]:
                o = Intersection(o,e)
            return o
            


        print("could not parse this line:")
        print(source[0])
        assert False

        
        

    o = parse()
    while len(source) > 0 and all( c.isspace() for c in source[0] ):
        source.pop(0)
    if source:
        print("had some leftover things to parse")
        print(source)
        assert False

    return o


def loadScads():
    s = loadScad("data/CSG/011/sketch_final.scad")
    import glob
    for f in glob.glob("data/CSG/011/sketch_final.scad"): # "data/CSG/**/*"
        print(f)
        try:
            scene = loadScad(f)
            print(scene)
            assert isinstance(scene,Intersection)
            boundingBox = scene.a
            union = scene.b
            assert isinstance(boundingBox,Cuboid)
            assert isinstance(union,Union)
            elements = []
            while True:
                elements.append(union.elements[1])
                assert not isinstance(union.elements[1],Union)
                union = union.elements[0]
                if not isinstance(union,Union):
                    elements.append(union)
                    break                
            
            print("Loaded each of these elements:")
            for e in elements:
                print(e)
            assert False

            print("before translating")
            print(scene.extent())
            m,_ = scene.extent()
            scene = scene.translate(-m[0],-m[1],-m[2])
            print("after translation")
            print(scene.extent())
            _,m = scene.extent()
            scene = scene.scale(RESOLUTION/m.max()).discrete(range(RESOLUTION))
            print(scene.extent())
            print("without that code")
            scene = scene.removeDeadCode()
            print(scene)
            if scene is None:
                print("no way of removing dead code")
            else:
                scene.show()

        except Exception as e:
            print(traceback.format_exc())
            print("FAIL",e)
            sys.exit(0)
    assert False


def random3D(maxShapes=13,minShapes=3,rotate=False):
    cs = range(0, RESOLUTION, int(RESOLUTION/8))
    def randomSpherical():
        r = random.choice([4,8,12])
        x = random.choice([c for c in cs if c - r >= 0 and c + r < RESOLUTION ])
        y = random.choice([c for c in cs if c - r >= 0 and c + r < RESOLUTION ])
        z = random.choice([c for c in cs if c - r >= 0 and c + r < RESOLUTION ])
        return Sphere(x,y,z,r)
    def randomCuboid():
        def randomPair():
            c0 = random.choice(cs)
            c1 = random.choice([c for c in cs if c != c0 ])
            return min(c0,c1),max(c0,c1)
        x0,x1 = randomPair()
        y0,y1 = randomPair()
        z0,z1 = randomPair()
        
        return Cuboid(x0,y0,z0,
                      x1,y1,z1)
    def randomCylinder():
        if (not rotate) or (random.random() < 0.7): # axis aligned
            r = random.choice([2,4,8,12])
            l = random.choice([4,8,12,16,20])
            # sample the center, aligned with the axis of the cylinder
            a = random.choice([c for c in range(RESOLUTION)
                               if c - l/2 >= 0 and c + l/2 < RESOLUTION and c + l//2 in cs and c - l//2 in cs ])
            b = random.choice([c for c in cs if c - r >= 0 and c + r < RESOLUTION ])
            c = random.choice([c for c in cs if c - r >= 0 and c + r < RESOLUTION ])

            # oriented along z axis
            if random.choice([False,False,True]):
                p0 = [b,c, a - l//2]
                p1 = [b,c, a + l//2]
            elif random.choice([False,True]): # oriented along y axis
                p0 = [b, a - l//2, c]
                p1 = [b, a + l//2, c]
            else: # oriented along x-axis
                p0 = [a - l//2, b, c]
                p1 = [a + l//2, b, c]
        elif random.random() < 0.5: # random 45 angle, in axis-aligned plane
            while True:
                r = random.choice([2,4,8,12])
                rd = r/(2**0.5)
                l = random.choice([4,8,12,16,20])
                axisCoordinates = [c for c in range(RESOLUTION)
                                   if c - l/2 - rd >= 0 and c + l/2 + rd <= RESOLUTION - 1 and c + l//2 in cs and c - l//2 in cs ]
                if axisCoordinates: break
                
            # sample the center, aligned with the axis of the cylinder
            a = random.choice(axisCoordinates)
            b = random.choice(axisCoordinates)
            c = random.choice([c for c in cs if c - r >= 0 and c + r < RESOLUTION ])

            yo = random.choice([1,-1])

            if random.random() < 0.33: # constant Z
                p0 = [a - l//2,b - yo*l//2,c]
                p1 = [a + l//2,b + yo*l//2,c]
                #(cylinder r=12 p0=[ 6 26 12] p1=[26  6 12])
            elif random.random() < 0.5: # constant Y
                p0 = [a - l//2,c,b - yo*l//2]
                p1 = [a + l//2,c,b + yo*l//2]
            else: # constant X
                p0 = [c,a - l//2,b - yo*l//2]
                p1 = [c,a + l//2,b + yo*l//2]
        else: # random 45 angle, not axis aligned
            while True:
                r = random.choice([2,4,8,12])
                rd = r/(2**0.5)
                l = random.choice([4,8,12,16,20])
                axisCoordinates = [c for c in range(RESOLUTION)
                                   if c - l/2 - rd >= 0 and c + l/2 + rd <= RESOLUTION - 1 and c + l//2 in cs and c - l//2 in cs ]
                if axisCoordinates: break
                
            # sample the center, aligned with the axis of the cylinder
            a = random.choice(axisCoordinates)
            b = random.choice(axisCoordinates)
            c = random.choice(axisCoordinates)

            yo = random.choice([1,-1])
            zo = random.choice([1,-1])
            p0 = [a - l//2,b - yo*l//2,c - zo*l//2]
            p1 = [a + l//2,b + yo*l//2,c + zo*l//2]
            
            

        return Cylinder(*([r] + p0 + p1))

    def randomShape():
        if random.random() < 0.15: return randomSpherical()
        if random.choice([True,False]): return randomCylinder()
        return randomCuboid()

    while True:
        s = None
        numberOfShapes = 0
        desiredShapes = random.choice(range(minShapes, 1 + maxShapes))
        for _ in range(desiredShapes):
            o = randomShape()
            if s is None:
                s = o
            else:
                if random.random() < 0.7:
                    new = Union(s,o)
                else:
                    new = Difference(s,o)
                # Change at least ten percent of the pixels
                oldOn = s.render().sum()
                newOn = new.render().sum()
                pc = abs(oldOn - newOn)/oldOn
                if pc < 0.1 or pc > 0.6:
                    continue
                s = new
        try:
            finalScene = s.removeDeadCode()
            assert np.all(finalScene.render() == s.render())
            finalScene = finalScene.removeCodeNotChangingProjections()
            s = finalScene
            break
        except BadCSG:
            continue

    return s

    
            
"""Neural networks"""
class ObjectEncoder(CNN):
    """Encodes a 2d object"""
    def __init__(self):
        super(ObjectEncoder, self).__init__(channels=2,
                                            inputImageDimension=RESOLUTION*2,
                                            filterSizes=[3,3,3,3],
                                            poolSizes=[2,2,1,1],
                                            numberOfFilters=[32,32,32,16])
                                            

    def forward(self, spec, obj):
        if isinstance(spec, list):
            # batching both along specs and objects
            assert isinstance(obj, list)
            B = len(spec)
            assert len(obj) == B
            return super(ObjectEncoder, self).forward(np.stack([np.stack([s,o]) for s,o in zip(spec, obj)]))
        elif isinstance(obj, list): # batched - expect a single spec and multiple objects
            spec = np.repeat(spec[np.newaxis,:,:],len(obj),axis=0)
            obj = np.stack(obj)
            return super(ObjectEncoder, self).forward(np.stack([spec, obj],1))
        else: # not batched
            return super(ObjectEncoder, self).forward(np.stack([spec, obj]))
        

class SpecEncoder(CNN):
    """Encodes a 2d spec"""
    def __init__(self):
        super(SpecEncoder, self).__init__(channels=1,
                                          inputImageDimension=RESOLUTION*2,
                                          filterSizes=[3,3,3,3],
                                          poolSizes=[2,2,1,1],
                                          numberOfFilters=[32,32,32,16])



        
class MultiviewEncoder(Module):
    def __init__(self, H=512,
                 nchannels=1):
        super(MultiviewEncoder, self).__init__()

        self.nviews = 6
        self.nchannels = nchannels

        # This will run six times
        layers = 2
        self.singleView = CNN(channels=self.nchannels, inputImageDimension=RESOLUTION, flattenOutput=False, layers=layers)

        self.mergeViews = CNN(channels=2*self.singleView.outputChannels,
                              inputImageDimension=self.singleView.outputResolution,
                              mp=1) # do not max pool while merging views
        self.flatten = nn.Sequential(nn.Linear(self.mergeViews.outputDimensionality*3, H),
                                     nn.ReLU(),
                                     nn.Linear(H,H))
        
        self.outputDimensionality = H
        
        self.finalize()

    def forward(self, v):
        """Expects: Bx6x(nchannels)xRESOLUTIONxRESOLUTION. Returns: BxH"""
        v = self.tensor(v)

        assert v.size(2) == self.nchannels
        assert v.size(1) == 6
        B = v.size(0)

        # Run the single view encoder - but first we have to flatten batch/view
        reshaped = v.view(B*6,self.nchannels,RESOLUTION,RESOLUTION)
        bv = self.singleView.encoder(reshaped)
        # bv: [6B, self.singleView.outputChannels, r,r]
        bv = bv.view(B,6,
                     self.singleView.outputChannels,
                     self.singleView.outputResolution, self.singleView.outputResolution).contiguous()



        
        # Merge the parallel views
        merged = \
          self.mergeViews.encoder(bv.view(B*3,2*self.singleView.outputChannels,
                                          self.singleView.outputResolution, self.singleView.outputResolution))
        # ys = []
        # for b in range(B):
        #     predictions = [] # 6 predictions - one for each view
        #     for viewing in range(6):
        #         prediction = self.singleView.encoder(v[b, viewing].unsqueeze(0)).squeeze(0)
        #         difference = prediction - bv[b,viewing]
        #         assert torch.all(difference.abs() < 0.0001)
        #         predictions.append(prediction)
        #     mergeInput = [torch.cat([predictions[0],predictions[1]]),
        #                   torch.cat([predictions[2],predictions[3]]),
        #                   torch.cat([predictions[4],predictions[5]])]
        #     MERGE = []
        #     for i,mi in enumerate(mergeInput):
        #         mo = self.mergeViews.encoder(mi.unsqueeze(0)).squeeze(0)
        #         MERGE.append(mo)
        #         y = merged[3*b + i]
        #         assert torch.all( (y - mo).abs() < 0.0001)
        #     yf = self.flatten(torch.cat(MERGE))
        #     print(yf.shape)
        #     ys.append(yf.unsqueeze(0))
        merged = merged.view(B,-1)
        y = self.flatten(merged)
        
        # assert torch.all( (torch.cat(ys) - y).abs() < 0.001 )

        return y


class MultiviewSpec(Module):
    def __init__(self):
        super(MultiviewSpec, self).__init__()
        self.encoder = MultiviewEncoder()
        self.outputDimensionality = self.encoder.outputDimensionality
        self.finalize()

    def forward(self, v):
        """v: either Bx(voxels) or (voxels)"""
        if len(v.shape) == 4:
            v = self.tensor(np.array([ voxels2dm(v[b]) for b in range(v.shape[0])]))
            return self.encoder(v.unsqueeze(2))
        elif len(v.shape) == 3:
            v = self.tensor(voxels2dm(v))
            return self.encoder(v.unsqueeze(0).unsqueeze(2)).unsqueeze(0)
        else:
            assert False

class MultiviewObject(Module):
    def __init__(self):
        super(MultiviewObject, self).__init__()
        self.encoder = MultiviewEncoder(nchannels=2)
        self.outputDimensionality = self.encoder.outputDimensionality
        self.finalize()

    def forward(self, spec, obj):
        if isinstance(spec, list):
            # batching both along specs and objects
            assert isinstance(obj, list)
            B = len(spec)
            assert len(obj) == B
            spec = self.tensor(np.stack([ np.stack(voxels2dm(s)) for s in spec])).unsqueeze(2)
            obj = self.tensor(np.stack([ np.stack(voxels2dm(o)) for o in obj])).unsqueeze(2)
            return self.encoder(torch.cat([spec,obj],2))
        elif isinstance(obj, list): # batched - expect a single spec and multiple objects
            spec = np.stack([voxels2dm(spec)])
            spec = self.tensor(np.repeat(spec[np.newaxis,:,:],len(obj),axis=0)).unsqueeze(2)
            obj = self.tensor(np.stack([ np.stack(voxels2dm(o)) for o in obj])).unsqueeze(2)
            return self.encoder(torch.cat([spec,obj],2))
        else: # not batched
            spec = self.tensor(np.stack([voxels2dm(spec)])).unsqueeze(0).unsqueeze(2)
            obj = self.tensor(np.stack([voxels2dm(obj)])).unsqueeze(0).unsqueeze(2)
            return self.encoder(torch.cat([spec,obj],2)).squeeze(0)
        

class HeatMap(Module):
    def __init__(self):
        super(HeatMap, self).__init__()

        h = 64
        o = 3
        def residual():
            return ResidualCNNBlock(h)
        self.model = nn.Sequential(nn.Conv2d(1, h, 3, padding=1),
                                   nn.ReLU(),
                                   residual(), residual(), residual(), residual(),
                                   nn.Conv2d(h, o, 3, padding=1))

        self.finalize()

    def forward(self, x):
        return self.model(x)


def learnHeatMap(checkpoint='checkpoints/hm.p'):
    data = getTrainingData('CSG_data.p')
    B = 32

    if os.path.exists(checkpoint):
        print("Found checkpoint - going to do a demo, and then resume training")
        hm = torch.load(checkpoint)
        with torch.no_grad():
            ps = [data() for _ in range(B)]
            xs = hm.device(torch.tensor(np.array([p.execute() for p in ps ])).float())
            xs = xs.unsqueeze(1)
            ys = hm.tensor(torch.tensor(1.*np.array([p.heatMapTarget() for p in ps])).float())
            ys = ys.permute(*[0,3,1,2])
            predictions = F.sigmoid(hm(xs)).cpu().numpy()

        os.system("mkdir  -p data/hm/")

        for b in range(B):
            i = ps[b].execute()
            saveMatrixAsImage(i, f"data/hm/{b}.png")
            t = ps[b].heatMapTarget()
            for j in range(3):
                saveMatrixAsImage(t[:,:,j], f"data/hm/{b}_{j}_target.png")
                saveMatrixAsImage(predictions[b,j,:,:], f"data/hm/{b}_{j}_prediction.png")
    else:        
        hm = HeatMap()
    startTime = time.time()
    optimizer = torch.optim.Adam(hm.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    i = 0
    while time.time() < startTime + 3600*5:
        i += 1
        
        hm.zero_grad()
        
        ps = [data() for _ in range(B)]
        xs = hm.device(torch.tensor(np.array([p.execute() for p in ps ])).float())
        xs = xs.unsqueeze(1)
        ys = hm.tensor(torch.tensor(1.*np.array([p.heatMapTarget() for p in ps])).float())
        ys = ys.permute(*[0,3,1,2])
        predictions = hm(xs)
        l = F.binary_cross_entropy_with_logits(predictions,ys,reduction='sum')/B
        l.backward()
        optimizer.step()

        if i%1000 == 1:
            print(l.data)
            torch.save(hm, checkpoint)
        
    


    
"""Training"""
def randomScene(resolution=32, maxShapes=3, minShapes=1, verbose=False, export=None,
                nudge=False, translate=False, loops=False):
    assert not translate
    import matplotlib.pyplot as plot

    def translation(x,y,child):
        if not nudge: return Translation(x,y,child)
        for _ in range(x):
            child = Translation(1,0,child)
        for _ in range(y):
            child = Translation(0,1,child)
        return child
    dc = 16 # number of distinct coordinates
    choices = [c
               for c in range(0, resolution, resolution//dc) ]
    def slantedQuadrilateral():
        while True:
            x0 = random.choice(choices)
            y0 = random.choice(choices)
            a = [a for a in range(resolution)
                 if x0 - a in choices and y0 + a in choices and a > 1]
            if len(a) == 0: continue
            a = random.choice(a)
            b = [b for b in range(resolution)
                 if x0 + b in choices and y0 + b in choices and x0 - a + b in choices and y0 + a + b in choices and b > 1]
            if len(b) == 0: continue
            b = random.choice(b)
            return Rectangle(x0,y0,
                             x0 - a, y0 + a,
                             x0 - a + b, y0 + a + b,
                             x0 + b, y0 + b)

    def quadrilateral():
        if random.random() < 0.5:
            x0 = random.choice(choices[:-1])
            y0 = random.choice(choices[:-1])
            x1 = random.choice([x for x in choices if x > x0 ])
            y1 = random.choice([y for y in choices if y > y0 ])
            return Rectangle(x0,y0,
                             x0,y1,
                             x1,y1,
                             x1,y0)
        else:
            return slantedQuadrilateral()
    def circular():
        d = random.choice([d for d in choices if d > 4])
        x = random.choice([x for x in choices if x - d/2 >= 0 and x + d/2 < resolution ])
        y = random.choice([y for y in choices if y - d/2 >= 0 and y + d/2 < resolution ])
        return Circle(x,y,d)
    def triangle():
        q = slantedQuadrilateral()
        x0 = q.x1
        y0 = q.y0
        x1 = q.x3
        y1 = q.y1
        p = Rectangle(x0,y0,
                      x0,y1,
                      x1,y1,
                      x1,y0)
        return p - q        


    while True:
        s = None
        numberOfShapes = 0
        desiredShapes = random.choice(range(minShapes, 1 + maxShapes))
        haveLooped = False
        for _ in range(desiredShapes):
            if loops and not haveLooped and s is None:
                while True:
                    o = quadrilateral() if random.choice([True,False]) else circular()
                    minimum, maximum = o.extent()
                    size = maximum - minimum
                    if random.random() > 0.5: # translate along X
                        cs = [c for c in choices
                              if maximum[0] + 2*c < resolution and c > size[0] ]
                        if len(cs) == 0: continue
                        dx = random.choice(cs)
                        dy = 0
                    else:
                        cs = [c for c in choices
                              if maximum[1] + 2*c < resolution and c > size[1] ]
                        if len(cs) == 0: continue
                        dy = random.choice(cs)
                        dx = 0
                    n = random.choice([n for n in [2,3,4,5]
                                       if maximum[0] + n*dx < resolution and minimum[1] + n*dy < resolution ])
                    s = Loop2(o,n,
                              dx,dy)
                    s.render()
                    break
                break
            
            
            o = quadrilateral() if random.choice([True,False]) else circular()
            if s is None:
                s = o
            else:
                if random.choice([True,True,False]):
                    new = s + o
                else:
                    new = s - o
                # Change at least ten percent of the pixels
                oldOn = s.render().sum()
                newOn = new.render().sum()
                pc = abs(oldOn - newOn)/oldOn
                if pc < 0.1 or pc > 0.6:
                    continue
                s = new
        try:
            finalScene = s.removeDeadCode()
            assert np.all(finalScene.render() == s.render())
            s = finalScene.removeCodeNotChangingRender()
            break
        except BadCSG:
            continue
    
    if verbose:
        print(s)
        print(ProgramGraph.fromRoot(s, oneParent=True).prettyPrint())
        s.show()
    if export:
        plot.imshow(s.execute())
        plot.savefig(export)
        plot.imshow(s.highresolution(256))
        plot.savefig(f"{export}_hr.png")
        if not translate:
            for j in range(3):
                plot.imshow(s.heatMapTarget()[:,:,j])
                plot.savefig(f"{export}_hm_{j}.png")
    
    return s

def trainCSG(m, getProgram, trainTime=None, checkpoint=None):
    print("cuda?",m.use_cuda)
    assert checkpoint is not None, "must provide a checkpoint path to export to"
    
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    
    startTime = time.time()
    reportingFrequency = 100
    totalLosses = []
    movedLosses = []
    iteration = 0

    B = 16

    while trainTime is None or time.time() - startTime < trainTime:
        ss = [getProgram() for _ in range(B)]
        ls = m.gradientStepTraceBatched(optimizer, [(s, s.toTrace())
                                                    for s in ss])
        for l in ls:
            totalLosses.append(sum(l))
            movedLosses.append(sum(l)/len(l))
        iteration += 1
        if iteration%reportingFrequency == 1:
            print(f"\n\nAfter {iteration*B} training examples...\n\tTrace loss {sum(totalLosses)/len(totalLosses)}\t\tMove loss {sum(movedLosses)/len(movedLosses)}\n{iteration*B/(time.time() - startTime)} examples/sec\n{iteration/(time.time() - startTime)} grad steps/sec")
            totalLosses = []
            movedLosses = []
            torch.save(m, checkpoint)



def testCSG(m, getProgram, timeout, export):
    random.seed(0)
    oneParent = m.oneParent
    print(f"One parent restriction?  {oneParent}")
    solvers = [# RandomSolver(dsl),
               # MCTS(m, reward=lambda l: 1. - l),
               #SMC(m),
        # BeamSearch(m, criticCoefficient=1),
                # BeamSearch(m, criticCoefficient=0.),
               ForwardSample(m, maximumLength=18)]
    loss = lambda spec, program: 1-max( o.IoU(spec) for o in program.objects() ) if len(program) > 0 else 1.

    twodimensional = True
    def exportProgram(program, path):
        if program is None:
            if twodimensional:
                program = Difference(Circle(1,1,1),Circle(1,1,1))
            else:
                program = Difference(Sphere(1,1,1,1),Sphere(1,1,1,1))
                
        if twodimensional:
            saveMatrixAsImage(program.render(256), path)
        else:
            program.show(export=path)            

    testResults = [[] for _ in solvers]

    os.system("mkdir data/test")

    if isinstance(getProgram, list):
        specs = getProgram
    else:
        specs = [getProgram() for _ in range(30) ]
    twodimensional = len(specs[0].execute().shape) == 2
    for ti,spec in enumerate(specs):
        print("Trying to explain the program:")
        print(ProgramGraph.fromRoot(spec, oneParent=oneParent).prettyPrint())
        print()
        
        exportProgram(spec, "data/test/%03d.png"%ti)
        with open("data/test/%03d_spec.pickle"%ti,"wb") as handle:
            pickle.dump(spec, handle)
        for n, solver in enumerate(solvers):
            print(f"Running solver {solver.name}")
            solver.maximumLength = len(ProgramGraph.fromRoot(spec).nodes) + 1
            testSequence = solver.infer(spec, loss, timeout)
            if len(testSequence) == 0:
                testSequence = [SearchResult(ProgramGraph([]), 1., 0., 1)]
            testResults[n].append(testSequence)
            for result in testSequence:
                print(f"After time {result.time}, achieved loss {result.loss} w/")
                print(result.program.prettyPrint())
                print()
            if len(testSequence) > 0:
                obs = testSequence[-1].program.objects()
                if len(obs) == 0:
                    bestProgram = None
                else:
                    bestProgram = max(obs, key=lambda bp: bp.IoU(spec))
                with open("data/test/%03d_%s.pickle"%(ti,solver.name),"wb") as handle:
                    pickle.dump(bestProgram, handle)
                exportProgram(bestProgram,
                              "data/test/%03d_%s.png"%(ti,solver.name))
                if not twodimensional:
                    bestProgram.scad("data/test/%03d_%s.scad"%(ti,solver.name))
                

    names = [s.name for s in solvers]
    if export is not None:
        with open(os.path.splitext(export)[0] + ".pickle","wb") as handle:
             pickle.dump(list(zip(names, testResults)),handle)
    plotTestResults(testResults, timeout,
                    defaultLoss=1.,
                    names=names,
                    export=export)

def plotTestResults(testResults, timeout, defaultLoss=None,
                    names=None, export=None):
    import matplotlib.pyplot as plot

    def averageLoss(n, predicate):
        results = testResults[n] # list of list of results, one for each test case
        # Filter out results that occurred after time T
        results = [ [r for r in rs if predicate(r)]
                    for rs in results ]
        losses = [ min([defaultLoss] + [r.loss for r in rs]) for rs in results ]
        return sum(losses)/len(losses)

    plot.figure()
    plot.xlabel('Time')
    plot.ylabel('Average Loss')
    plot.ylim(bottom=0.)
    for n in range(len(testResults)):
        xs = list(np.arange(0,timeout,0.1))
        plot.plot(xs, [averageLoss(n,lambda r: r.time < x) for x in xs],
                  label=names[n])
    plot.legend()
    if export:
        plot.savefig(export)
    else:
        plot.show()
    plot.figure()
    plot.xlabel('Evaluations')
    plot.ylabel('Average Loss')
    plot.ylim(bottom=0.)
    for n in range(len(testResults)):
        xs = list(range(max(r.evaluations for tr in testResults[n] for r in tr )))
        plot.plot(xs, [averageLoss(n,lambda r: r.evaluations < x) for x in xs],
                  label=names[n])
    plot.legend()
    if export:
        plot.savefig(f"{export}_evaluations.png")
    else:
        plot.show()
        
        
    
def makeTrainingData():
    import os
    import matplotlib.pyplot as plot
    
    data = {} # Map from image to (size, {programs})
    lastUpdate = time.time()
    n_samples = 0
    maximumSize = 0
    startTime = time.time()
    while time.time() < startTime + 3600*5:
        n_samples += 1
        program = randomScene(maxShapes=20, minShapes=5)
        size = len(program.toTrace())
        im = program.render()
        im = tuple( im[x,y] > 0.5
                    for x in range(RESOLUTION)
                    for y in range(RESOLUTION) )
        program.clearRendering()
        if im in data:
            oldSize, oldPrograms = data[im]
            if oldSize < size:
                pass
            elif oldSize == size:
                data[im] = (size, {program}|oldPrograms)
            elif oldSize > size:
                data[im] = (size, {program})
            else:
                assert False
        else:
            data[im] = (size, {program})
            maximumSize = max(maximumSize,size)

        if time.time() > lastUpdate + 10:
            print(f"After {n_samples} samples; {n_samples/(time.time() - startTime)} samples per second")
            for sz in range(maximumSize):
                n = sum(size == sz
                        for (size,_) in data.values() )
                print(f"{n} images w/ programs of size {sz}")
            print()
            lastUpdate = time.time()

    with open('CSG_data.p','wb') as handle:
        pickle.dump(data, handle)


def getTrainingData(path):
    import copy
    
    with open(path,'rb') as handle:
        data = pickle.load(handle)
    print(f"Loaded {len(data)} images from {path}")
    print(f"Contains {sum(len(ps) for _,ps in data.values() )} programs")
    data = [list(ps) for _,ps in data.values()]

    def getData():
        programs = random.choice(data)
        # make a deep copy because we are caching the renders, and we want these to be garbage collected
        p = copy.deepcopy(random.choice(programs))
        if random.choice([True,False]):
            # print("X flip")
            # showMatrixAsImage(p.highresolution(128),
            #                   p.flipX().highresolution(128))
            p = p.flipX()
        if random.choice([True,False]):
            # print("Y flip")
            # showMatrixAsImage(p.highresolution(128),
            #                   p.flipY().highresolution(128))
            p = p.flipY()
            
        return p

    return getData
        
if __name__ == "__main__":
    m = NoExecution(SpecEncoder(), dsl_2d)
    p = Union(Circle(1,2,3),
              Circle(2,21,9))

    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    while True:
        losses = m.gradientStepTraceBatched(optimizer, [(p,p.toTrace())])
        L = sum(l for ls in losses for l in ls  )
        print(L)
        if L < 0.2:
            print(m.sample(p))
