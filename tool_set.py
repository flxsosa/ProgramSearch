from CAD import *

def make3DTools():
    def car():
        l1 = 8
        r = 8
        l2 = 12

        h1 = 8
        h2 = r

        Y = 16

        W = 4

        z = Cuboid(0,0,0,
                   l1 + r + l2,Y,h1)
        z = z + Cylinder(r,
                         l1 + r,0,h1,
                         l1 + r,Y,h1)
        z = z + Cuboid(l1 + r,0,h1,
                       l1 + r + l2,Y,h1 + h2)
        #z = z.translate(4,4,8)
        z = z + Cylinder(4,
                         4,0,0,
                         4,W,0)
        z = z + Cylinder(4,
                         4,Y - W,0,
                         4,Y,0)
        z = z + Cylinder(4,
                         l1 + r + l2 - 4,Y - W,0,
                         l1 + r + l2 - 4,Y,0)
        z = z + Cylinder(4,
                         l1 + r + l2 - 4,0,0,
                         l1 + r + l2 - 4,W,0)
        
        return z.translate(0,0,4)
    return [car()]
    def cylinderArray(r,nc,nr,spacing=12,yspacing=None,z0=0,z1=12):
        yspacing = yspacing or spacing
        z = None
        for x in range(nc):
            for y in range(nr):
                b = Cylinder(r,
                             r + x*spacing,
                             r + y*yspacing,
                             z0,
                             r + x*spacing,
                             r + y*yspacing,
                             z1)
                if z is None:
                    z = b
                else:
                    z = z + b
        return z

    def table(height, nc,nr, spacing):
        z = cylinderArray(4,nc,nr,spacing,z0=0,z1=height)
        z = z + Cuboid(0,0,height,
                       spacing*(nc - 1) + 2*4,
                       spacing*(nr - 1) + 2*4,
                       height + 4)

        assert z.extent()[1].max() < 32, f"TABLE {height} {nc} {nr} {spacing}"
        return z

    def chair(z1,r):
        # r = 2
        # z1 = 12
        z = cylinderArray(r,2,2,16,z0=0,z1=z1).translate(2,2,0)
        z = z + Cuboid(0,0,z1,
                       20 + 2*r, 20, z1 + 4)
        z = z + Cuboid(0, 16 + 0,  z1 + 4,
                       20 + 2*r, 16 + 4, 28)
        return z

    if True:
        chairs = [chair(z1,r)
                for z1 in [8,12] for r in [2] ]
        tables = [table(height,nc,nr,16)
                  for nc in [2] for nr in [2] for height in [12,16,20,24] ]
        return chairs + tables
    
    def Lego(r,w,h,spacing=12):
        z = None
        for x in range(w):
            for y in range(h):
                b = Cylinder(r,
                             r + x*spacing,
                             r + y*spacing,
                             12,
                             r + x*spacing,
                             r + y*spacing,
                             12 + 4)
                if z is None:
                    z = b
                else:
                    z = z + b
        z = z + Cuboid(0,0,0,
                       spacing*(w - 1) + 2*r,
                       spacing*(h - 1) + 2*r,12)
        return z

        
    return [Lego(4,w,h,8)
            for w in range(1,4)
            for h in range(1,4)
    ] + \
    [Lego(4,w,h,12)
            for w in range(1,3)
            for h in range(1,3)
    ]
                

def make2DTools():
    import os
    os.system("mkdir  -p demo")
    everyTool = []


    def Zelda():
        z = Rectangle(12+4,12,
                      6+4,18,
                      12+4,24,
                      18+4,18)
        
        z = z - Rectangle(6+4,12,
                          6+4,18,
                          18+4,18,
                          18+4,12)
        z = z + Rectangle(18+4,6,
                          12+4,12,
                          18+4,18,
                          24+4,12)
        z = z + Rectangle(0+4,12,
                          6+4,18,
                          12+4,12,
                          6+4,6)
        z = z - Rectangle(0+4,0,
                          0+4,12,
                          24+4,12,
                          24+4,6)
        return z
        
    Zelda().export("demo/Zelda.png",256)
    everyTool.append(Zelda())
                  
                  
    def rectangle(x,y,w,h):
        return Rectangle(x*2 - w, y*2 - h,
                         x*2 - w, y*2 + h,
                         x*2 + w, y*2 + h,
                         x*2 + w, y*2 - h)

    def rectangle2(x,y,w,h):
        return Rectangle(x*2 - w + 1, y*2 - h + 1,
                         x*2 - w + 1, y*2 + h + 1,
                         x*2 + w + 1, y*2 + h + 1,
                         x*2 + w + 1, y*2 - h + 1)
    def slanted(x,y,w,h):
        x = x*2
        y = y*2
        w = w*2
        h = h*2
        return Rectangle(x,y,
                         x - w,y + w,
                         x - w + h,y + w + h,
                         x + h,y + h)
    def circle(x,y,d):
        return Circle(x*2,y*2,d)

    # wrench
    def wrench():
        s1 = rectangle(8, 9, 2, 8)
        s2 = circle(8, 4, 10)
        s3 = slanted(5,0,1,3)
        return s1 + s2 - s3

    wrench().export("demo/wrench.png",256)
    everyTool.append(wrench())

    # shovel 
    def shovel():
        s1 = rectangle(8, 6, 2, 10)
        s2 = circle(8, 12, 8)
        s3 = rectangle(8, 10, 4, 4)
        s4 = rectangle(8, 2, 4, 2)
        return s1 + s2 + s3 + s4
    # return s1 + s2 + s3 + s4

    shovel().export("demo/shovel.png",256)
    everyTool.append(shovel())

    # key 
    def key():
        s1 = rectangle(8, 8, 8, 2)
        s2 = circle(12, 8, 8)
        s3 = circle(12, 8, 4)
        s4 = slanted(4, 7, 1, 1)
        s5 = slanted(6, 6, 1, 1)
        s6 = slanted(8, 6, 1, 1)
        return s1 + (s2 - s3) + s4 + s5 + s6

    key().export("demo/key.png",256)
    everyTool.append(key())

    # hammer 
    def hammer():
        s1 = rectangle(10, 8, 6, 2)
        s2 = rectangle(6, 8, 4, 6)
        return s1 + s2

    hammer().export("demo/hammer.png",256)
    everyTool.append(hammer())

    # plier 
    def plier():
        s1 = rectangle(6, 8, 10, 2)
        s2 = slanted(10, 8, 5, 1)
        s3 = circle(12, 8, 8)
        s4 = rectangle(14, 6, 4, 4)
        s5 = slanted(12, 4, 2, 2)
        #return s4 + s5
        return s1 + s2 + (s3 - (s4 - s5))

    plier().export("demo/plier.png",256)
    everyTool.append(plier())

    # umbrella 
    def umbrella():
        s1 = circle(8, 6, 24)
        s3 = rectangle(8, 10, 20, 8)
        s4 = rectangle2(8, 8, 1, 7)
        s5 = circle(7,12,8)
        s7 = circle(7,12,4)
        s6 = rectangle(7,10,4,4)
        return (s1 - s3) + s4 + (s5 -s7 - s6)

    umbrella().export("demo/umbrella.png",256)
    everyTool.append(umbrella())

    # mag 
    def mag():
        s1 = circle(8, 8, 20)
        s2 = circle(8, 8, 16)
        s3 = slanted(11, 11, 1, 3)
        return (s1 - s2) + s3

    mag().export("demo/mag.png",256)
    everyTool.append(mag())

    # sickle 
    def sickle():
        s1 = circle(8, 7, 20)
        s2 = circle(7, 7, 18)
        s3 = slanted(5, 10, 3, 1)
        return (s1 - s2) + s3

    sickle().export("demo/sickle.png",256)
    everyTool.append(sickle())

    # comrad 
    def comrad():
        s1 = circle(8, 7, 22)
        s2 = circle(7, 7, 20)
        s3 = slanted(5, 10, 3, 1)
        s4 = slanted(8, 7, 1, 6)
        s5 = slanted(8, 5, 3, 2)
        return (s1 - s2) + s3 + s4 + s5

    comrad().export("demo/comrad.png",256)
    everyTool.append(comrad())
    return everyTool

if __name__ == '__main__':
    progs = make2DTools()
    import re
    for i, p in enumerate(progs):
        p_str = str(p)
        #print (p_str)
        nums = [int(s) for s in re.findall('\\d+', p_str)]
        #print (nums)
        parity_even = [x % 2 == 0 for x in nums]
        assert (all(parity_even))
        print (f"shape {i} passed even coordinate check")

    
