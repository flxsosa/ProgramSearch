from CAD import *

def make3DTools():
    def cup():
        r1 = 12
        r2 = 8
        l1 = 20
        bottom_thickness = 4
        x1, y1, z1 = 12, 12, 4

        cup_body = Cylinder(r1, x1, y1, z1, 
                                x1, y1, z1 + l1)

        cup_hole = Cylinder(r2, x1, y1, z1 + bottom_thickness, 
                                x1, y1, z1 + l1)

        r3 = 8
        r4 = 4
        zz = 16
        yy = 12
        xx = 24
        cup_handle = Cylinder(r3, xx, yy - 4, zz,
                                  xx, yy + 4, zz)
        cup_handle_hole = Cylinder(r4, xx, yy - 4, zz,
                                       xx, yy + 4, zz)

        return cup_body + (cup_handle - cup_handle_hole)- cup_hole 

    def bob():
        body = Cuboid(8, 8, 8,
                      24,16,28)
        arm1 = Cuboid(24, 8, 16,
                      24+8, 8 + 4, 16 + 4)
        arm2 = Cuboid(0, 8, 16,
                      8, 8 + 4, 16 + 4)
        leg1 = Cuboid(8, 8, 0,
                      8 + 4, 8 + 4, 0 + 8)
        leg2 = Cuboid(12 + 8, 8, 0,
                      12 + 8 + 4, 8 + 4, 0 + 8)
        smile = Sphere(16, 8, 20, 4)
        smile_sub = Cuboid(12, 4, 20,
                           12+8, 4 + 8, 20 + 8) 

        return body + arm1 + arm2 + leg1 + leg2 - (smile - smile_sub)

    def glass():
        x, y = 12,12
        body = Cylinder(8, x, y, 28,
                           x, y, 20)
        body1 = Sphere(x, y, 16, 8)
        body_sub = Cylinder(4, x, y, 28,
                               x, y, 20)
        bot1 = Sphere(x, y, 12, 4)
        bot2 = Sphere(x, y, 4, 8)
        bot_sub = Cuboid(0, 0, 0, 20, 20, 4)
        return (body + body1 - body_sub) + bot1 + (bot2 - bot_sub)

    def cake():
        x, y = 12,12
        body = Cylinder(12, x, y, 16,
                            x, y, 4)
        cut1 = Cuboid(0, 0, 0,
                      28, 8, 20)
        cut2 = Cuboid(0, 0, 0,
                      12, 28, 20)
        cherry = Sphere(16, 16, 20, 4)
        return body - cut1 - cut2 + cherry

    def lamp():
        x, y = 12,12
        light = Cylinder(8, x, y, 24,
                            x, y, 16)
        light1 = Sphere(x, y, 24, 8)

        stem = Cylinder(12, x, y-4, 12,
                            x, y, 12)
        stem_sub = Cylinder(8, x, y-4, 12,
                               x, y, 12)
        stem_half = Cuboid(0, 0, 0,
                           x, 28, 28)
        bot = Cylinder(8, x, y, 0,
                          x, y, 4)

        return light + light1 + (stem - stem_sub - stem_half) + bot



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
    # return [car()]

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
        furnitures = chairs + tables
    
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
    ] + \
    furnitures + \
    [car()] +\
    [cake()] +\
    [lamp()] +\
    [glass()] +\
    [bob()] +\
    [cup()] 
                

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

    
