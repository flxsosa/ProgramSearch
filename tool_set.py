from CAD import *

"""basically all of this is broken"""

if __name__ == '__main__':
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
                  
                  
    def rectangle(x,y,w,h):
        return Rectangle(x*2 - w, y*2 - h,
                         x*2 - w, y*2 + h,
                         x*2 + w, y*2 + h,
                         x*2 + w, y*2 - h)
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

    # shovel 
    def shovel():
        s1 = rectangle(8, 7, 1, 8)
        s2 = circle(8, 13, 6)
        s3 = rectangle(8, 3, 3, 1)
        s4 = rectangle(8, 12, 3, 2)
        return s1 + s2 + s3 + s4

    shovel().export("demo/shovel.png",256)

    # key 
    def key():
        s1 = rectangle(8, 8, 8, 1)
        s2 = circle(13, 8, 8)
        s3 = circle(13, 8, 4)
        s4 = rectangle(5, 8, 3, 2)
        s5 = rectangle(5, 8, 1, 2)
        return s1 + (s2 - s3) + (s4 - s5)

    key().export("demo/key.png",256)

    # hammer 
    def hammer():
        s1 = rectangle(9, 8, 6, 2)
        s2 = rectangle(5, 8, 3, 5)
        return s1 + s2

    hammer().export("demo/hammer.png",256)

    # plier 
    def plier():
        s1 = rectangle(6, 8, 8, 1)
        s2 = slanted(10, 8, 5, 1)
        s3 = circle(12, 8, 8)
        s4 = rectangle(14, 6, 3, 3)
        return s1 + s2 + (s3 - s4)

    plier().export("demo/plier.png",256)

    # double_wrench 
    def double_wrench():
        s1 = slanted(11, 4, 7, 1)
        s2 = circle(12, 4, 6)
        s3 = rectangle(12, 4, 1, 1)
        s4 = circle(4, 12, 6)
        s5 = rectangle(4, 13, 1, 2)
        return s1 + (s2 - s3) + (s4 - s5)

    double_wrench().export("demo/double_wrench.png",256)

    # mag 
    def mag():
        s1 = circle(8, 8, 20)
        s2 = circle(8, 8, 16)
        s3 = slanted(11, 11, 1, 3)
        return (s1 - s2) + s3

    mag().export("demo/mag.png",256)

    # sickle 
    def sickle():
        s1 = circle(8, 7, 20)
        s2 = circle(7, 7, 18)
        s3 = slanted(5, 10, 3, 1)
        return (s1 - s2) + s3

    sickle().export("demo/sickle.png",256)

    # comrad 
    def comrad():
        s1 = circle(8, 7, 22)
        s2 = circle(7, 7, 20)
        s3 = slanted(5, 10, 3, 1)
        s4 = slanted(8, 7, 1, 6)
        s5 = slanted(8, 5, 3, 2)
        return (s1 - s2) + s3 + s4 + s5

    comrad().export("demo/comrad.png",256)
