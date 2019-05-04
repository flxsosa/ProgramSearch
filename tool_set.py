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
        s1 = tool.Rect(8, 9, 2, 8, 0)
        s2 = tool.Circ(8, 4, 5)
        s3 = tool.Rect(7, 3, 3, 2, 1)
        s = s1 + (s2 - s3)
        scipy.misc.imsave('demo/wrench.png', s.to_np())
    def wrench():
        s1 = rectangle(8, 9, 2, 8)
        s2 = circle(8, 4, 10)
        s3 = slanted(8,4,1,3)
        return s1 + s2 - s3

    wrench().export("demo/whatever.png",256)
    assert False

    # shovel 
    def shovel():
        s1 = tool.Rect(8, 7, 1, 8, 0)
        s2 = tool.Circ(8, 13, 3)
        s3 = tool.Rect(8, 3, 3, 1, 0)
        s4 = tool.Rect(8, 12, 3, 2, 0)
        s = s1 + s2 + s3 + s4
        scipy.misc.imsave('demo/shovel.png', s.to_np())
    shovel()

    # key 
    def key():
        s1 = tool.Rect(7, 8, 1, 8, 2)
        s2 = tool.Circ(12, 8, 4)
        s3 = tool.Circ(12, 8, 2)
        s4 = tool.Rect(5, 7, 3, 2, 0)
        s5 = tool.Rect(5, 7, 1, 3, 0)
        s = s1 + (s2 - s3) + (s4 - s5)
        scipy.misc.imsave('demo/key.png', s.to_np())
    key()

    # Ibeam 
    def Ibeam():
        s1 = tool.Rect(8, 8, 2, 8, 0)
        s2 = tool.Rect(8, 4, 5, 2, 0)
        s3 = tool.Rect(8, 12, 5, 2, 0)
        s = s1 + s2 + s3
        scipy.misc.imsave('demo/Ibeam.png', s.to_np())
    Ibeam()

    # dumbell 
    def dumbell():
        s1 = tool.Rect(8, 8, 2, 8, 2)
        s2 = tool.Circ(4, 8, 6)
        s3 = tool.Circ(12, 8, 6)
        s4 = tool.Rect(6, 8, 3, 6, 0)
        s5 = tool.Rect(10, 8, 3, 6, 0)
        s = s1 + (s2 - s4) + (s3 - s5)
        scipy.misc.imsave('demo/dumbell.png', s.to_np())
    dumbell()

    # double_wrench 
    def double_wrench():
        s1 = tool.Rect(8, 8, 1, 10, 3)
        s2 = tool.Circ(4, 4, 3)
        s3 = tool.Rect(3, 3, 1, 2, 3)
        s4 = tool.Circ(12, 12, 3)
        s5 = tool.Rect(12, 12, 1, 1, 1)
        s = s1 + (s2 - s3) + (s4 - s5)
        scipy.misc.imsave('demo/double_wrench.png', s.to_np())
    double_wrench()

    # mag 
    def mag():
        s1 = tool.Rect(11, 11, 1, 3, 3)
        s2 = tool.Circ(8, 8, 7)
        s3 = tool.Circ(8, 8, 5)
        s = s1 + (s2 - s3)
        scipy.misc.imsave('demo/mag.png', s.to_np())
    mag()

    # sickle 
    def sickle():
        s1 = tool.Rect(5, 11, 1, 3, 1)
        s2 = tool.Circ(8, 7, 8)
        s3 = tool.Circ(7, 7, 7)
        s = s1 + (s2 - s3)
        scipy.misc.imsave('demo/sickle.png', s.to_np())
    sickle()

    # comrad 
    def comrad():
        s1 = tool.Rect(5, 11, 1, 3, 1)
        s2 = tool.Circ(8, 7, 8)
        s3 = tool.Circ(7, 7, 7)
        ss = s1 + (s2 - s3)

        s4 = tool.Rect(10, 10, 1, 6, 3)
        s5 = tool.Rect(7, 7, 3, 2, 3)
        sh = s4 + s5
        
        s = ss + sh
        scipy.misc.imsave('demo/comrad.png', s.to_np())
    comrad()
