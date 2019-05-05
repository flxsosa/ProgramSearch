import tool_gen as tool
import scipy.misc

if __name__ == '__main__':
    # wrench 
    def wrench():
        s1 = tool.Rect(8, 9, 2, 8, 0)
        s2 = tool.Circ(8, 4, 5)
        s3 = tool.Rect(7, 3, 3, 2, 1)
        s = s1 + (s2 - s3)
        scipy.misc.imsave('drawings/tool_sets/wrench.png', s.to_np())
    wrench()

    # shovel 
    def shovel():
        s1 = tool.Rect(8, 7, 1, 8, 0)
        s2 = tool.Circ(8, 13, 3)
        s3 = tool.Rect(8, 3, 3, 1, 0)
        s4 = tool.Rect(8, 12, 3, 2, 0)
        s = s1 + s2 + s3 + s4
        scipy.misc.imsave('drawings/tool_sets/shovel.png', s.to_np())
    shovel()

    # key 
    def key():
        s1 = tool.Rect(7, 8, 1, 8, 2)
        s2 = tool.Circ(12, 8, 4)
        s3 = tool.Circ(12, 8, 2)
        s4 = tool.Rect(5, 7, 3, 2, 0)
        s5 = tool.Rect(5, 7, 1, 3, 0)
        s = s1 + (s2 - s3) + (s4 - s5)
        scipy.misc.imsave('drawings/tool_sets/key.png', s.to_np())
    key()

    # Ibeam 
    def Ibeam():
        s1 = tool.Rect(8, 8, 2, 8, 0)
        s2 = tool.Rect(8, 4, 5, 2, 0)
        s3 = tool.Rect(8, 12, 5, 2, 0)
        s = s1 + s2 + s3
        scipy.misc.imsave('drawings/tool_sets/Ibeam.png', s.to_np())
    Ibeam()

    # dumbell 
    def dumbell():
        s1 = tool.Rect(8, 8, 2, 8, 2)
        s2 = tool.Circ(4, 8, 6)
        s3 = tool.Circ(12, 8, 6)
        s4 = tool.Rect(6, 8, 3, 6, 0)
        s5 = tool.Rect(10, 8, 3, 6, 0)
        s = s1 + (s2 - s4) + (s3 - s5)
        scipy.misc.imsave('drawings/tool_sets/dumbell.png', s.to_np())
    dumbell()

    # double_wrench 
    def double_wrench():
        s1 = tool.Rect(8, 8, 1, 10, 3)
        s2 = tool.Circ(4, 4, 3)
        s3 = tool.Rect(3, 3, 1, 2, 3)
        s4 = tool.Circ(12, 12, 3)
        s5 = tool.Rect(12, 12, 1, 1, 1)
        s = s1 + (s2 - s3) + (s4 - s5)
        scipy.misc.imsave('drawings/tool_sets/double_wrench.png', s.to_np())
    double_wrench()

    # mag 
    def mag():
        s1 = tool.Rect(11, 11, 1, 3, 3)
        s2 = tool.Circ(8, 8, 7)
        s3 = tool.Circ(8, 8, 5)
        s = s1 + (s2 - s3)
        scipy.misc.imsave('drawings/tool_sets/mag.png', s.to_np())
    mag()

    # sickle 
    def sickle():
        s1 = tool.Rect(5, 11, 1, 3, 1)
        s2 = tool.Circ(8, 7, 8)
        s3 = tool.Circ(7, 7, 7)
        s = s1 + (s2 - s3)
        scipy.misc.imsave('drawings/tool_sets/sickle.png', s.to_np())
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
        scipy.misc.imsave('drawings/tool_sets/comrad.png', s.to_np())
    comrad()
