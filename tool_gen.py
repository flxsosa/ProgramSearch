import time
import random
from PIL import Image, ImageDraw
import cv2
import numpy as np
import scipy.misc

"""
tool -> handle head
head -> hammer | sissor | plier | screw-driver | shovel | file | 
handle -> stick | stick stick
stick -> bar end-piece
end-piece -> ball | ring | square
"""

# the basic unit in math where things are created
L = 16
# the scale factor, so we can actually see it in image
C = 4

# ========================= RENDERABLE PRIMITIVES =======================

class Shape:
    def is_in(self, xx, yy):
        raise NotImplementedError

    def to_np(self):
        ret = np.zeros((L*C, L*C))
        # yy and xx is flipped for image, as outer dimension is yy, inner is xx
        for yy in range(L * C):
            for xx in range(L * C):
                if self.is_in(xx, yy):
                    ret[yy][xx] = 1
        return ret

    def __add__(self, x):
        return Union(self, x)

    def __sub__(self, x):
        return Subtract(self, x)

class Circ(Shape):

    def __init__(self, x, y, r):
        self.x, self.y, self.r = x, y, r//2

    def is_in(self, xx, yy):
        # scale xx, yy back down to math space
        xx, yy = xx / C, yy / C
        # translate to center
        xx, yy = xx - self.x, yy - self.y
        return xx ** 2 + yy ** 2 <= self.r ** 2

class Rect(Shape):

    def __init__(self, x, y, w, h, a):
        try:
            wa = int(x+y+w+h+a) 
        except:
            print ("input something wrong . . . ")
            print (x,y,w,h,a)
            assert 0
        self.x, self.y, self.w, self.h, self.a = x, y, w, h, a

    def is_in(self, xx, yy):
        # scale xx, yy back down to math space
        xx, yy = xx / C, yy / C
        # translate to center
        xx, yy = xx - self.x, yy - self.y
        # rotate to allign
        if self.a == 0:
            Txx, Tyy = xx, yy
        if self.a == 1:
            Txx = np.dot(np.array([xx,yy]), np.array([np.sqrt(2)/2, np.sqrt(2)/2]))
            Tyy = np.dot(np.array([xx,yy]), np.array([np.sqrt(2)/2, -np.sqrt(2)/2]))
        if self.a == 2:
            Txx = yy
            Tyy = -xx
        if self.a == 3:
            Txx = np.dot(np.array([xx,yy]), np.array([-np.sqrt(2)/2, np.sqrt(2)/2]))
            Tyy = np.dot(np.array([xx,yy]), np.array([np.sqrt(2)/2, np.sqrt(2)/2]))

            
        in_x = -self.w / 2 <= Txx <= self.w / 2
        in_y = -self.h / 2 <= Tyy <= self.h / 2
        return in_x and in_y

class Union(Shape):

    def __init__(self, S1, S2):
        self.S1, self.S2 = S1, S2

    def is_in(self, xx, yy):
        return self.S1.is_in(xx, yy) or self.S2.is_in(xx, yy)

class Subtract(Shape):
    def __init__(self, S1, S2):
        self.S1, self.S2 = S1, S2

    def is_in(self, xx, yy):
        return self.S1.is_in(xx, yy) and (not self.S2.is_in(xx, yy))

# =========================== TOOL GENERATION ============================
def gen_tool():
    loc_choices = range(L//4, 3*L//4)
    tool_x, tool_y = random.choice(loc_choices), random.choice(loc_choices)
    # handle info
    w = random.choice([1,2])
    h = w + random.choice([4,5,6,7,8])
    a = random.choice([0,1,2,3])

    head1_loc, head2_loc =  get_attached((tool_x, tool_y), h, a)
    stick = Rect(tool_x, tool_y,w,h,a)
    head1 = gen_head(jiggle(head1_loc))
    head2 = gen_head(jiggle(head2_loc))
    return stick + head1 + head2

# add in bit variability to attachments
def jiggle(loc):
    loc_x, loc_y = loc
    return loc_x + random.choice([-1, 0, 1]), loc_y + random.choice([-1, 0, 1])

def get_attached(loc, h, a):
    loc_x, loc_y = loc
    if a == 0:
        xy1 = loc_x, loc_y - h / 2
        xy2 = loc_x, loc_y + h / 2
    if a == 1:
        xy1 = loc_x + h / 2, loc_y - h / 2
        xy2 = loc_x - h / 2, loc_y + h / 2
    if a == 2:
        xy1 = loc_x + h / 2, loc_y
        xy2 = loc_x - h / 2, loc_y
    if a == 3:
        xy1 = loc_x + h / 2, loc_y + h / 2
        xy2 = loc_x - h / 2, loc_y - h / 2
    xy1 = (int(xy1[0]), int(xy1[1]))
    xy2 = (int(xy2[0]), int(xy2[1]))
    return random.choice([(xy1, xy2),(xy2,xy1)])

def gen_head(loc):
    possible_heads = [gen_hammer,
                      gen_plier,
                      gen_sickle,
            #                      gen_shovel,
            #                      gen_file,
                      ]
    return random.choice(possible_heads)(loc)

def gen_hammer(loc):
    head_sizes = [4, 5, 6]
    head_orients = [0, 1, 2, 3]
    return Rect(loc[0], loc[1], 
                random.choice(head_sizes), random.choice(head_sizes),
                random.choice(head_orients))

def gen_plier(loc):
    def get_head():
        if random.random() < 0.5:
            return gen_hammer(loc)
        else:
            return Circ(loc[0], loc[1], random.choice([4,5,6]))
    def get_sub():
        head_sizes = [1,2,3]
        head_orients = [0, 1, 2, 3]
        sub_loc = jiggle(loc)
        return Rect(sub_loc[0], sub_loc[1],
                    random.choice(head_sizes), random.choice(head_sizes),
                    random.choice(head_orients))
    return get_head() - get_sub()

def gen_sickle(loc):
    W_big = random.choice([4,5,6,7])
    W_small = W_big - random.choice([1,2,3])
    head = Circ(loc[0], loc[1], W_big)
    sub_loc = jiggle(loc)
    sub = Circ(sub_loc[0], sub_loc[1], W_small)
    return head - sub

# =========================== TEST AND MAIN =========================
def test_primitives():
    s1 = Rect(6, 6, 2, 8, 0)
    s2 = Rect(7, 7, 2, 8, 1)
    s3 = Rect(8, 8, 2, 8, 2)
    s4 = Rect(9, 9, 2, 8, 3)
    s5 = Circ(8, 8, 2)
    s = (s1 + s2 + s3 + s4) - s5
    scipy.misc.imsave('drawings/outfile.png', s.to_np())

if __name__ == '__main__':
    # test_primitives()
    s = gen_tool()
    scipy.misc.imsave('drawings/outfile.png', s.to_np())


