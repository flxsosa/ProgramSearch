from CAD import *


def randomSceneSubTriangle(resolution=32, maxShapes=3, minShapes=1, verbose=False, export=None,
                nudge=False, translate=False, loops=False):
    assert not translate
    assert not nudge
    assert not loops
    import matplotlib.pyplot as plot

    dc = 16 # number of distinct coordinates
    choices = [c
               for c in range(0, resolution, resolution//dc) ]
    def slantedQuadrilateral(square=False):
        while True:
            x0 = random.choice(choices)
            y0 = random.choice(choices)
            a = [a for a in range(resolution)
                 if x0 - a in choices and y0 + a in choices and a > 1]
            if len(a) == 0: continue
            a = random.choice(a)
            if not square:
                b = [b for b in range(resolution)
                     if x0 + b in choices and y0 + b in choices and x0 - a + b in choices and y0 + a + b in choices and b > 1]
            else:
                b = [b for b in range(resolution)
                     if x0 + b in choices and y0 + b in choices and x0 - a + b in choices and y0 + a + b in choices and b > 1 and b == a]
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
        q = slantedQuadrilateral(square=True)
        # facing upward
        if random.random() < 0.25:
            x0 = q.x1
            y0 = q.y0
            x1 = q.x3
            y1 = q.y1
            p = Rectangle(x0,y0,
                          x0,y1,
                          x1,y1,
                          x1,y0)
        elif random.random() < 0.33: # facing downward
            x0 = q.x1
            y0 = q.y1
            x1 = q.x3
            y1 = q.y2
            p = Rectangle(x0,y0,
                          x0,y1,
                          x1,y1,
                          x1,y0)
        elif random.random() < 0.5: # facing rightward
            x0 = q.x1
            y0 = q.y0
            x1 = q.x2
            y1 = q.y2
            p = Rectangle(x0,y0,
                          x0,y1,
                          x1,y1,
                          x1,y0)
        else: # facing left
            x0 = q.x0
            y0 = q.y0
            x1 = q.x3
            y1 = q.y2
            p = Rectangle(x0,y0,
                          x0,y1,
                          x1,y1,
                          x1,y0)
        return q - p


    while True:
        s = None
        numberOfShapes = 0
        desiredShapes = random.choice(range(minShapes, 1 + maxShapes))

        subtractedTriangle = False
        is_triangle = False

        for _ in range(desiredShapes):
            
            if random.random() < 0.1:
                o = triangle()
                is_triangle = True
            else:
                o = quadrilateral() if random.choice([True,False]) else circular()
                is_triangle = False
            if s is None:
                s = o
            else:
                if (is_triangle and subtractedTriangle):# or random.choice([True,True,False]):    
                    new = s + o
                else:
                    new = s - o
                    if is_triangle:
                        subtractedTriangle = True

                # Change at least ten percent of the pixels
                oldOn = s.render().sum()
                newOn = new.render().sum()
                pc = abs(oldOn - newOn)/oldOn
                if pc < 0.1 or pc > 0.6:
                    continue
                s = new
        try:
            if not subtractedTriangle: continue
            #import pdb; pdb.set_trace()
            print(s.abstract())
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


if __name__=='__main__':
    s = randomSceneSubTriangle(maxShapes=30, minShapes=30) 
    print(s.abstract())
    print(ProgramGraph.fromRoot(s.abstract(), oneParent=True).prettyPrint())