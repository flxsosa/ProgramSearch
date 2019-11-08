#test

#from utilities import load_checkpoint
from driver import *




if __name__=='__main__':
    noEx_path = 'checkpoints/2d_imitation_noExecution_abstraction_2019-11-07T10:50:12.pickle'
    value_path = 'checkpoints/2d_critic_abstraction_2019-11-07T11:08:33.pickle'
    m = load_checkpoint(noEx_path)
    v = load_checkpoint(value_path)

    dataGenerator = lambda: randomScene(maxShapes=20, minShapes=20)

    objectEncodings = ScopeEncoding(v)

    for i in range(1):
        spec = dataGenerator()
        print(spec)

        B = []
        for ll, objects in m.beaming(spec, B=10, maximumLines=4,maximumTokens=100):
            B.append((ll, objects))
            print(ll, objects)

            objEncodings = objectEncodings.encoding(spec, objects)
            #print(objEncodings.shape)
            specEncodings = v.specEncoder(spec.execute()) #m.specEncoder(np.array([concrete_p2.execute()] ) )
            #print(specEncodings.shape)

            dist = v.distance(objEncodings, specEncodings)
            print("DISTANCE", dist.item())
            print()



