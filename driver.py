from CAD import *
from tool_set import *
import matplotlib.pyplot as plot

from SMC import instrumentSMC

import numpy as np
import torch
import random

from datetime import datetime



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("mode", choices=["imitation","exit","test","demo","makeData","heatMap",
                                         "critic","render"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--maxShapes", default=20,
                            type=int)
    parser.add_argument("--2d", default=False, action='store_true', dest='td')
    parser.add_argument("--viewpoints", default=False, action='store_true', dest='viewpoints')
    parser.add_argument("--trainTime", default=None, type=float,
                        help="Time in hours to train the network")
    parser.add_argument("--attention", default=0, type=int,
                        help="Number of rounds of self attention to perform upon objects in scope")
    parser.add_argument("--heads", default=2, type=int,
                        help="Number of attention heads")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed")
    parser.add_argument("--ntest", default=30, type=int,
                        help="size of testing set")
    parser.add_argument("--hidden", "-H", type=int, default=512,
                        help="Size of hidden layers")
    parser.add_argument("--timeout", default=5, type=float,
                        help="Test time maximum timeout")
    parser.add_argument("--nudge", default=False, action='store_true')
    parser.add_argument("--oneParent", default=True, action='store_true')
    parser.add_argument("--noTranslate", default=True, action='store_true')
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--render", default=[],type=str,nargs='+')
    parser.add_argument("--tools", default=False, action='store_true')
    parser.add_argument("--noExecution", default=False, action='store_true')
    parser.add_argument("--rotate", default=False, action='store_true')
    parser.add_argument("--solvers",default=["fs"],nargs='+')

    timestamp = datetime.now().strftime('%FT%T')
    print(f"Invoking @ {timestamp} as:\n\tpython {' '.join(sys.argv)}")
    
    arguments = parser.parse_args()
    arguments.translate = not arguments.noTranslate

    if arguments.render:
        for path in arguments.render:
            if path.endswith(".pickle") or path.endswith(".p"):
                with open(path,"rb") as handle:
                    program = pickle.load(handle)
                    print(f"LOADED {path}")
                    print(ProgramGraph.fromRoot(program).prettyPrint(True))
                    program.scad("/tmp/render.scad")
            elif path.endswith(".scad"):
                os.system(f"cp {path} /tmp/render.scad")
            else: assert False, f"unknown file extension {path}"
            

            os.chdir("render_scad_tool")
            os.system("python render_scad.py /tmp/render.scad")
            os.chdir("..")
            os.system(f"cp render_scad_tool/example.png {path}_pretty.png")
        sys.exit(0)


    if arguments.mode == "demo":
        os.system("mkdir demo")
        if arguments.td:
            rs = lambda : randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes, nudge=arguments.nudge, translate=arguments.translate)
        else:
            rs = lambda : random3D(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes,
                                   rotate=arguments.rotate)
        startTime = time.time()
        ns = 50
        for _ in range(ns):
            rs().execute()
        print(f"{ns/(time.time() - startTime)} (renders + samples)/second")
        if arguments.tools:
            if not arguments.td: rs = make3DTools()
            else: rs = make2DTools()
        else:
            rs = [rs() for _ in range(10) ]
            
        for n,s in enumerate(rs):
            if arguments.td:
                s.export(f"demo/CAD_{n}_hr.png",256)
                s.exportDecomposition(f"demo/CAD_{n}_trace.png",256)
                plot.imshow(s.execute())
                plot.savefig(f"demo/CAD_{n}_lr.png")
                print(s)
            else:
                s.show(export=f"demo/CAD_{n}_3d.png")
                print(s)
                s.scad(f"demo/CAD_{n}_model.png")

        import sys
        sys.exit(0)
        
            
    if arguments.checkpoint is None:
        arguments.checkpoint = f"checkpoints/{'2d' if arguments.td else '3d'}_{arguments.mode}"
        if arguments.noExecution:
            arguments.checkpoint += "_noExecution"
        if arguments.viewpoints:
            arguments.checkpoint += "_viewpoints"
        if arguments.attention > 0:
            arguments.checkpoint += f"_attention{arguments.attention}_{arguments.heads}"
        if not arguments.td:
            if not arguments.rotate:
                arguments.checkpoint += "_noRotate"
        arguments.checkpoint += f"_{timestamp}.pickle"
        print(f"Setting checkpointpath to {arguments.checkpoint}")
    if arguments.mode == "imitation":
        if not arguments.td:
            dsl = dsl_3d
            if not arguments.viewpoints:
                oe = CNN_3d(channels=2, channelsAsArguments=True,
                            inputImageDimension=RESOLUTION,
                            filterSizes=[3,3,3],
                            poolSizes=[4,1,1],
                            numberOfFilters=[32,32,16])
                se = CNN_3d(channels=1, inputImageDimension=RESOLUTION,
                            filterSizes=[3,3,3],
                            poolSizes=[4,1,1],
                            numberOfFilters=[32,32,16])
            else:
                oe = MultiviewObject()
                se = MultiviewSpec()
            training = lambda : random3D(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes,
                                         rotate=arguments.rotate)
        else:
            dsl = dsl_2d
            oe = ObjectEncoder()
            se = SpecEncoder()
            training = lambda : randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes)

        print(f"CNN output dimensionalitys are {oe.outputDimensionality} & {se.outputDimensionality}")

        if arguments.resume:
            m = torch.load(arguments.resume)
            print(f"Resuming checkpoint {arguments.resume}")
        else:
            if arguments.noExecution:
                m = NoExecution(se,dsl)
            else:
                m = ProgramPointerNetwork(oe, se, dsl,
                                          oneParent=arguments.oneParent,
                                          attentionRounds=arguments.attention,
                                          heads=arguments.heads,
                                          H=arguments.hidden)
        trainCSG(m, training,
                 trainTime=arguments.trainTime*60*60 if arguments.trainTime else None,
                 checkpoint=arguments.checkpoint)
    elif arguments.mode == "critic":
        assert arguments.resume is not None, "You need to specify a checkpoint with --resume, which bootstraps the policy"
        m = torch.load(arguments.resume)
        critic = A2C(m)
        def R(spec, program):
            if len(program) == 0 or len(program) > len(spec.toTrace()): return False
            for o in program.objects():
                if o.IoU(spec) > 0.95: return True
            return False
        if arguments.td:
            training = lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes)
        else:
            training = lambda: random3D(maxShapes=arguments.maxShapes,minShapes=arguments.maxShapes,
                                        rotate=arguments.rotate)
        critic.train(arguments.checkpoint,
                     training,
                     R)
        
    elif arguments.mode == "heatMap":
        learnHeatMap()
    elif arguments.mode == "makeData":
        makeTrainingData()
    elif arguments.mode == "exit":
        with open(arguments.checkpoint,"rb") as handle:
            m = pickle.load(handle)
        searchAlgorithm = BeamSearch(m, maximumLength=arguments.maxShapes*3 + 1)
        loss = lambda spec, program: 1-max( o.IoU(spec) for o in program.objects() ) if len(program) > 0 else 1.
        searchAlgorithm.train(getTrainingData('CSG_data.p'),
                              loss=loss,
                              policyOracle=lambda spec: spec.toTrace(),
                              timeout=1,
                              exitIterations=-1)
    elif arguments.mode == "test":
        m = load_checkpoint(arguments.checkpoint)
        if arguments.td:
            if arguments.tools:
                dataGenerator = make2DTools()
            else:
                dataGenerator = lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes)
        else:
            if arguments.tools:
                dataGenerator = make3DTools()
            else:
                dataGenerator = lambda : random3D(maxShapes=arguments.maxShapes,
                                                  minShapes=arguments.maxShapes,
                                                  rotate=arguments.rotate)
        if arguments.tools:
            suffix = "tools"
        else:
            suffix = f"{arguments.maxShapes}_shapes"
        testCSG(m,
                dataGenerator,
                arguments.timeout,
                solvers=arguments.solvers,
                timestamp=timestamp,
                solverSeed=arguments.seed,
                n_test=arguments.ntest)
