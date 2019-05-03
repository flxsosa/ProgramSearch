from CAD import *




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("mode", choices=["imitation","exit","test","demo","makeData","heatMap",
                                         "critic"])
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
    parser.add_argument("--hidden", "-H", type=int, default=512,
                        help="Size of hidden layers")
    parser.add_argument("--timeout", default=5, type=float,
                        help="Test time maximum timeout")
    parser.add_argument("--nudge", default=False, action='store_true')
    parser.add_argument("--oneParent", default=True, action='store_true')
    parser.add_argument("--noTranslate", default=True, action='store_true')
    parser.add_argument("--resume", default=False, action='store_true')
    
    arguments = parser.parse_args()
    arguments.translate = not arguments.noTranslate

    if arguments.mode == "demo":
        os.system("mkdir demo")
        if arguments.td:
            rs = lambda : randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes, nudge=arguments.nudge, translate=arguments.translate)
        else:
            rs = lambda : random3D(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes)
        startTime = time.time()
        ns = 50
        for _ in range(ns):
            rs().execute()
        print(f"{ns/(time.time() - startTime)} (renders + samples)/second")
        for n in range(100):
            s = rs()
            if arguments.td:
                plot.imshow(s.highresolution(256))
                plot.savefig(f"demo/CAD_{n}_hr.png")
            else:
                s.show(export=f"demo/CAD_{n}_3d.png")
                print(s)
            s.scad(f"demo/CAD_{n}_model.scad")

        import sys
        sys.exit(0)
        
            
    if arguments.checkpoint is None:
        arguments.checkpoint = f"checkpoints/{'2d' if arguments.td else '3d'}_{arguments.mode}"
        if arguments.viewpoints:
            arguments.checkpoint += "_viewpoints"
        if arguments.attention > 0:
            arguments.checkpoint += f"_attention{arguments.attention}_{arguments.heads}"
        arguments.checkpoint += ".pickle"
        print(f"Setting checkpointpath to {arguments.checkpoint}")
    if arguments.mode == "imitation":
        if not arguments.td:
            dsl = dsl_3d
            if not arguments.viewpoints:
                oe = CNN_3d(channels=2, channelsAsArguments=True,
                            hiddenChannels=32, outputChannels=32,
                            inputImageDimension=RESOLUTION,
                            layers=2)
                se = CNN_3d(channels=1, inputImageDimension=RESOLUTION,
                            hiddenChannels=32, outputChannels=32,
                            layers=2)
            else:
                oe = MultiviewObject()
                se = MultiviewSpec()
            training = lambda : random3D(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes)
        else:
            dsl = dsl_2d
            oe = ObjectEncoder()
            se = SpecEncoder()
            training = getTrainingData('CSG_data.p')

        print(f"CNN output dimensionalitys are {oe.outputDimensionality} & {se.outputDimensionality}")

        if arguments.resume:
            m = torch.load(arguments.checkpoint)
            print(f"Resuming checkpoint {arguments.checkpoint}")
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
        m = torch.load(arguments.checkpoint)
        critic = A2C(m)
        def R(spec, program):
            if len(program) == 0 or len(program) > len(spec.toTrace()): return False
            spec = spec.execute() > 0.5
            for o in program.objects():
                if np.all((o.execute() > 0.5) == spec): return True
            return False
        if arguments.td:
            training = lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes)
        else:
            training = lambda: random3D(maxShapes=arguments.maxShapes,minShapes=arguments.maxShapes)
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
            dataGenerator = getTrainingData('CSG_data.p')
        else:
            dataGenerator = lambda : random3D(maxShapes=arguments.maxShapes,
                                              minShapes=arguments.maxShapes)
        testCSG(m,
                dataGenerator,
                arguments.timeout,
                export=f"figures/CAD_{arguments.maxShapes}_shapes.png")
