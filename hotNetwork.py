from utilities import *
from CNN import *

from torch.nn.modules.container import OrderedDict
import torch
import torch.nn as nn

class HeatEncoder(Module):
    def __init__(self, input_channels, hidden_channels, output_channels,
                 downsample=1,
                 layers=3):
        super(HeatEncoder, self).__init__()

        if downsample == 1:
            preprocess = nn.Sequential(nn.Conv3d(input_channels, hidden_channels, 3, padding=1), nn.ReLU())
        else:
            preprocess = nn.Sequential(nn.Conv3d(input_channels, hidden_channels, 3, padding=1), nn.ReLU(),
                                       nn.MaxPool3d(downsample))

        self.model = nn.Sequential(
            # preprocess
            preprocess,
            # Highway
            nn.Sequential(*[ResidualCNNBlock_3d(hidden_channels)
                                       for _ in range(layers - 2) ]),
            # post process
            nn.Conv3d(hidden_channels, output_channels, 3, padding=1),
            nn.ReLU()
        )
        self.finalize()

    def forward(self, x):
        return self.model(x)

class HeatLogSoftmax(Module):
    def __init__(self):
        super(HeatLogSoftmax, self).__init__()

        self.finalize()

    def forward(self, x):
        """x: either [b,c,voxel] or [c,voxel]"""
        noBatch = len(x.shape) == 4
        if noBatch: x = x.unsqueeze(0)

        maximum = x
        for _ in range(4):
            maximum = maximum.max(-1)[0]

        # for broadcasting
        maximumb = maximum.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        lz = (x - maximumb).exp().sum(-1).sum(-1).sum(-1).sum(-1).log() + maximum
        x = x - lz.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        if noBatch: x = x.squeeze(0)
        return x
        
class HeatNetwork(Module):
    def __init__(self, resolution=32, hotResolution=16, maxObjects=4):
        super(HeatNetwork, self).__init__()

        self.maxObjects = maxObjects
        self.resolution = resolution
        self.downsample = resolution//hotResolution
        self.hotResolution = hotResolution

        self.heatMaps = ["cuboid1","cuboid2"] + \
                        ["spherical"] + \
                        ["cylinder1","cylinder2"]
        
        self.heatMaps = dict(zip(self.heatMaps, range(len(self.heatMaps))))
        self.inputChannels = 1 + \
                             maxObjects + \
                             len(self.heatMaps) + 1

        self.shape2index = {"cuboid": 0,
                            "cylinder": 1,
                            "sphere": 2}
        self.index2shape = {v:k for k,v in self.shape2index.items() }

        self.inputHeat = {"cuboid1": 1 + maxObjects,
                          "cylinder1": 2 + maxObjects}

        self.sphericalRadii = [4,8,12]
        self.cylindricalRadii = [4,8,12]

        self.outputChannels = 16

        self.shapePredictor = nn.Sequential(OrderedDict([
            ("shape_conv",
             nn.Conv3d(self.outputChannels,
                       3, # cube1/cylinder1/sphere
                       1)), # 1x1x1 convolution
            ("shape_sm",HeatLogSoftmax())]))
        self.sphericalRadius = nn.Sequential(OrderedDict([
            ("sphere_radius_conv",
             nn.Conv3d(self.outputChannels,
                       len(self.sphericalRadii),
                       1)),
            ("sphere_radius_sm",
             nn.LogSoftmax(dim=1))])) # soft max over channel dimension
        self.cylindricalRadius = nn.Sequential(OrderedDict([
            ("cylinder_radius_conv",
             nn.Conv3d(self.outputChannels,
                       len(self.cylindricalRadii),
                       1)),
             ("cylinder_radius_sm",
              nn.LogSoftmax(dim=1))]))
        self.cuboid2 = nn.Sequential(OrderedDict([
            ("cuboid2_conv",
             nn.Conv3d(self.outputChannels,
                       1, # cube2
                       1)), # 1x1x1 convolution
            ("cuboid2_hsm",
             HeatLogSoftmax())]))
        self.cylinder2 = nn.Sequential(OrderedDict([
            ("cylinder2_conv",
             nn.Conv3d(self.outputChannels,
                       1, # cylinder2
                       1)), # 1x1x1 convolution
            ("cylinder2_hsm",HeatLogSoftmax())]))

        commands = ["STOP","DRAW"]
        for n1 in range(maxObjects):
            for n2 in range(maxObjects):
                if n1 < n2: commands.append(("UNION",n1,n2))
                if n1 != n2: commands.append(("DIFFERENCE",n1,n2))
        self.command2index = {k: n for n,k in enumerate(commands) }
        self.index2command = {v:k for k,v in self.command2index.items() }
        self.command = nn.Sequential(Flatten(),
                                     nn.Linear(self.outputChannels*(hotResolution*hotResolution*hotResolution),
                                               len(self.command2index)),
                                     nn.LogSoftmax(dim=-1))
        self._value = nn.Sequential(Flatten(),
                                    nn.Linear(self.outputChannels*(hotResolution*hotResolution*hotResolution),
                                              256),
                                    nn.ReLU(),
                                    nn.Linear(256,1),
                                    nn.Softplus(),
                                    NegationModule())
        
        self.encoder = HeatEncoder(input_channels=self.inputChannels,
                                   downsample=resolution//hotResolution,
                                   hidden_channels=32,
                                   layers=4,
                                   output_channels=self.outputChannels)
        self.finalize()

    def initialInput(self, spec, objects):
        x0 = np.zeros((1 + self.maxObjects, self.resolution, self.resolution, self.resolution))
        x0[0] = spec.execute() if isinstance(spec,CSG) else spec
        for n,o in enumerate(objects): x0[n + 1] = o.execute()
        x = np.zeros((self.inputChannels, self.resolution, self.resolution, self.resolution))
        x[:x0.shape[0],:,:,:] = x0
        return x

    def line2io(self, spec, objects, line):
        """Returns [inputArrays], [(head, *index)]"""
        x = self.initialInput(spec, objects)
        
        # inputSequence: list of CNN inputs
        inputSequence = [x]
        # outputSequence: for each CNN input, list of (head, *indices)
        outputSequence = []

        if line is None:
            outputSequence.append([(self.command, self.command2index["STOP"])])
            return inputSequence, outputSequence
        
        line = line.serialize()

        if line[0] == '+':
            n1 = objects.index(line[1])
            n2 = objects.index(line[2])
            n1,n2 = min(n1,n2),max(n1,n2)
            outputSequence.append([(self.command, self.command2index[("UNION",n1,n2)])])
        if line[0] == '-':
            n1 = objects.index(line[1])
            n2 = objects.index(line[2])
            outputSequence.append([(self.command, self.command2index[("DIFFERENCE",n1,n2)])])

        if line[0] == 'cuboid':
            outputSequence.append([(self.command, self.command2index["DRAW"]),
                                   (self.shapePredictor, self.shape2index[line[0]],
                                    line[1]//self.downsample,
                                    line[2]//self.downsample,
                                    line[3]//self.downsample)])
            outputSequence.append([(self.cuboid2, 0,
                                    line[4]//self.downsample,
                                    line[5]//self.downsample,
                                    line[6]//self.downsample)])
            x1 = np.copy(x)
            x1[self.inputHeat["cuboid1"],line[1],line[2],line[3]] = 1.
            inputSequence.append(x1)
        if line[0] == 'cylinder':
            outputSequence.append([(self.command, self.command2index["DRAW"]),
                                   (self.shapePredictor, self.shape2index[line[0]],
                                    line[2]//self.downsample,
                                    line[3]//self.downsample,
                                    line[4]//self.downsample)])
            outputSequence.append([(self.cylinder2, 0,
                                    line[5]//self.downsample,
                                    line[6]//self.downsample,
                                    line[7]//self.downsample),
                                   (self.cylindricalRadius,
                                    self.cylindricalRadii.index(line[1]),
                                    line[5]//self.downsample,
                                    line[6]//self.downsample,
                                    line[7]//self.downsample)])
            x1 = np.copy(x)
            x1[self.inputHeat["cylinder1"],line[2],line[3],line[4]] = 1.
            inputSequence.append(x1)
        if line[0] == 'sphere':
            outputSequence.append([(self.command, self.command2index["DRAW"]),
                                   (self.shapePredictor, self.shape2index[line[0]],
                                    line[1]//self.downsample,
                                    line[2]//self.downsample,
                                    line[3]//self.downsample),
                                   (self.sphericalRadius, self.sphericalRadii.index(line[4]),
                                    line[1]//self.downsample,
                                    line[2]//self.downsample,
                                    line[3]//self.downsample)])
            
        assert len(inputSequence) == len(outputSequence)

        return inputSequence, outputSequence

    def batchedProgramLikelihood(self, batch):
        """batch: [program]"""
        specs = batch
        triples = []
        for p in batch:
            objects = [] # stuff in scope
            for action in p.toTrace():
                triples.append((p,objects,action))
                objects = [o for o in objects if o not in action.children() ] + [action]
            triples.append((p,objects,None))
        for t in triples: print(t)
        return self.batchedLogLikelihood(triples)
        

    def batchedLogLikelihood(self, batch):
        """batch: [(spec, objects, line)]"""

        examples = [self.line2io(*arguments) for arguments in batch]
        inputs = [i_
                  for i,o in examples
                  for i_ in i ]
        outputs = [o_
                   for i,o in examples
                   for o_ in o ]

        inputs = np.stack(inputs)
        heats = self.encoder(self.tensor(inputs))

        L = 0
        pd = PointerDictionary()
        for b,yhs in enumerate(outputs):
            for yh in yhs:
                head = yh[0]
                index = yh[1:]

                if head in pd:
                    pd[head].append((b,index))
                else:
                    pd[head] = [(b,index)]
        
        for head,indices in pd.d:
            bs = [b for b,_ in indices]
            yh = head(heats[bs,...])
            for n,(_,index) in enumerate(indices):
                index = tuple([n] + list(index))
                L += yh[index]

        return L

    def logLikelihood(self, spec, objects, line):
        inputs, outputs = self.line2io(spec, objects, line)

        L = 0
        for i,os in zip(inputs, outputs):
            heat = self.encoder(self.tensor(i).unsqueeze(0))
            for head, *index in os:
                yh = head(heat).squeeze(0)
                L += yh[tuple(index)]
        return L

    def sample(self, spec, objects):
        x0 = self.initialInput(spec, objects)
        heat = self.encoder(self.tensor(x0).unsqueeze(0))
        command = self.index2command[torch.multinomial(self.command(heat).squeeze(0).exp(),1).data.item()]
        if command == "STOP": return None
        if command[0] == "UNION":
            try:
                return Union(objects[command[1]], objects[command[2]])
            except: return None
        if command[0] == "DIFFERENCE":
            try:
                return Difference(objects[command[1]], objects[command[2]])
            except: return None
        
        
        objectPrediction = self.shapePredictor(heat).squeeze(0).contiguous().view(self.hotResolution*self.hotResolution*self.hotResolution*len(self.shape2index)).exp()
        i = torch.multinomial(objectPrediction,1).data.item()

        def index2position(index):
            V = self.hotResolution*self.hotResolution*self.hotResolution
            c = index//V
            index = index%V
            x = index//(self.hotResolution**2)
            index = index%(self.hotResolution**2)
            y = index//(self.hotResolution)
            index = index%(self.hotResolution)
            z = index
            return c,x*self.downsample,y*self.downsample,z*self.downsample

        shape,x,y,z = index2position(i)
        shape = self.index2shape[shape]
        if shape == "sphere":
            r = self.sphericalRadius(heat).squeeze(0)[:,
                                                      x//self.downsample,
                                                      y//self.downsample,
                                                      z//self.downsample].exp()
            r = torch.multinomial(r,1).data.item()
            r = self.sphericalRadii[r]
            return Sphere(x,y,z,r)
        if shape == "cuboid":
            x1 = np.copy(x0)
            x1[self.inputHeat["cuboid1"],x,y,z] = 1.
            heat = self.cuboid2(self.encoder(self.tensor(x1).unsqueeze(0))).squeeze(0).squeeze(0)
            i = torch.multinomial(heat.exp().view(-1),1).data.item()
            _,x1,y1,z1 = index2position(i)
            return Cuboid(x,y,z,
                          x1,y1,z1)
        if shape == "cylinder":
            x1 = np.copy(x0)
            x1[self.inputHeat["cylinder1"],x,y,z] = 1.
            heat = self.encoder(self.tensor(x1).unsqueeze(0))
            r = self.cylindricalRadius(heat).squeeze(0)[:,
                                                      x//self.downsample,
                                                      y//self.downsample,
                                                      z//self.downsample].exp()
            r = torch.multinomial(r,1).data.item()
            r = self.cylindricalRadii[r]

            i = torch.multinomial(self.cylinder2(heat).exp().view(-1),1).data.item()
            _,x1,y1,z1 = index2position(i)
            return Cylinder(r,
                            x,y,z,
                            x1,y1,z1)
        return shape

    def rollout(self, spec, maxMoves=None):
        objects = []
        moves = 0
        while maxMoves is None or moves < maxMoves:
            moves += 1

            action = self.sample(spec, objects)
            if action is None: return objects
            objects = [o for o in objects if o not in action.children() ] + [action]
            if len(objects) > self.maxObjects: break
            
        return objects            
            
                                    
                    

        
if __name__ == "__main__":
    from CAD import *
    m = HeatNetwork(32,hotResolution=8)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    losses = []
    while True:
        def randomShape():
            return random3D(maxShapes=1,minShapes=1)
        def randomTriple():
            s = randomShape()
            return (s,[],s)
        programs = [random3D(maxShapes=4,minShapes=1) for _ in range(8) ]

        m.zero_grad()
        batched = m.batchedProgramLikelihood(programs)/sum(1 + len(p.toTrace()) for p in programs )
        (-batched).backward()
        optimizer.step()
        L = -batched.data.item()
        losses.append(L)
        if len(losses) > 100:
            print("Average loss:",sum(losses)/len(losses))
            losses = []
            for p in programs:
                print(p)
                print(m.rollout(p,maxMoves=len(p.toTrace()) + 1))
