from utilities import *
from CNN import *

from torch.nn.modules.container import OrderedDict
import torch
import torch.nn as nn

class HeatEncoder(Module):
    def __init__(self, input_channels, hidden_channels, output_channels,
                 layers=3):
        super(HeatEncoder, self).__init__()

        self.model = nn.Sequential(
            # preprocess
            nn.Sequential(nn.Conv3d(input_channels, hidden_channels, 3, padding=1), nn.ReLU()),
            # Highway
            nn.Sequential(*[ResidualCNNBlock_3d(hidden_channels)
                                       for _ in range(layers - 2) ]),
            # post process
            nn.Conv3d(hidden_channels, output_channels, 3, padding=1)
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
    def __init__(self, resolution=32,maxObjects=5):
        super(HeatNetwork, self).__init__()

        self.maxObjects = maxObjects
        self.resolution = resolution

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
        
        self.encoder = HeatEncoder(input_channels=self.inputChannels,
                                   hidden_channels=32,
                                   output_channels=self.outputChannels)
        self.finalize()

    def initialInput(self, spec, objects, line):
        x0 = np.zeros((1 + self.maxObjects, self.resolution, self.resolution, self.resolution))
        x0[0] = spec.execute()
        for n,o in enumerate(objects): x0[n + 1] = o.execute()
        x = np.zeros((self.inputChannels, self.resolution, self.resolution, self.resolution))
        x[:x0.shape[0],:,:,:] = x0
        return x

    def line2io(self, spec, objects, line):
        """Returns [inputArrays], [(head, *index)]"""
        x = self.initialInput(spec, objects, line)
        
        # inputSequence: list of CNN inputs
        inputSequence = [x]
        # outputSequence: for each CNN input, list of (head, *indices)
        outputSequence = []

        line = line.serialize()

        if line[0] == 'cuboid':
            outputSequence.append([(self.shapePredictor, self.shape2index[line[0]],
                                    line[1], line[2], line[3])])
            outputSequence.append([(self.cuboid2, 0,
                                    line[4], line[5], line[6])])
            x1 = np.copy(x)
            x1[self.inputHeat["cuboid1"],line[1],line[2],line[3]] = 1.
            inputSequence.append(x1)
        if line[0] == 'cylinder':
            outputSequence.append([(self.shapePredictor, self.shape2index[line[0]],
                                    line[2], line[3], line[4])])
            outputSequence.append([(self.cylinder2, 0,
                                    line[5], line[6], line[7]),
                                   (self.cylindricalRadius,
                                    self.cylindricalRadii.index(line[1]), line[5], line[6], line[7])])
            x1 = np.copy(x)
            x1[self.inputHeat["cylinder1"],line[2],line[3],line[4]] = 1.
            inputSequence.append(x1)
        if line[0] == 'sphere':
            outputSequence.append([(self.shapePredictor, self.shape2index[line[0]],
                                    line[1], line[2], line[3]),
                                   (self.sphericalRadius, self.sphericalRadii.index(line[4]),
                                   line[1], line[2], line[3])])
            
        assert len(inputSequence) == len(outputSequence)

        return inputSequence, outputSequence

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
        x = self.initialInput(spec, objects, line)
        heat = self.encoder(self.tensor(x).unsqueeze(0))
        objectPrediction = self.shapePredictor(heat).squeeze(0).contiguous().view(self.resolution*self.resolution*self.resolution*len(self.shape2index)).exp()
        i = torch.multinomial(objectPrediction,1).data.item()

        def index2position(index):
            V = self.resolution*self.resolution*self.resolution
            c = index//V
            index = index%V
            x = index//(self.resolution**2)
            index = index%(self.resolution**2)
            y = index//(self.resolution)
            index = index%(self.resolution)
            z = index
            return c,x,y,z

        shape,x,y,z = index2position(index)
        shape = self.shape2index[shape]
        if shape == "sphere":
            radiusPrediction = self.sphericalRadius(heat).squeeze(0)[:,x,y,z].exp()
            r = torch.multinomial(radiusProduction,1).data.item()
            r = self.sphericalRadii[r]
            return Sphere(x,y,z,r)
        return shape
            
            
            

        
            
            
                                    
                    

        
if __name__ == "__main__":
    from CAD import *
    m = HeatNetwork(32)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    while True:
        def randomShape():
            return random3D(maxShapes=1,minShapes=1)
        def randomTriple():
            s = randomShape()
            return (s,[],s)
        triples = [randomTriple() for _ in range(3) ]

        m.zero_grad()
        batched = m.batchedLogLikelihood(triples)
        (-batched).backward()
        optimizer.step()
        L = -batched.data.item()
        print(L)
        if L < 3.:
            for spec, objects,_ in triples:
                print(spec, objects, m.sample(spec, objects))

        
            
    m.batchedLogLikelihood([(Sphere(1,1,1,4),[Sphere(1,1,1,4)],
                             Cuboid(1,2,1,2,2,2))])
