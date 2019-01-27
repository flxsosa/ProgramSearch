from utilities import *
import torch.nn.functional as F

class MultiHeadAttention(Module):
    def __init__(self, heads, entity_dimensionality, rounds=1, residual=True,
                 layers=2):
        super().__init__()
        self.entity_dimensionality = entity_dimensionality
        self.heads = heads

        assert entity_dimensionality%heads == 0,\
        "dimensionality of entities must be divisible by number of heads"

        # Dimensionality of each head
        self.d = entity_dimensionality//heads

        self.Q = nn.Linear(entity_dimensionality, entity_dimensionality)
        self.V = nn.Linear(entity_dimensionality, entity_dimensionality)
        self.K = nn.Linear(entity_dimensionality, entity_dimensionality)
        self.output = nn.Sequential(*[ layer
                                       for _ in range(layers)
                                       for layer in [nn.Linear(entity_dimensionality, entity_dimensionality),
                                                     nn.ReLU()]])

        self.rounds = rounds
        self.residual = residual

        self.finalize()

    def forward(self, entities):
        """
        entities: (# entities)x(entity_dimensionality)
        returns: (# entities)x(entity_dimensionality)
        """
        for _ in range(self.rounds):
            # query, values, and keys should all be of size HxExD
            q = self.Q(entities).view(entities.size(0), self.heads, self.d).permute(1,0,2)
            v = self.V(entities).view(entities.size(0), self.heads, self.d).permute(1,0,2)
            k = self.K(entities).view(entities.size(0), self.heads, self.d).permute(1,0,2)

            # attention[i,j] = q_i . k_j
            # i.e., amount that object I is attending to object J
            attention = F.softmax(q@(k.permute(0,2,1))/(self.d**0.5), dim=-1)

            # Mix together values
            o = (attention@v).transpose(0,1).contiguous().view(entities.size(0), self.entity_dimensionality)
            
            # Apply output transformation
            o = self.output(o)

            # residual connection
            if self.residual:
                entities = entities + o
            else:
                entities = o
        
        return entities

