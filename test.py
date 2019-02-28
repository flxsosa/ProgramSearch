from randomSolver import RandomSolver
from CAD import dsl, randomScene, CSG

solver = RandomSolver(dsl)
solver._infer(randomScene, CSG.IoU, 3)