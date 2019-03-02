"""
ExIt-style training
https://davidbarber.github.io/blog/2017/11/07/Learning-From-Scratch-by-Thinking-Fast-and-Slow-with-Deep-Learning-and-Tree-Search/
"""

from API import *

class ExitSolver(Solver):
    """Abstract class for solvers supporting Exit-style training"""

    def _report(self, program, trajectory):
        l = self.loss(program)
        if len(self.reportedSolutions) == 0 or self.reportedSolutions[-1].loss > l:
            self.reportedSolutions.append(SearchResult(program, l, time.time() - self.startTime))
            self.bestTrajectory = trajectory

    def sampleTrainingTrajectory(self, spec, loss, timeout):
        self.bestTrajectory = None
        self.infer(spec, loss, timeout)
        trajectory = self.bestTrajectory
        self.bestTrajectory = None
        return trajectory

    def train(self, getSpec, loss, timeout,
              _=None, exitIterations=1, trainingSetSize=100):
        if exitIterations < 1: return 

        print(f"Generating {trainingSetSize} expert trajectories")
        trainingData = []
        for _ in range(trainingSetSize):
            spec = getSpec()
            trajectory = self.sampleTrainingTrajectory(spec, loss, timeout)
            trainingData.append((spec, trajectory))

        print(f"Taking gradient steps...")
        for spec, trace in trainingData:
            self.model.gradientStepTrace(spec, trace)

        self.train(getSpec, loss, timeout,
                   exitIterations=exitIterations, trainingSetSize=trainingSetSize)
        
        
            
            

        
            


            
