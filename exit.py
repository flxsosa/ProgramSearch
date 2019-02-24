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

    
            
