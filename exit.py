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
              _=None, exitIterations=1, trainingSetSize=10,
              policyOracle=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

        reportingFrequency = 1
        n_attempts = 0
        n_successes = 0

        for iteration in range(1000000):
            print(f"Generating {trainingSetSize} expert trajectories")
            trainingData = []
            n_solutions = 0
            for _ in range(trainingSetSize):
                n_attempts += 1
                spec = getSpec()
                trajectory = self.sampleTrainingTrajectory(spec, loss, timeout)

                print("For the spec:")
                print(spec)
                print("We get the training trajectory:")
                print(trajectory)
                if self.reportedSolutions[-1].loss < 0.01:
                    trainingData.append((spec, trajectory))
                    print(trajectory[-1])
                    print("SOLVED")
                    n_solutions += 1
                    n_successes += 1
                else:
                    print("Did not solve!")
                    if policyOracle is not None:
                        print("Asking the Oracle for solution!")
                        trajectory = policyOracle(spec)
                        trainingData.append((spec, trajectory))

            print(f"Taking {len(trainingData)} gradient steps - {n_solutions} of which we found ourselves...")
            self.model.gradientStepTraceBatched(optimizer, trainingData)

            if iteration > 0 and iteration%reportingFrequency == 0:
                print(f"After {iteration*trainingSetSize} episodes, average success rate is {n_successes/n_attempts}")
                n_successes = 0
                n_attempts = 0
                with open("checkpoints/exit.pickle","wb") as handle:
                    pickle.dump(self.model, handle)

                print()
                print()

            
