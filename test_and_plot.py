import os
from datetime import datetime

#TODO add timeout stuff so can rerun
timestamp = datetime.now().strftime('%FT%T')
timestamp += '40obj30secTriplet'

# checkpoints = [
# ('abs_modular', '2d_imitation_abstraction_2019-11-23T17:53:35.pickle', 'SMC'),
# ('abs_modular_rl' , '2d_critic_abstraction_2019-11-22T11:42:53.pickle', 'SMC'),
# #('modularRLfinetune', '2d_rl_after_contrastive.pickle'),
# ('abs_non_modular', 'nonmodularAbstractContrastiveNoSpecInOE.pickle', 'SMC'),
# ('no_REPL', '2d_imitation_noExecution_abstraction_2019-11-07T10:50:12.pickle', 'noExecution'),
# ('abs_non_modular_withSpec', '2d_imitation_abstraction_2019-11-28T13:21:02.pickle', 'SMC'),
# ('full_repl', '2d_critic_2019-11-30T01:01:12.pickle', 'SMC'),
# ]

checkpoints = [
('AbsContNorm','AbstractContrastiveNormVecLoss.pickle','SMC'),
('AbsContTriplet.2', 'AbstractContrastiveTripletVecLossAlpha2.pickle', 'SMC'),
('abs_modular', 'AbstractContrastive.pickle', 'SMC'),
('AbsContTriplet.05', 'AbstractContrastiveTripletVecLossAlpha05.pickle', 'SMC'),
('AbsContTriplet1', 'AbstractContrastiveTripletVecLossAlpha1.pickle', 'SMC'),
('AbsContTriplet5', 'AbstractContrastiveTripletVecLossAlpha5.pickle', 'SMC'),
]


for name, checkpoint, solver in checkpoints:
	os.system(f"python driver.py test --2d --train_abstraction --checkpoint checkpoints/{checkpoint} --solvers {solver} --timeout 30 --outputFolder {name}{timestamp} --maxShapes 40")

names, _, _ = zip(*checkpoints)
testResults = " ".join(f"experimentOutputs/{name}{timestamp}/testResults.pickle" for name in names)
names = " ".join(names)

cmd = f"python graphs.py {testResults} --names {names} -t 20 -e figures/graph{timestamp}.png -n testgraph"
print(cmd)
os.system(cmd)


#python driver.py test --2d --train_abstraction --checkpoint checkpoints/2d_imitation_abstraction_2019-11-23T17:53:35.pickle --solvers fs --timeout 20 --maxShapes 30

#python driver.py test --2d --train_abstraction --checkpoint checkpoints/2d_critic_abstraction_2019-11-22T11:42:53.pickle --solvers fs --outputFolder fs_NonContrastive --timeout 20 --maxShapes 30
#python driver.py test --2d --train_abstraction --checkpoint checkpoints/nonmodularAbstractContrastiveNoSpecInOE.pickle --solvers SMC --timeout 30 --outputFolder abs_non_modular2019-11-30T18:32:1840obj30sec --maxShapes 40