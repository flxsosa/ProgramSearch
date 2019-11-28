#test_differentiate.py

from CAD import *
from torch import optim
from torch.nn import Parameter

def test_full():
	spec = Union(Circle(1,2,3),
	          Circle(2,21,9))

	abstract_prog = spec.abstract()

	params = [] #todo
	optimizer = optim.Adam(params, lr=0.1)

	for i in range(100):
		optimizer.zero_grad()
		prog = abstract_prog.concretize(params)
		
		score = 1-spec.IoU(prog) #probably needs to be changed

		(-score).backward(retain_graph=True)
		optimizer.step()

		print(i, score)

def clip_params(params, bounds=None): #should take bounds as well
	if not bounds:
		for p in params:
			p.data.clamp_(min=0, max=RESOLUTION) 

	else:
		for p, (low, high) in zip(params, bounds):
			p.data.clamp_(min=0, max=RESOLUTION) 

def jostle_params(params, std=1.):
	for p in params:
		p.data += np.random.normal()*std

def test_simple():
	# spec = Circle(1,2,3)
	# tx = Parameter(torch.tensor(3.))
	# ty = Parameter(torch.tensor(3.))
	# td = Parameter(torch.tensor(4.))
	# params = [tx, ty, td]
	from collections import deque
	spec = Union(Circle(4,5,10),
				Difference(Rectangle(14,2,14,30,26,30,26,2), Circle(14,14,5)))
	spec.export('spec.png', 32)
	#spec=Rectangle(14,2,14,30,26,30,26,2)

	abstract_prog = spec.abstract()
	specTensor = torch.tensor(spec.execute()).float()

	param_count = abstract_prog.get_param_count()
	old_params = spec.get_params()
	params = deque([Parameter(torch.tensor(random.choice(range(RESOLUTION))).float()) 
						for _ in range(param_count)])

	params = deque( [Parameter(torch.tensor(x + random.choice(range(-int(RESOLUTION/4), int(RESOLUTION/4)))).float())
					 for x in old_params])
	#params = deque([Parameter(torch.tensor(x).float()) for x in [14,3,14,30,26,30,26,3]]) #whatever

	optimizer = optim.Adam(params, lr=0.5)

	t = time.time()
	tot = 100
	for i in range(tot):
		optimizer.zero_grad()

		render = abstract_prog._diff_render(params.copy(), r=64, temp=10000.)#/float(tot-i)) #todo R hack...
		score = (specTensor*render).sum()/(specTensor + render - specTensor*render).sum()
		(-score).backward(retain_graph=True)
		optimizer.step()
		print(i, score)

		if i%5==0:
			pass#jostle_params(params, std=RESOLUTION/4)
		clip_params(params)


	new_concrete = abstract_prog.concretize(deque(round(param.item()) for param in params))
	print("concrete IoU:", new_concrete.IoU(spec))
	new_concrete.export('found.png', 32)
	print(params)
	print("time of loop:", time.time()-t)


#annealing is possible

def ParamSearchR(spec, program, timeout=2):
		#.95
	return False


if __name__=='__main__':
	test_simple()