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


def test_simple():
	# spec = Circle(1,2,3)
	# tx = Parameter(torch.tensor(3.))
	# ty = Parameter(torch.tensor(3.))
	# td = Parameter(torch.tensor(4.))
	# params = [tx, ty, td]
	from collections import deque
	spec = Union(Circle(1,2,5),
	          Circle(2,21,9))

	#spec=Rectangle(14,2,14,30,26,30,26,2)

	abstract_prog = spec.abstract()
	specTensor = torch.tensor(spec.execute()).float()

	param_count = abstract_prog.get_param_count()

	old_params = spec.get_params()

	params = deque([Parameter(torch.tensor(random.choice(range(RESOLUTION))).float()) 
						for _ in range(param_count)])

	#params = deque( [Parameter(torch.tensor(x).float()) for x in old_params])
	#params = deque([Parameter(torch.tensor(x).float()) for x in [14,3,14,30,26,30,26,3]]) #whatever

	optimizer = optim.Adam(params, lr=0.5)

	for i in range(300):
		optimizer.zero_grad()

		render = abstract_prog._diff_render(params.copy(), r=64) #todo R hack...
		score = (specTensor*render).sum()/(specTensor + render - specTensor*render).sum()
		(-score).backward(retain_graph=True)
		optimizer.step()
		print(i, score)


	new_concrete = abstract_prog.concretize(deque(round(param.item()) for param in params))
	print("concrete IoU:", new_concrete.IoU(spec))

	print(params)

if __name__=='__main__':
	test_simple()