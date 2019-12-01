


#48 hour triplet runs: Due Monday midnight

#sbatch execute_gpu_1080.sh python driver.py imitation --2d --train_abstraction --contrastive --vector_loss_type norm --checkpoint AbstractContrastiveNormVecLoss.pickle --trainTime 48
#15675749
#sbatch execute_gpu_1080.sh python driver.py imitation --2d --train_abstraction --contrastive --vector_loss_type triplet --checkpoint AbstractContrastiveTripletVecLossAlpha05.pickle --alpha 0.05 --trainTime 48
#15675858
#sbatch execute_gpu_1080.sh python driver.py imitation --2d --train_abstraction --contrastive --vector_loss_type triplet --checkpoint AbstractContrastiveTripletVecLossAlpha2.pickle --alpha 0.2 --trainTime 48
#15675818
#sbatch execute_gpu_1080.sh python driver.py imitation --2d --train_abstraction --contrastive --vector_loss_type triplet --checkpoint AbstractContrastiveTripletVecLossAlpha1.pickle --alpha 1.0 --trainTime 48
#15675859
#sbatch execute_gpu_1080.sh python driver.py imitation --2d --train_abstraction --contrastive --vector_loss_type triplet --checkpoint AbstractContrastiveTripletVecLossAlpha5.pickle --alpha 5.0 --trainTime 48
#15675873

#sbatch execute_gpu_1080.sh python driver.py imitation --2d --train_abstraction --contrastive --checkpoint AbstractContrastive.pickle --trainTime 48
#15676229






##########
normal actor and critic, w abstraction:
	just sample:
	python driver.py test --2d --checkpoint 'checkpoints/2d_critic_abstraction_2019-11-07T11:08:33.pickle' --train_abstraction
	2019-11-11T11/06/06
	
	python driver.py test --2d --checkpoint 'checkpoints/2d_critic_abstraction_2019-11-07T11:08:33.pickle' --train_abstraction --solvers SMC
	2019-11-11T11:25:58

noexecution with sample:
	python driver.py test --2d --checkpoint 'checkpoints/2d_imitation_noExecution_abstraction_2019-11-07T10:50:12.pickle' --train_abstraction --noExecution --solvers noExecution
	2019-11-11T11/20/02



need to run:
retrained critic:
python driver.py test --2d --checkpoint 'checkpoints/2d_critic_abstraction_2019-11-11T14:12:50.pickle' --train_abstraction --solvers SMC --timeout 10

base imitation:
python driver.py test --2d --checkpoint 'checkpoints/2d_imitation_2019-11-11T14:37:51.pickle' --solvers fs --timeout 10