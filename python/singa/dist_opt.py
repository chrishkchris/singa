from . import singa_wrap as singa

class Dist_SGD(object):
	def __init__(self, lr=0.01, nDev=1):
		self.lr=lr
		# def start_MPI():
		# 	pass
		# def create_communicator():
		# 	pass
			# could be combined with start_MPI
		self.communicator=singa.Communicator(nDev)
		self.world_size=self.communicator.totalMPIRanksInGlobal
		self.rank_in_local=self.communicator.MPIRankInLocal
		self.rank_in_global=self.communicator.MPIRankInGlobal

	def dist_update(self, param, grad):
		# singa.synch(grad.data, self.communicator)
		# grad /= self.communicator.totalMPIRanksInGlobal
		grad = self.synch(grad)
		param -= grad * self.lr 

	def synch(self, tensor):
		singa.synch(tensor.data, self.communicator)
		tensor /= self.world_size
		return tensor
	

