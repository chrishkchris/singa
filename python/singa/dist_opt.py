from . import singa_wrap as singa
from .opt import SGD


class DistOpt(object):

    def __init__(self, opt=SGD(), nDev=1):
        # The class is designed to wrap an optimizer to do disttributed training.
        # opt: The optimizer to be wrapped. nDev: number of devices(GPUs) a
        # process will control/use.

        # world_size: total number of processes.
        # rank_in_local: local rank of a process on the current node.
        # rank_in_global: global rank of a process

        self.opt = opt
        self.communicator = singa.Communicator(nDev)
        self.world_size = self.communicator.totalMPIRanksInGlobal
        self.rank_in_local = self.communicator.MPIRankInLocal
        self.rank_in_global = self.communicator.MPIRankInGlobal

    def update(self, param, grad):
        # singa.synch(grad.data, self.communicator)
        # grad /= self.communicator.totalMPIRanksInGlobal
        grad = self.synch(grad)
        #param -= grad * self.lr
        self.opt.update(param, grad)

    def synch(self, tensor):
        singa.synch(tensor.data, self.communicator)
        tensor /= self.world_size
        return tensor
