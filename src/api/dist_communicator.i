%module dist_communicator

%{
#include "singa/dist/communicator.h"
%}

namespace singa{

class Communicator {
public:
  int MPIRankInGlobal;
  int totalMPIRanksInGlobal;
  int MPIRankInLocal;
  Communicator(int nDev);
};

void synch(Tensor &t1, Communicator &c);

}