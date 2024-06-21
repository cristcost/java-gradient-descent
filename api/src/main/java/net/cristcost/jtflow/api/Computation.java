package net.cristcost.jtflow.api;

public interface Computation {
  Operation getOperation();

  Tensor[] getOperands();
}
