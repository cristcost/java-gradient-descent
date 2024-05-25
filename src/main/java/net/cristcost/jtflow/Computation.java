package net.cristcost.jtflow;

import lombok.AccessLevel;
import lombok.Getter;

class Computation {
  public Computation(Operation operation, Tensor... operands) {
    this.operation = operation;
    this.operands = operands;
  }

  @Getter(AccessLevel.PACKAGE)
  private final Operation operation;

  @Getter(AccessLevel.PACKAGE)
  private final Tensor[] operands;
}
