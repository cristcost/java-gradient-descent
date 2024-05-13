package net.cristcost.differentiable;

import lombok.AccessLevel;
import lombok.Getter;

class Computation {
  public Computation(Operation operation, Scalar... operands) {
    this.operation = operation;
    this.operands = operands;
  }

  @Getter(AccessLevel.PACKAGE)
  private final Operation operation;

  @Getter(AccessLevel.PACKAGE)
  private final Scalar[] operands;
}
