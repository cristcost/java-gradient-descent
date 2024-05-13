package net.cristcost.differentiable;

class Computation {
  public Computation(Operation operation, Scalar... operands) {
    this.operation = operation;
    this.operands = operands;
  }

  private final Operation operation;
  private final Scalar[] operands;
}
