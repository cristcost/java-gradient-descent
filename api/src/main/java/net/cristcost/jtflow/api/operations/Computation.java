package net.cristcost.jtflow.api.operations;

import lombok.Getter;
import net.cristcost.jtflow.api.Tensor;

public class Computation {
  public Computation(Operation operation, Tensor... operands) {
    this.operation = operation;
    this.operands = operands;
  }

  @Getter
  private final Operation operation;

  @Getter
  private final Tensor[] operands;
}
