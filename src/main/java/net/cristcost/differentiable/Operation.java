package net.cristcost.differentiable;

import java.util.function.Function;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
class Operation {

  private final Function<Scalar[], Double> operationFunction;

  public ComputedScalar compute(Scalar... operands) {
    double operationResult = operationFunction.apply(operands);

    Computation computation = new Computation(this, operands);
    return new ComputedScalar(operationResult, computation);
  }

}
