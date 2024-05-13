package net.cristcost.differentiable;

import java.util.function.Function;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
enum Operation {

  ADDITION(MathOperationsImplementation::sum),

  MULTIPLICATION(MathOperationsImplementation::multiply),

  ESPONENTIATION(operands -> MathOperationsImplementation.pow(operands[0], operands[1])),

  RELU(operands -> MathOperationsImplementation.relu(operands[0]));


  private final Function<Scalar[], Double> operationFunction;

  public ComputedScalar compute(Scalar... operands) {
    double operationResult = operationFunction.apply(operands);

    Computation computation = new Computation(this, operands);
    return new ComputedScalar(operationResult, computation);
  }

}
