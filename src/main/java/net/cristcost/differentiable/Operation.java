package net.cristcost.differentiable;

import java.util.function.Function;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
enum Operation {

  ADDITION(MathOperationsImplementation::sum),

  MULTIPLICATION(MathOperationsImplementation::multiply),

  ESPONENTIATION(operands -> MathOperationsImplementation.pow(operands[0], operands[1])),

  RELU(operands -> MathOperationsImplementation.relu(operands[0]));


  private final Function<Tensor[], double[]> operationFunction;

  public ComputedTensor compute(Tensor... operands) {
    int[] shape = MathLibrary.findResultShape(operands);
    double[] operationResult = operationFunction.apply(MathLibrary.broadCast(shape, operands));
    // double[] result = MathOperationsImplementation.multiply(broadCast(shape, operands));
    // return new ConstantTensor(result, shape);

    Computation computation = new Computation(this, operands);
    return new ComputedTensor(operationResult, shape, computation);
  }

}
