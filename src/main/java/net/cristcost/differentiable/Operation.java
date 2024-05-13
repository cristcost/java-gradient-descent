package net.cristcost.differentiable;

import java.util.function.BiConsumer;
import java.util.function.Function;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
enum Operation {

  ADDITION(
      MathOperationsImplementation::sum,
      MathOperationsBackpropagation::sum),

  MULTIPLICATION(
      MathOperationsImplementation::multiply,
      MathOperationsBackpropagation::multiply),

  ESPONENTIATION(
      operands -> MathOperationsImplementation.pow(operands[0], operands[1]),
      (grad, operands) -> MathOperationsBackpropagation.pow(grad, operands[0], operands[1])),

  RELU(
      operands -> MathOperationsImplementation.relu(operands[0]),
      (grad, operands) -> MathOperationsBackpropagation.relu(grad, operands[0]));


  private final Function<Scalar[], Double> operationFunction;

  private final BiConsumer<Double, Scalar[]> backpropagationFunction;

  public ComputedScalar compute(Scalar... operands) {
    double operationResult = operationFunction.apply(operands);

    Computation computation = new Computation(this, operands);
    return new ComputedScalar(operationResult, computation);
  }

  public void backpropagate(double gradient, Scalar... operands) {
    backpropagationFunction.accept(gradient, operands);
  }
}
