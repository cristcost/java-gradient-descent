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
      (grad, operands) -> MathOperationsBackpropagation.relu(grad, operands[0])),


  MATMUL(
      operands -> MathOperationsImplementation.matmul(operands[0], operands[1]),
      Operation::notImplemented),

  SOFTMAX(
      Operation::notImplemented,
      Operation::notImplemented),

  MSE(
      Operation::notImplemented,
      Operation::notImplemented);


  private final Function<Tensor[], double[]> operationFunction;

  private final BiConsumer<double[], Tensor[]> backpropagationFunction;


  public ComputedTensor compute(Tensor... operands) {
    int[] shape = MathLibrary.findResultShape(operands);
    double[] operationResult = operationFunction.apply(MathLibrary.broadCast(shape, operands));

    Computation computation = new Computation(this, operands);
    return new ComputedTensor(operationResult, shape, computation);
  }

  void backpropagate(double[] gradient, Tensor... operands) {
    backpropagationFunction.accept(gradient, operands);
  }

  private static double[] notImplemented(Tensor... operands) {
    throw new UnsupportedOperationException("Not yet implemented");
  }

  private static void notImplemented(double[] gradient, Tensor... operands) {
    throw new UnsupportedOperationException("Not yet implemented");
  }


}
