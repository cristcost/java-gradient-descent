package net.cristcost.differentiable;

import java.util.function.BiConsumer;
import java.util.function.Function;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
enum Operation {

  ADDITION(
      true,
      MathOperationsImplementation::sum,
      MathOperationShapeComputation::maxShape,
      MathOperationsBackpropagation::sum),

  MULTIPLICATION(
      true,
      MathOperationsImplementation::multiply,
      MathOperationShapeComputation::maxShape,
      MathOperationsBackpropagation::multiply),

  ESPONENTIATION(
      true,
      operands -> MathOperationsImplementation.pow(operands[0], operands[1]),
      MathOperationShapeComputation::maxShape,
      (grad, operands) -> MathOperationsBackpropagation.pow(grad, operands[0], operands[1])),

  RELU(
      false,
      operands -> MathOperationsImplementation.relu(operands[0]),
      operands -> MathOperationShapeComputation.identity(operands[0]),
      (grad, operands) -> MathOperationsBackpropagation.relu(grad, operands[0])),


  MATMUL(
      false,
      operands -> MathOperationsImplementation.matmul(operands[0], operands[1]),
      Operation::resultShapeNotImplemented,
      Operation::backPropagationNotImplemented),

  MATMUL_NDIM(
      false,
      operands -> MathOperationsImplementation.matmulNdim(operands[0], operands[1]),
      Operation::resultShapeNotImplemented,
      Operation::backPropagationNotImplemented),

  SOFTMAX(
      false,
      Operation::operationNotImplemented,
      Operation::resultShapeNotImplemented,
      Operation::backPropagationNotImplemented),

  MSE(
      false,
      Operation::operationNotImplemented,
      Operation::resultShapeNotImplemented,
      Operation::backPropagationNotImplemented);

  private final boolean broadcastSupported;

  private final Function<Tensor[], double[]> operationFunction;

  private final Function<Tensor[], int[]> resultShapeFunction;

  private final BiConsumer<double[], Tensor[]> backpropagationFunction;



  public ComputedTensor compute(Tensor... operands) {
    int[] shape = resultShapeFunction.apply(operands);

    double[] operationResult = operationFunction
        .apply(broadcastSupported ? MathLibrary.broadCast(shape, operands) : operands);

    Computation computation = new Computation(this, operands);
    return new ComputedTensor(operationResult, shape, computation);
  }

  void backpropagate(double[] gradient, Tensor... operands) {
    backpropagationFunction.accept(gradient, operands);
  }

  private static double[] operationNotImplemented(Tensor... operands) {
    throw new UnsupportedOperationException("Not yet implemented");
  }

  private static void backPropagationNotImplemented(double[] gradient, Tensor... operands) {
    throw new UnsupportedOperationException("Not yet implemented");
  }

  private static int[] resultShapeNotImplemented(Tensor... operands) {
    throw new UnsupportedOperationException("Not yet implemented");
  }


}
