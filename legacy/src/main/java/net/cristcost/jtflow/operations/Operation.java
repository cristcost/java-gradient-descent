package net.cristcost.jtflow.operations;

import java.util.function.BiConsumer;
import java.util.function.Function;
import lombok.RequiredArgsConstructor;
import net.cristcost.jtflow.JTFlow;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.operations.impl.CategoricalCrossentropy;
import net.cristcost.jtflow.operations.impl.DotProduct;
import net.cristcost.jtflow.operations.impl.MatMul2;
import net.cristcost.jtflow.operations.impl.MathOperationShapeComputation;
import net.cristcost.jtflow.operations.impl.MathOperationsBackpropagation;
import net.cristcost.jtflow.operations.impl.MathOperationsImplementation;
import net.cristcost.jtflow.operations.impl.MeanSquareError;
import net.cristcost.jtflow.operations.impl.SoftMax;
import net.cristcost.jtflow.tensors.ComputedTensor;

@RequiredArgsConstructor
public enum Operation {

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

  DOT(
      false,
      operands -> DotProduct.dot(operands[0], operands[1]),
      operands -> DotProduct.shape(operands[0], operands[1]),
      (grad, operands) -> DotProduct.chain(grad, operands[0], operands[1])),

  MSE(
      false,
      operands -> MeanSquareError.mse(operands[0], operands[1]),
      operands -> MeanSquareError.shape(operands[0], operands[1]),
      (grad, operands) -> MeanSquareError.chain(grad, operands[0], operands[1])),

  CATEGORICAL_CROSSENTROPY(
      false,
      operands -> CategoricalCrossentropy.cce(operands[0], operands[1]),
      operands -> CategoricalCrossentropy.shape(operands[0], operands[1]),
      (grad, operands) -> CategoricalCrossentropy.chain(grad, operands[0], operands[1])),

  SOFTMAX(
      false,
      operands -> SoftMax.softmax(operands[0]),
      operands -> MathOperationShapeComputation.identity(operands[0]),
      (grad, operands) -> SoftMax.chain(grad, operands[0])),

  MATMUL(
      false,
      operands -> MatMul2.matmul(operands[0], operands[1]),
      operands -> MatMul2.matmulShape(operands[0], operands[1]),
      (grad, operands) -> MatMul2.chain(grad, operands[0], operands[1]));

  private final boolean broadcastSupported;

  private final Function<Tensor[], double[]> operationFunction;

  private final Function<Tensor[], int[]> resultShapeFunction;

  private final BiConsumer<double[], Tensor[]> backpropagationFunction;



  public ComputedTensor compute(Tensor... operands) {
    int[] shape = resultShapeFunction.apply(operands);

    double[] operationResult = operationFunction
        .apply(broadcastSupported ? JTFlow.broadCast(shape, operands) : operands);

    Computation computation = new Computation(this, operands);
    return new ComputedTensor(operationResult, shape, computation);
  }

  public void backpropagate(double[] gradient, Tensor... operands) {
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
