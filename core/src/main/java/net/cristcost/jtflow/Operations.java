package net.cristcost.jtflow;

import java.util.function.BiConsumer;
import java.util.function.Function;
import lombok.RequiredArgsConstructor;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.api.operations.Computation;
import net.cristcost.jtflow.api.operations.Operation;
import net.cristcost.jtflow.operations.impl.Addition;
import net.cristcost.jtflow.operations.impl.CategoricalCrossentropy;
import net.cristcost.jtflow.operations.impl.Common;
import net.cristcost.jtflow.operations.impl.DotProduct;
import net.cristcost.jtflow.operations.impl.Exponentiation;
import net.cristcost.jtflow.operations.impl.MatMul;
import net.cristcost.jtflow.operations.impl.MeanSquareError;
import net.cristcost.jtflow.operations.impl.Multiplication;
import net.cristcost.jtflow.operations.impl.Relu;
import net.cristcost.jtflow.operations.impl.SoftMax;
import net.cristcost.jtflow.tensors.ComputedTensor;

@RequiredArgsConstructor
public enum Operations implements Operation {

  ADDITION(
      true,
      Addition::compute,
      Common::maxShape,
      Addition::chain),

  MULTIPLICATION(
      true,
      Multiplication::compute,
      Common::maxShape,
      Multiplication::chain),

  ESPONENTIATION(
      true,
      operands -> Exponentiation.compute(operands[0], operands[1]),
      Common::maxShape,
      (grad, operands) -> Exponentiation.chain(grad, operands[0], operands[1])),

  RELU(
      false,
      operands -> Relu.compute(operands[0]),
      operands -> Common.identity(operands[0]),
      (grad, operands) -> Relu.chain(grad, operands[0])),

  DOT(
      false,
      operands -> DotProduct.compute(operands[0], operands[1]),
      operands -> DotProduct.shape(operands[0], operands[1]),
      (grad, operands) -> DotProduct.chain(grad, operands[0], operands[1])),

  MSE(
      false,
      operands -> MeanSquareError.compute(operands[0], operands[1]),
      operands -> MeanSquareError.shape(operands[0], operands[1]),
      (grad, operands) -> MeanSquareError.chain(grad, operands[0], operands[1])),

  CATEGORICAL_CROSSENTROPY(
      false,
      operands -> CategoricalCrossentropy.compute(operands[0], operands[1]),
      operands -> CategoricalCrossentropy.shape(operands[0], operands[1]),
      (grad, operands) -> CategoricalCrossentropy.chain(grad, operands[0], operands[1])),

  SOFTMAX(
      false,
      operands -> SoftMax.compute(operands[0]),
      operands -> Common.identity(operands[0]),
      (grad, operands) -> SoftMax.chain(grad, operands[0])),

  MATMUL(
      false,
      operands -> MatMul.compute(operands[0], operands[1]),
      operands -> MatMul.shape(operands[0], operands[1]),
      (grad, operands) -> MatMul.chain(grad, operands[0], operands[1]));

  private static void backPropagationNotImplemented(double[] gradient, Tensor... operands) {
    throw new UnsupportedOperationException("Not yet implemented");
  }

  private static double[] operationNotImplemented(Tensor... operands) {
    throw new UnsupportedOperationException("Not yet implemented");
  }

  private static int[] resultShapeNotImplemented(Tensor... operands) {
    throw new UnsupportedOperationException("Not yet implemented");
  }

  private final boolean broadcastSupported;


  private final Function<Tensor[], double[]> operationFunction;

  private final Function<Tensor[], int[]> resultShapeFunction;

  private final BiConsumer<double[], Tensor[]> backpropagationFunction;

  @Override
  public void backpropagate(double[] gradient, Tensor... operands) {
    backpropagationFunction.accept(gradient, operands);
  }

  @Override
  public ComputedTensor compute(Tensor... operands) {
    int[] shape = resultShapeFunction.apply(operands);

    double[] operationResult = operationFunction
        .apply(broadcastSupported ? JTFlow.broadCast(shape, operands) : operands);

    Computation computation = new Computation(this, operands);
    return new ComputedTensor(operationResult, shape, computation);
  }


}
