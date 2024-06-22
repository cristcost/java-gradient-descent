package net.cristcost.jtflow.operations.impl;

import java.util.Arrays;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.operations.raw.RawMultiplication;


public class Multiplication {

  public static void chain(double[] outerFunctionGradient, Tensor... operands) {
    double[][] rawOperands = Arrays.stream(operands).map(t -> t.getData()).toArray(double[][]::new);

    for (int i = 0; i < operands.length; i++) {
      final int index = i;
      operands[i].ifChainable(c -> c.backpropagate(
          RawMultiplication.gradient(outerFunctionGradient, index, rawOperands)));
    }
  }

  public static double[] compute(Tensor... operands) {
    return RawMultiplication.compute(operands[0].size(),
        Arrays.stream(operands).map(t -> t.getData()).toArray(double[][]::new));
  }
}
