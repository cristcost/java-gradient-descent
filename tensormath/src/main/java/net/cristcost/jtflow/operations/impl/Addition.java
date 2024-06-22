package net.cristcost.jtflow.operations.impl;

import java.util.Arrays;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.operations.raw.RawAddition;

public class Addition {

  public static void chain(double[] outerFunctionGradient, Tensor... operands) {
    for (Tensor operand : operands) {
      operand.ifChainable(c -> c.backpropagate(RawAddition.gradient(outerFunctionGradient)));
    }
  }

  public static double[] compute(Tensor... operands) {
    return RawAddition.compute(operands[0].size(),
        Arrays.stream(operands).map(t -> t.getData()).toArray(double[][]::new));
  }

}
