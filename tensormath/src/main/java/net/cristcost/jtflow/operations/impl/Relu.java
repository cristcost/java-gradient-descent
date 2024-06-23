package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.operations.raw.RawRelu;

public class Relu {

  public static void chain(double[] outerFunctionGradient, Tensor operand) {
    operand.ifChainable(
        c -> c.backpropagate(RawRelu.gradient(outerFunctionGradient, operand.getData())));
  }

  public static double[] compute(Tensor operand) {
    return RawRelu.compute(operand.getData());
  }

}
