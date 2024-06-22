package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.operations.raw.RawExponentiation;

public class Exponentiation {

  public static void chain(double[] outerFunctionGradient, Tensor base, Tensor exponent) {

    base.ifChainable(c -> c.backpropagate(
        RawExponentiation.baseGradient(outerFunctionGradient, base.getData(), exponent.getData())));

    exponent.ifChainable(c -> c.backpropagate(
        RawExponentiation.exponentGradient(outerFunctionGradient, base.getData(),
            exponent.getData())));
  }

  public static double[] compute(Tensor base, Tensor exponent) {
    return RawExponentiation.compute(base.getData(), exponent.getData());
  }

}
