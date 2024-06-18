package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Tensor;

public class Common {

  public static final int[] SCALAR_SHAPE = {};

  public static int[] identity(Tensor tensor) {
    return tensor.getShape();
  }

  public static double[] makeData(double... data) {
    return data;
  }

  public static int[] maxShape(Tensor... operands) {
    int[] fullShape = SCALAR_SHAPE;
    for (Tensor t : operands) {
      if (fullShape.length < t.getShape().length) {
        fullShape = t.getShape();
      }
    }
    return fullShape.clone();
  }

}
