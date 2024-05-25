package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.JTFlow;
import net.cristcost.jtflow.api.Tensor;

public class MathOperationShapeComputation {

  public static int[] identity(Tensor tensor) {
    return tensor.getShape();
  }

  public static int[] maxShape(Tensor... operands) {
    int[] fullShape = JTFlow.shape();
    for (Tensor t : operands) {
      if (fullShape.length < t.getShape().length) {
        fullShape = t.getShape();
      }
    }
    return fullShape.clone();
  }


}
