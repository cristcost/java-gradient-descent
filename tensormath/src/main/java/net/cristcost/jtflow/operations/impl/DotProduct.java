package net.cristcost.jtflow.operations.impl;

import java.util.Arrays;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.operations.raw.RawDotProduct;

public class DotProduct {

  public static void chain(double outerFunctionGradient, Tensor a, Tensor b) {
    a.ifChainable(c -> c.backpropagate(RawDotProduct.gradient(outerFunctionGradient, b.getData())));
    b.ifChainable(c -> c.backpropagate(RawDotProduct.gradient(outerFunctionGradient, a.getData())));
  }


  public static double[] compute(Tensor a, Tensor b) {
    validateVectorCompatibility(a, b);

    return Common.makeData(RawDotProduct.compute(a.getData(), b.getData()));
  }

  public static int[] shape(Tensor tensor, Tensor other) {
    validateVectorCompatibility(tensor, other);
    return Common.SCALAR_SHAPE;
  }

  private static void validateVectorCompatibility(Tensor a, Tensor b) {
    if (a.getShape().length != 1 || b.getShape().length != 1) {
      throw new IllegalArgumentException(
          "Dot product operation requires vector of exavtly 1 dimension.");
    }

    int size = Math.max(a.getShape()[0], b.getShape()[0]);
    if (size % a.size() != 0 || size % a.size() != 0) {
      throw new IllegalArgumentException(
          String.format(
              "The vectors size are not compatible:  First Tensor Shape: (%s), Second Tensor Shape: (%s)",
              Arrays.toString(a.getShape()), Arrays.toString(b.getShape())));
    }
  }
}
