package net.cristcost.jtflow.operations.raw;


public class RawDotProduct {

  public static double compute(double[] firstVector, double[] secondVector) {

    int size = Math.max(firstVector.length, secondVector.length);
    if (size % firstVector.length != 0 || size % secondVector.length != 0) {
      // TODO: We can probably broadcast some dimensions, for now let's expect exact shape
      throw new IllegalArgumentException(
          String.format(
              "The vectors size are not compatible:  First Vector Size: %d, Second Vector Size: %d",
              firstVector.length, secondVector.length));
    }


    double result = 0.0;
    for (int i = 0; i < size; i++) {
      result += firstVector[i % firstVector.length] * secondVector[i % secondVector.length];
    }
    return result;
  }

  public static double[] gradient(double outerFunctionGradient, double[] otherVector) {
    double[] innerGradient = new double[otherVector.length];

    for (int k = 0; k < innerGradient.length; k++) {
      innerGradient[k] = outerFunctionGradient * otherVector[k % otherVector.length];
    }
    return innerGradient;
  }
}
