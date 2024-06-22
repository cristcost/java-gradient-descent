package net.cristcost.jtflow.operations.raw;

public class RawExponentiation {
  public static double[] compute(double[] base, double[] exponent) {
    if (base.length % exponent.length != 0) {
      throw new IllegalArgumentException("Shapes do not match nor are broadcastable");
    }

    double[] data = new double[base.length];
    for (int i = 0; i < data.length; i++) {
      data[i] = Math.pow(base[i], exponent[i % exponent.length]);
    }
    return data;
  }

  public static double[] baseGradient(double[] outerFunctionGradient, double[] base,
      double[] exponent) {
    double[] gradient = outerFunctionGradient.clone();

    for (int k = 0; k < gradient.length; k++) {
      // Mod over the size to broadcast implicitly
      gradient[k] *= exponent[k % exponent.length]
          * Math.pow(base[k % base.length], exponent[k % exponent.length] - 1);
    }
    return gradient;
  }

  public static double[] exponentGradient(double[] outerFunctionGradient, double[] base,
      double[] exponent) {
    double[] gradient = outerFunctionGradient.clone();

    for (int k = 0; k < gradient.length; k++) {
      // Mod over the size to broadcast implicitly
      gradient[k] *= Math.log(base[k % base.length])
          * Math.pow(base[k % base.length], exponent[k % exponent.length]);
    }
    return gradient;
  }
}
