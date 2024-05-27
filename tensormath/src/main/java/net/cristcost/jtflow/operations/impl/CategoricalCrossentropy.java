package net.cristcost.jtflow.operations.impl;

import java.util.Arrays;
import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;

public class CategoricalCrossentropy {

  private static final double EPSILON = 1e-15;

  public static double[] cce(Tensor prediction, Tensor oneHotEncodedLabels) {
    validateTensorCompatibility(prediction, oneHotEncodedLabels);

    double result = cce(prediction.getData(), oneHotEncodedLabels.getData());

    return Common.makeData(result);
  }

  protected static double cce(double[] predictionData, double[] oneHotEncodedData) {
    return NegativeLogLikelihoodLoss.nll(SoftMax.softmax(predictionData), oneHotEncodedData);
  }

  public static int[] shape(Tensor tensor, Tensor other) {
    validateTensorCompatibility(tensor, other);
    return Common.SCALAR_SHAPE;
  }

  private static void validateTensorCompatibility(Tensor a, Tensor b) {
    if (a.size() != b.size()) {
      throw new IllegalArgumentException(
          "Tensor dimensions are not compatible.");
    }
  }

  public static void chain(double[] outerFunctionGradient, Tensor prediction,
      Tensor oneHotEncodedLabels) {

    if (prediction instanceof Chainable) {

      double[] predictionData = prediction.getData();
      double[] oneHotEncodedLabelsData = oneHotEncodedLabels.getData();
      double[] innerGradient =
          predictionsGradient(outerFunctionGradient[0], predictionData, oneHotEncodedLabelsData);

      ((Chainable) prediction).backpropagate(innerGradient);
    }

    if (oneHotEncodedLabels instanceof Chainable) {
      // We could simply ignore chaining for label, but let's fail fast to allow detecting misuses
      // and eventually change the code in the future
      throw new RuntimeException("CategoricalCrossentropy Labels are not expected to be Variable.");
    }
  }

  protected static double[] predictionsGradient(double outerFunctionGradient,
      double[] predictionData,
      double[] oneHotEncodedLabelsData) {


    // f(g(x)) = nll(softmax(x), oneHotEncodedData)
    // ∂f(g(x))/∂x = f'(g(x)) * g'(x) = nll'(softmax(x), oneHotEncodedData) * softmax'(x)



    double[] nllGradient = NegativeLogLikelihoodLoss.predictionsGradient(outerFunctionGradient,
        SoftMax.softmax(predictionData), oneHotEncodedLabelsData);

    double[] gradient = SoftMax.gradient(nllGradient, predictionData);
    // for (int i = 0; i < gradient.length; i++) {
    // gradient[i] *= nllGradient[i];
    // }

    return gradient;

    // // CrossEntropy = − ∑ i=[0..length] (oneHotLabel[i] * log(pred[i]))
    // // ∂CrossEntropy / ∂pred[i] = − (oneHotLabel[i] / pred[i])
    //
    // double[] innerGradient = new double[predictionData.length];
    // for (int k = 0; k < innerGradient.length; k++) {
    //
    // double p = predictionData[k % predictionData.length];
    // // As we are clamping the prediction value, the derivate varies only within this range
    // if (p > EPSILON && p < 1.0 - EPSILON) {
    // innerGradient[k] =
    // -outerFunctionGradient
    // * oneHotEncodedLabelsData[k % oneHotEncodedLabelsData.length]
    // / predictionData[k % predictionData.length];
    // } else {
    // innerGradient[k] = 0.0;
    // }
    // }
    // return innerGradient;
  }


  public static double f(double[] x) {
    return Arrays.stream(x).sum();
  }

  public static double[] dfdx(double[] x) {
    double[] ret = new double[x.length];
    Arrays.fill(ret, 1.0);
    return ret;
  }

  public static double[] g(double[] x) {
    double[] ret = x.clone();
    for (int i = 0; i < ret.length; i++) {
      ret[i] *= 3;
    }
    return ret;
  }

  public static double[] dgdx(double[] x) {
    double[] ret = new double[x.length];
    Arrays.fill(ret, 3.0);
    return ret;
  }

  public static double[] dhdx(double[] x) {
    double[] ret = new double[x.length];

    double[] dfdxOfG = dfdx(g(x));
    double[] dgdx = dgdx(x);
    for (int i = 0; i < ret.length; i++) {
      ret[i] = dfdxOfG[i] * dgdx[i];
    }
    return ret;
  }


}
