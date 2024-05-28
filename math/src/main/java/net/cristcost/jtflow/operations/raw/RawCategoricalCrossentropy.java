package net.cristcost.jtflow.operations.raw;

public class RawCategoricalCrossentropy {

  public static double compute(double[] predictionData, double[] oneHotEncodedData) {
    return RawNegativeLogLikelihoodLoss.compute(RawSoftMax.compute(predictionData),
        oneHotEncodedData);
  }

  public static double[] gradient(double outerFunctionGradient, double[] predictionData,
      double[] oneHotEncodedLabelsData) {


    // f(g(x)) = nll(softmax(x), oneHotEncodedData)
    // ∂f(g(x))/∂x = f'(g(x)) * g'(x) = nll'(softmax(x), oneHotEncodedData) * softmax'(x)

    double[] gradient =
        RawSoftMax.gradient(RawNegativeLogLikelihoodLoss.gradient(outerFunctionGradient,
            RawSoftMax.compute(predictionData), oneHotEncodedLabelsData), predictionData);

    return gradient;
  }
}
