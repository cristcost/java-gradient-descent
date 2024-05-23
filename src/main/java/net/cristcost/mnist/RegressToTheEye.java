package net.cristcost.mnist;

import static net.cristcost.differentiable.MathLibrary.*;
import java.io.IOException;
import net.cristcost.differentiable.ComputationGraphStats;
import net.cristcost.differentiable.ComputedTensor;
import net.cristcost.differentiable.Tensor;
import net.cristcost.differentiable.VariableTensor;

public class RegressToTheEye {

  private static final double LEARNING_RATE = 0.01;

  public static void main(String[] args) throws IOException {


    // y = a * x + b
    VariableTensor layer1Weights = matrix(5, 5).variable().normal(0.0, 1.0);
    VariableTensor layer1Bias = vector(5).variable().normal(0.0, 1.0);

    for (int epoch = 0; epoch < 200000; epoch++) {
      System.out.println("=== Round " + epoch + "===");


      Tensor input = unsqueeze(0, vector(5).normal(0.0, 1.0));
      Tensor target = input;

      ComputedTensor prediction = sum(matmul(input, layer1Weights), layer1Bias);
      ComputedTensor mseLoss = mse(target, prediction);

      System.out.println(String.format("          Loss value: %f", mseLoss.get(0)));


      if (mseLoss.get(0) < 0.0000001) {

        System.out.println();
        System.out.println("=== Converged to solution in " + epoch + " epochs ===");

        System.out.println("## loss function stats: ");
        ComputationGraphStats.printComputationGraphStats(mseLoss);
        System.out.println("## computation graph:");
        ComputationGraphStats.printComputationGraph(mseLoss);

        System.out.println(layer1Weights.json(2));
        System.out.println(layer1Bias.json(2));
        break;
      }

      mseLoss.startBackpropagation();

      // W -= w.getGradient() * LEARNING_RATE;
      layer1Weights = matrix(5, 5).variable()
          .withData(
              optimizeData(layer1Weights.getData(), layer1Weights.getGradient(), LEARNING_RATE));
      layer1Bias = vector(5).variable()
          .withData(
              optimizeData(layer1Bias.getData(), layer1Bias.getGradient(), LEARNING_RATE));

    }
  }

  private static double[] optimizeData(double[] data, double[] gradient, double learningRate) {
    double[] result = new double[data.length];

    for (int i = 0; i < data.length; i++) {
      if (Double.isNaN(gradient[i])) {
        throw new RuntimeException("gradient has a NaN");
      }
      result[i] = data[i] - gradient[i] * learningRate;
      if (Double.isNaN(result[i])) {
        throw new RuntimeException("optimizeData resulted in a NaN");
      }
    }
    return result;
  }
}
