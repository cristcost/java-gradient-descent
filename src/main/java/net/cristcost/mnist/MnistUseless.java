package net.cristcost.mnist;

import static net.cristcost.differentiable.MathLibrary.*;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import net.cristcost.differentiable.ComputationGraphStats;
import net.cristcost.differentiable.ComputedTensor;
import net.cristcost.differentiable.ConstantTensor;
import net.cristcost.differentiable.Tensor;
import net.cristcost.differentiable.VariableTensor;

public class MnistUseless {

  private static final double LEARNING_RATE = 0.001;

  public static void main(String[] args) throws IOException {

    List<Sample> trainDataset = MnistLoad.findSamplesInFolder(
        Path.of(".").toAbsolutePath().getParent().resolve("mnist/test"));
    List<Sample> testDataset = MnistLoad.findSamplesInFolder(
        Path.of(".").toAbsolutePath().getParent().resolve("mnist/train"));

    Collections.shuffle(trainDataset);
    Collections.shuffle(testDataset);

    // y = a * x + b
    VariableTensor layer1Weights = matrix(28 * 28, 128).variable().normal(0.0, 1.0);
    VariableTensor layer1Bias = vector(128).variable().normal(0.0, 1.0);
    VariableTensor layer2Weights = matrix(128, 64).variable().normal(0.0, 1.0);
    VariableTensor layer2Bias = vector(64).variable().normal(0.0, 1.0);
    VariableTensor layer3Weights = matrix(64, 10).variable().normal(0.0, 1.0);
    VariableTensor layer3Bias = vector(10).variable().normal(0.0, 1.0);

    for (int epoch = 0; epoch < 50; epoch++) {
      System.out.println("=== Round " + epoch + "===");


      Sample sample = trainDataset.get(epoch);

      Tensor input = unsqueeze(0, sample.getTensor());
      ConstantTensor target = vector(10).withData(makeCategoryData(sample.getLabel()));

      ComputedTensor layer1Out = relu(sum(matmul(input, layer1Weights), layer1Bias));
      ComputedTensor layer2Out = relu(sum(matmul(layer1Out, layer2Weights), layer2Bias));
      ComputedTensor prediction = softmax(sum(matmul(layer2Out, layer3Weights), layer3Bias));

      ComputedTensor mseLoss = mse(target, prediction);

      System.out.println("     Loss value: " + mseLoss.get(0));
      System.out.println("     Target was: " + sample.getLabel() + " " + target.json());
      System.out.println(
          "  Prediction is: " + findMaxCategory(prediction.getData()) + " " + prediction.json());

      if (mseLoss.get(0) < 0.0000001) {

        System.out.println();
        System.out.println("=== Converged to solution in " + epoch + " epochs ===");

        System.out.println("## loss function stats: ");
        ComputationGraphStats.printComputationGraphStats(mseLoss);
        System.out.println("## computation graph:");
        ComputationGraphStats.printComputationGraph(mseLoss);
        break;
      }

      mseLoss.startBackpropagation();

      // W -= w.getGradient() * LEARNING_RATE;
      layer1Weights = matrix(28 * 28, 128).variable()
          .withData(
              optimizeData(layer1Weights.getData(), layer1Weights.getGradient(), LEARNING_RATE));
      layer1Bias = vector(128).variable()
          .withData(
              optimizeData(layer1Bias.getData(), layer1Bias.getGradient(), LEARNING_RATE));
      layer2Weights = matrix(128, 64).variable()
          .withData(
              optimizeData(layer2Weights.getData(), layer2Weights.getGradient(), LEARNING_RATE));
      layer2Bias = vector(64).variable()
          .withData(
              optimizeData(layer2Bias.getData(), layer2Bias.getGradient(), LEARNING_RATE));
      layer3Weights = matrix(64, 10).variable()
          .withData(
              optimizeData(layer3Weights.getData(), layer3Weights.getGradient(), LEARNING_RATE));
      layer3Bias = vector(10).variable()
          .withData(
              optimizeData(layer3Bias.getData(), layer3Bias.getGradient(), LEARNING_RATE));

    }
  }

  private static int findMaxCategory(double[] data) {
    int result = 0;
    for (int i = 1; i < data.length; i++) {
      if (data[result] < data[i]) {
        result = i;
      }
    }
    return result;
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

  private static double[] makeCategoryData(int label) {
    double[] ret = new double[10];
    ret[label] = 1.0;
    return ret;
  }
}
