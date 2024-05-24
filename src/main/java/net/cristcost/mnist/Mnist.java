package net.cristcost.mnist;

import static net.cristcost.differentiable.MathLibrary.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import net.cristcost.differentiable.ComputationGraphStats;
import net.cristcost.differentiable.ComputedTensor;
import net.cristcost.differentiable.ConstantTensor;
import net.cristcost.differentiable.Tensor;
import net.cristcost.differentiable.VariableTensor;

public class Mnist {

  private static final double LEARNING_RATE = 0.05;

  public static void main(String[] args) throws IOException {

    Path trainSamplesFolder = Path.of("").toAbsolutePath().resolve("mnist/train");
    Path testSamplesFolder = Path.of("").toAbsolutePath().resolve("mnist/test");

    if (!Files.exists(trainSamplesFolder)) {
      throw new RuntimeException(
          "Missing MNIST training folder: "
              + "download https://www.kaggle.com/datasets/alexanderyyy/mnist-png and unzip in the project root.");
    }

    List<Sample> trainDataset = MnistLoad.findSamplesInFolder(trainSamplesFolder);
    List<Sample> testDataset = MnistLoad.findSamplesInFolder(testSamplesFolder);

    Collections.shuffle(trainDataset);
    Collections.shuffle(testDataset);

    int batchSize = 64;
    int epochs = 25;
    int sampleIndex = 0;
    int trainDatasetSize = trainDataset.size();

    // y = a * x + b
    VariableTensor layer1Weights = matrix(28 * 28, 128).variable().normal(0.0, 1.0);
    VariableTensor layer1Bias = vector(128).variable().normal(0.0, 1.0);
    VariableTensor layer2Weights = matrix(128, 64).variable().normal(0.0, 1.0);
    VariableTensor layer2Bias = vector(64).variable().normal(0.0, 1.0);
    VariableTensor layer3Weights = matrix(64, 10).variable().normal(0.0, 1.0);
    VariableTensor layer3Bias = vector(10).variable().normal(0.0, 1.0);

    double minLoss = Double.MAX_VALUE;


    long initTime = System.currentTimeMillis();
    for (int epoch = 0; epoch < epochs; epoch++) {
      double averageLoss = 0.0;
      for (int round = 0; round < trainDatasetSize / batchSize; round++) {

        List<ComputedTensor> partialLosses = new ArrayList<>();
        int correctPredictions = 0;

        for (int i = 0; i < batchSize; i++) {

          Sample sample = trainDataset.get((sampleIndex++) % trainDataset.size());

          Tensor input = unsqueeze(0, sample.getTensor());
          ConstantTensor target = vector(10).withData(makeCategoryData(sample.getLabel()));

          ComputedTensor layer1Out = relu(sum(matmul(input, layer1Weights), layer1Bias));
          ComputedTensor layer2Out = relu(sum(matmul(layer1Out, layer2Weights), layer2Bias));
          ComputedTensor prediction = softmax(sum(matmul(layer2Out, layer3Weights), layer3Bias));

          ComputedTensor partialMseLoss = mse(target, prediction);

          if (findMaxCategory(prediction.getData()) == sample.getLabel()) {
            correctPredictions++;
          }

          partialLosses.add(partialMseLoss);
        }

        // Loss is the average of all the partial losses
        ComputedTensor mseLoss =
            multiply(sum(partialLosses.toArray(Tensor[]::new)), scalar(1.0 / batchSize));

        if (round % 10 == 0) {
          System.out.println("=== Epoch " + epoch + " Round " + round + "===");
          System.out.println(System.currentTimeMillis() - initTime);
          System.out.println(String.format("          Loss value: %f", mseLoss.get(0)));
          System.out.println(
              String.format(" Correct predictions: %d out of %d", correctPredictions, batchSize));
        }


        if (mseLoss.get(0) < 0.0000001) {

          System.out.println();
          System.out.println("=== Converged to solution in " + epoch + " epochs ===");

          System.out.println("## loss function stats: ");
          ComputationGraphStats.printComputationGraphStats(mseLoss);
          System.out.println("## computation graph:");
          ComputationGraphStats.printComputationGraph(mseLoss);
          break;
        }

        averageLoss += mseLoss.get(0) / (trainDatasetSize / batchSize);

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

      if (averageLoss < minLoss) {
        Files.createDirectories(Path.of("save"));
        layer1Weights.toFile(Path.of("save/layer1Weights.tensor"));
        layer1Bias.toFile(Path.of("save/layer1Bias.tensor"));
        layer2Weights.toFile(Path.of("save/layer2Weights.tensor"));
        layer2Bias.toFile(Path.of("save/layer2Bias.tensor"));
        layer3Weights.toFile(Path.of("save/layer3Weights.tensor"));
        layer3Bias.toFile(Path.of("save/layer3Bias.tensor"));
        minLoss = averageLoss;
      }
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
