package net.cristcost.jtflow.samples.mnist;

import static net.cristcost.jtflow.JTFlow.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.dataset.Sample;
import net.cristcost.jtflow.optimizer.SgdWithMomentumOptimizer;
import net.cristcost.jtflow.tensors.ComputedTensor;
import net.cristcost.jtflow.tensors.ConstantTensor;
import net.cristcost.jtflow.tensors.VariableTensor;

public class Mnist {

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

    boolean load = Arrays.stream(args).filter(arg -> arg.equals("--load")).findAny().isPresent();

    final VariableTensor layer1Weights =
        load ? matrix(28 * 28, 128).variable()
            .clone(Tensor.fromFile(Path.of("save/layer1Weights.tensor")))
            : matrix(28 * 28, 128).variable().kaimingUniform();
    final VariableTensor layer1Bias =
        load ? vector(128).variable().clone(Tensor.fromFile(Path.of("save/layer1Bias.tensor")))
            : vector(128).variable().zeros();
    final VariableTensor layer2Weights =
        load ? matrix(128, 64).variable()
            .clone(Tensor.fromFile(Path.of("save/layer2Weights.tensor")))
            : matrix(128, 64).variable().kaimingUniform();
    final VariableTensor layer2Bias =
        load ? vector(64).variable().clone(Tensor.fromFile(Path.of("save/layer2Bias.tensor")))
            : vector(64).variable().zeros();
    final VariableTensor layer3Weights =
        load ? matrix(64, 10).variable()
            .clone(Tensor.fromFile(Path.of("save/layer3Weights.tensor")))
            : matrix(64, 10).variable().kaimingUniform();
    final VariableTensor layer3Bias =
        load ? vector(10).variable().clone(Tensor.fromFile(Path.of("save/layer3Bias.tensor")))
            : vector(10).variable().zeros();

    layer1Weights.setOptimizer(new SgdWithMomentumOptimizer(0.01, 0.9));
    layer1Bias.setOptimizer(new SgdWithMomentumOptimizer(0.01, 0.9));
    layer2Weights.setOptimizer(new SgdWithMomentumOptimizer(0.01, 0.9));
    layer2Bias.setOptimizer(new SgdWithMomentumOptimizer(0.01, 0.9));
    layer3Weights.setOptimizer(new SgdWithMomentumOptimizer(0.01, 0.9));
    layer3Bias.setOptimizer(new SgdWithMomentumOptimizer(0.01, 0.9));
    // layer1Weights.setOptimizer(new RootMeanSquarePropagationOptimizer(0.01, 0.9));
    // layer1Bias.setOptimizer(new RootMeanSquarePropagationOptimizer(0.01, 0.9));
    // layer2Weights.setOptimizer(new RootMeanSquarePropagationOptimizer(0.01, 0.9));
    // layer2Bias.setOptimizer(new RootMeanSquarePropagationOptimizer(0.01, 0.9));
    // layer3Weights.setOptimizer(new RootMeanSquarePropagationOptimizer(0.01, 0.9));
    // layer3Bias.setOptimizer(new RootMeanSquarePropagationOptimizer(0.01, 0.9));

    double minLoss = Double.MAX_VALUE;


    long initTime = System.currentTimeMillis();
    long partialTime = System.currentTimeMillis();

    for (int epoch = 0; epoch < epochs; epoch++) {
      double epochLoss = 0.0;
      int epochCorrect = 0;
      int epochSamples = 0;

      System.out.println();
      System.out.println("### Epoch " + epoch + " ####");

      for (int round = 0; round < trainDatasetSize / batchSize; round++) {

        List<ComputedTensor> partialLosses = new ArrayList<>();
        int correctPredictions = 0;

        for (int i = 0; i < batchSize; i++) {

          Sample sample = trainDataset.get((sampleIndex++) % trainDataset.size());

          Tensor input = unsqueeze(0, sample.getTensor());
          ConstantTensor oneHotEncodedLabel =
              vector(10).withData(makeOneHotEncodedData(sample.getLabel()));

          ComputedTensor layer1Out = relu(sum(matmul(input, layer1Weights), layer1Bias));
          ComputedTensor layer2Out = relu(sum(matmul(layer1Out, layer2Weights), layer2Bias));
          ComputedTensor prediction = sum(matmul(layer2Out, layer3Weights), layer3Bias);

          ComputedTensor partialCategoricalCrossEntropyLoss =
              categoricalCrossentropy(prediction, oneHotEncodedLabel);

          if (findMaxCategory(prediction.getData()) == sample.getLabel()) {
            correctPredictions++;
            epochCorrect++;
          }
          epochSamples++;

          partialLosses.add(partialCategoricalCrossEntropyLoss);
        }

        // Loss is the average of all the partial losses
        ComputedTensor categoricalCrossentropyLoss =
            multiply(sum(partialLosses.toArray(Tensor[]::new)), scalar(1.0 / batchSize));

        if (round == 10 || round == 50 || round == 100 || round == 500
            || round == trainDatasetSize - 1) {
          System.out.println("=== Epoch " + epoch + " Round " + round + "===");
          System.out
              .println("    Elapsed seconds: " + (System.currentTimeMillis() - initTime) / 1000);
          System.out.println(
              String.format("          Loss value: %f", categoricalCrossentropyLoss.get(0)));
          System.out.println(
              String.format(" Correct predictions: %d out of %d", correctPredictions, batchSize));
        }

        if (System.currentTimeMillis() - partialTime > 30000) {
          partialTime += 30000;
          System.err
              .println("                   ( " + (System.currentTimeMillis() - initTime) / 1000
                  + " sec, round " + round + ")");
        }

        epochLoss += categoricalCrossentropyLoss.get(0) / (trainDatasetSize / batchSize);

        categoricalCrossentropyLoss.startBackpropagation();

        layer1Weights.optimize();
        layer1Bias.optimize();
        layer2Weights.optimize();
        layer2Bias.optimize();
        layer3Weights.optimize();
        layer3Bias.optimize();
      }

      // if (categoricalCrossentropyLoss.get(0) < 0.0000001) {
      //
      // System.out.println();
      // System.out.println("=== Converged to solution in " + epoch + " epochs ===");
      //
      // System.out.println("## loss function stats: ");
      // ComputationGraphStats.printComputationGraphStats(categoricalCrossentropyLoss);
      // System.out.println("## computation graph:");
      // ComputationGraphStats.printComputationGraph(categoricalCrossentropyLoss);
      // break;
      // }
      if (epochLoss < minLoss) {
        Files.createDirectories(Path.of("save"));

        Files.write(Path.of("save/info.txt"), List.of(
            "Average loss of dataset: " + epochLoss,
            "      Samples predicted: " + epochSamples,
            "    Correct predictions: " + epochCorrect));

        layer1Weights.toFile(Path.of("save/layer1Weights.tensor"));
        layer1Bias.toFile(Path.of("save/layer1Bias.tensor"));
        layer2Weights.toFile(Path.of("save/layer2Weights.tensor"));
        layer2Bias.toFile(Path.of("save/layer2Bias.tensor"));
        layer3Weights.toFile(Path.of("save/layer3Weights.tensor"));
        layer3Bias.toFile(Path.of("save/layer3Bias.tensor"));
        minLoss = epochLoss;
      }
      System.out.println("         Epoch Loss value: " + epochLoss);
      System.out.println("Epoch Correct predictions: " + epochCorrect + " out of " + epochSamples);

      if ((double) epochCorrect / epochSamples > 0.90) {
        break;
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

  private static double[] makeOneHotEncodedData(int label) {
    double[] ret = new double[10];
    ret[label] = 1.0;
    return ret;
  }
}
