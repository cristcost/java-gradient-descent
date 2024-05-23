package net.cristcost.mnist;

import static net.cristcost.differentiable.MathLibrary.vector;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.imageio.ImageIO;
import net.cristcost.differentiable.ConstantTensor;
import net.cristcost.differentiable.Tensor;

public class MnistLoad {

  public static void main(String[] args) throws IOException {

    // long initTime = System.currentTimeMillis();
    List<Sample> trainDataset = findSamplesInFolder(
        Path.of(".").toAbsolutePath().getParent().resolve("mnist/test"));
    List<Sample> testDataset = findSamplesInFolder(
        Path.of(".").toAbsolutePath().getParent().resolve("mnist/train"));

    System.out.println("Found " + trainDataset.size() + " samples in the train dataset");
    System.out.println("Found " + testDataset.size() + " samples in the test dataset");

    // System.out.println("Time " + (System.currentTimeMillis() - initTime));
    Collections.shuffle(trainDataset);
    Collections.shuffle(testDataset);
    // System.out.println("Time " + (System.currentTimeMillis() - initTime));


    System.out.println(trainDataset.stream().limit(50)
        .map(Sample::getLabel)
        .map(l -> Integer.toString(l))
        .collect(Collectors.joining(",")));

    System.out.println(testDataset.stream().limit(50)
        .map(Sample::getLabel)
        .map(l -> Integer.toString(l))
        .collect(Collectors.joining(",")));

  }

  public static List<Sample> findSamplesInFolder(Path folder) throws IOException {
    return findSamplesInFolderAsStream(folder)
        .collect(Collectors.toList());

  }

  public static Stream<Sample> findSamplesInFolderAsStream(Path folder) throws IOException {
    // note: case sensitive extension and fixed layout, designed to work with mnist_png layout
    return Files.walk(folder)
        .filter(f -> !Files.isDirectory(f))
        .filter(f -> f.getFileName().toString().endsWith(".png"))
        .map(MnistLoad::sampleFromMnistPath);
  }

  public static Sample sampleFromMnistPath(Path mnistPath) {
    Path imageAbsolutePath = mnistPath.toAbsolutePath();
    if (!Files.exists(imageAbsolutePath)) {
      throw new RuntimeException("File " + imageAbsolutePath + " does not exists");
    }

    final int label = Integer.parseInt(mnistPath.getParent().getFileName().toString());

    return new Sample() {
      Tensor tensor = null;

      @Override
      public Tensor getTensor() {
        if (tensor == null) {
          try {
            tensor = vectorFromImgPath(imageAbsolutePath);
          } catch (IOException e) {
            throw new RuntimeException("Cannot lazily load image at " + imageAbsolutePath, e);
          }
        }
        return tensor;
      }

      @Override
      public int getLabel() {
        return label;
      }
    };
  }

  public static ConstantTensor vectorFromImgPath(Path imagePath) throws IOException {
    BufferedImage img = ImageIO.read(imagePath.toFile());
    DataBufferByte dataBuffer = (DataBufferByte) img.getData().getDataBuffer();
    double[] data = new double[dataBuffer.getSize()];
    for (int i = 0; i < data.length; i++) {
      data[i] = dataBuffer.getElemDouble(i) / 255.0;
    }
    ConstantTensor vector = vector(data);
    return vector;
  }



}
