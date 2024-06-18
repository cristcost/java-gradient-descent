package net.cristcost.jtflow.builder;

import java.io.IOException;
import java.nio.file.Path;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.file.FileUtils;

public class TensorLoader {
  public static Tensor fromFile(Path file) throws IOException {
    return FileUtils.fromFile(file, Tensor.class, (shape, data) -> new Tensor() {
      @Override
      public double[] getData() {
        return data;
      }

      @Override
      public int[] getShape() {
        return shape;
      }
    });
  }

  public static void toFile(Tensor tensor, Path file) throws IOException {
    FileUtils.saveToFile(file, tensor.getShape(), tensor.getData());
  }
}
