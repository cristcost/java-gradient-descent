package net.cristcost.jtflow.file;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.BiFunction;

public class FileUtils {

  public static <R> R fromFile(Path file, Class<R> resultType,
      BiFunction<int[], double[], R> fromFileFunction) throws IOException {

    try (DataInputStream dis = new DataInputStream(Files.newInputStream(file))) {
      int shapeSize = dis.readInt();
      int dataSize = dis.readInt();

      final int[] shape = new int[shapeSize];
      for (int i = 0; i < shapeSize; i++) {
        shape[i] = dis.readInt();
      }

      final double[] data = new double[dataSize];
      for (int i = 0; i < dataSize; i++) {
        data[i] = dis.readDouble();
      }

      return fromFileFunction.apply(shape, data);
    }
  }

  public static void saveToFile(Path file, int[] shape, double[] data) throws IOException {
    try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(file))) {
      dos.writeInt(shape.length);
      dos.writeInt(data.length);
      for (int i : shape) {
        dos.writeInt(i);
      }

      for (double d : data) {
        dos.writeDouble(d);
      }
    }
  }

}
