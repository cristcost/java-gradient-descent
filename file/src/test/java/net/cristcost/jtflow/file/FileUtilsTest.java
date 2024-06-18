package net.cristcost.jtflow.file;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import java.io.IOException;
import java.nio.file.Path;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class FileUtilsTest {

  @Test
  void testSimpleSaveAndLoad(@TempDir Path tempDir) throws IOException {
    int[] shape = {1, 2, 3};
    double[] data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    Path filePath = tempDir.resolve("sample.dat");

    FileUtils.saveToFile(filePath, shape, data);

    Object[] result = FileUtils.fromFile(filePath, Object[].class, (s, d) -> new Object[] {s, d});

    Assertions.assertInstanceOf(int[].class, result[0]);
    Assertions.assertInstanceOf(double[].class, result[1]);
    assertArrayEquals(shape, (int[]) result[0]);
    assertArrayEquals(data, (double[]) result[1]);

  }

}
