package net.cristcost.jtflow.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

class TensorToStringTest {

  ObjectMapper objectMapper = new ObjectMapper();

  @Test
  void test() throws JsonMappingException, JsonProcessingException {

    // Tensor tensor = scalar().withData(42.0);
    Tensor tensor = scalar(42.0);
    assertEquals(objectMapper.readTree("42.0"), objectMapper.readTree(tensor.json()));

    tensor = vector(42.0);
    assertEquals(objectMapper.readTree("[42.0]"), objectMapper.readTree(tensor.json()));

    tensor = vector(3.0, 2.0, 1.0);
    assertEquals(objectMapper.readTree("[3.0, 2.0, 1.0]"),
        objectMapper.readTree(tensor.json()));

    tensor = matrix(2, 2, 4.0, 3.0, 2.0, 1.0);
    assertEquals(objectMapper.readTree("[[4.0, 3.0], [2.0, 1.0]]"),
        objectMapper.readTree(tensor.json()));

    tensor = matrix(3, 3, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    assertEquals(objectMapper.readTree("[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]"),
        objectMapper.readTree(tensor.json()));


  }

  private static Tensor matrix(int rows, int cols, final double... data) {
    return new Tensor() {

      @Override
      public int[] getShape() {
        return new int[] {rows, cols};
      }

      @Override
      public double[] getData() {
        return data;
      }
    };
  }

  private static Tensor vector(final double... data) {
    return new Tensor() {

      @Override
      public int[] getShape() {
        return new int[] {data.length};
      }

      @Override
      public double[] getData() {
        return data;
      }
    };
  }

  private static Tensor scalar(final double data) {
    return new Tensor() {

      @Override
      public int[] getShape() {
        return new int[0];
      }

      @Override
      public double[] getData() {
        return new double[] {data};
      }
    };

  }
}

