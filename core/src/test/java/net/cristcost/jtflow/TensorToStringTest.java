package net.cristcost.jtflow;

import static net.cristcost.jtflow.JTFlow.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import net.cristcost.jtflow.api.Tensor;

class TensorToStringTest {

  ObjectMapper objectMapper = new ObjectMapper();

  @Test
  void test() throws JsonMappingException, JsonProcessingException {

    Tensor tensor = scalar().withData(42.0);
    assertEquals(objectMapper.readTree("42.0"), objectMapper.readTree(tensor.json()));

    tensor = vector(1).withData(42.0);
    assertEquals(objectMapper.readTree("[42.0]"), objectMapper.readTree(tensor.json()));

    tensor = vector(3).withData(3, 2, 1);
    assertEquals(objectMapper.readTree("[3.0, 2.0, 1.0]"),
        objectMapper.readTree(tensor.json()));

    tensor = matrix(2, 2).withData(4, 3, 2, 1);
    assertEquals(objectMapper.readTree("[[4.0, 3.0], [2.0, 1.0]]"),
        objectMapper.readTree(tensor.json()));

    tensor = tensor(3, 3).withData(9, 8, 7, 6, 5, 4, 3, 2, 1);
    assertEquals(objectMapper.readTree("[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]"),
        objectMapper.readTree(tensor.json()));


  }
}

