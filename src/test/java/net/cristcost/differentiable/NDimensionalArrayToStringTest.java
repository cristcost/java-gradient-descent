package net.cristcost.differentiable;

import static net.cristcost.differentiable.NDimensionalArray.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

class NDimensionalArrayToStringTest {

  ObjectMapper objectMapper = new ObjectMapper();

  @Test
  void test() throws JsonMappingException, JsonProcessingException {

    NDimensionalArray nd = NDimensionalArray.ndscalar(42);
    assertEquals(objectMapper.readTree("42.0"), objectMapper.readTree(nd.toString()));

    nd = new NDimensionalArray(data(42), shape(1));
    assertEquals(objectMapper.readTree("[42.0]"), objectMapper.readTree(nd.toString()));

    nd = new NDimensionalArray(data(3, 2, 1));
    assertEquals(objectMapper.readTree("[3.0, 2.0, 1.0]"), objectMapper.readTree(nd.toString()));

    nd = new NDimensionalArray(data(4, 3, 2, 1), shape(2, 2));
    assertEquals(objectMapper.readTree("[[4.0, 3.0], [2.0, 1.0]]"),
        objectMapper.readTree(nd.toString()));

    nd = new NDimensionalArray(data(9, 8, 7, 6, 5, 4, 3, 2, 1), shape(3, 3));
    assertEquals(objectMapper.readTree("[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]"),
        objectMapper.readTree(nd.toString()));


  }
}

