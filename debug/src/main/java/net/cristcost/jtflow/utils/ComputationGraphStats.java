package net.cristcost.jtflow.utils;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.api.operations.Computation;
import net.cristcost.jtflow.api.operations.Operation;
import net.cristcost.jtflow.tensors.ComputedTensor;

public class ComputationGraphStats {

  private final Map<Operation, Integer> operationStats = new HashMap<>();
  private final Set<Tensor> uniqueOperands = new HashSet<>();
  private int duplicated = 0;
  private int constants = 0;
  private int deepestLevel = 0;

  private ComputationGraphStats(Tensor tensor) {
    exploreTensor(tensor, 0);
  }

  private void exploreTensor(Tensor tensor, int level) {

    if (level > deepestLevel) {
      deepestLevel = level;
    }

    if (uniqueOperands.add(tensor)) {
      if (tensor instanceof ComputedTensor) {
        ComputedTensor computedTensor = (ComputedTensor) tensor;

        Computation computation = computedTensor.getFromComputation();

        operationStats.put(computation.getOperation(),
            operationStats.getOrDefault(computation.getOperation(), 0) + 1);

        for (Tensor operand : computation.getOperands()) {
          exploreTensor(operand, level + 1);
        }
      } else {
        constants++;
      }
    } else {
      duplicated++;
    }
  }

  public static void printComputationGraphStats(Tensor tensor) {
    ComputationGraphStats stats = new ComputationGraphStats(tensor);

    System.out.println(stats.uniqueOperands.size() + " unique operands");
    System.out.println(stats.constants + " of which are constants");
    System.out.println(stats.duplicated + " duplicated operands");
    System.out.println(stats.deepestLevel + " is the longest chain of operations");

    for (Entry<Operation, Integer> entry : stats.operationStats.entrySet()) {
      System.out.println(entry.getKey().name() + ": " + entry.getValue());
    }

  }

  public static void printComputationGraph(Tensor tensor) {
    printComputationGraph(tensor, 0);
  }

  private static void printComputationGraph(Tensor tensor, int indentationLevel) {

    String indentation = " ".repeat(indentationLevel);

    if (tensor instanceof ComputedTensor) {
      ComputedTensor computedTensor = (ComputedTensor) tensor;

      Computation computation = computedTensor.getFromComputation();

      System.out.println(String.format("%s%s %s = %s:",
          indentation,
          indentationLevel > 0 ? "{" : "",
          computedTensor.json(),
          computation.getOperation().name()));
      for (Tensor operand : computation.getOperands()) {
        printComputationGraph(operand, indentationLevel + 2);
      }

    } else {
      System.out.println(String.format("%s%s %s",
          indentation,
          indentationLevel > 0 ? "|" : "",
          tensor.json()));
    }
  }
}
