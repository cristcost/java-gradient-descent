package net.cristcost.differentiable;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class ComputationGraphStats {

  private final Map<Operation, Integer> operationStats = new HashMap<>();
  private final Set<Scalar> uniqueOperands = new HashSet<>();
  private int duplicated = 0;
  private int constants = 0;
  private int deepestLevel = 0;

  private ComputationGraphStats(Scalar scalar) {
    exploreScalar(scalar, 0);
  }

  private void exploreScalar(Scalar scalar, int level) {

    if (level > deepestLevel) {
      deepestLevel = level;
    }

    if (uniqueOperands.add(scalar)) {
      if (scalar instanceof ComputedScalar) {
        ComputedScalar computedScalar = (ComputedScalar) scalar;

        Computation computation = computedScalar.getFromComputation();

        operationStats.put(computation.getOperation(),
            operationStats.getOrDefault(computation.getOperation(), 0) + 1);

        for (Scalar operand : computation.getOperands()) {
          exploreScalar(operand, level + 1);
        }
      } else {
        constants++;
      }
    } else {
      duplicated++;
    }
  }

  public static void printComputationGraphStats(Scalar scalar) {
    ComputationGraphStats stats = new ComputationGraphStats(scalar);

    System.out.println(stats.uniqueOperands.size() + " unique operands");
    System.out.println(stats.constants + " of which are constants");
    System.out.println(stats.duplicated + " duplicated operands");
    System.out.println(stats.deepestLevel + " is the longest chain of operations");

    for (Entry<Operation, Integer> entry : stats.operationStats.entrySet()) {
      System.out.println(entry.getKey().name() + ": " + entry.getValue());
    }

  }

  public static void printComputationGraph(Scalar scalar) {
    printComputationGraph(scalar, 0);
  }

  private static void printComputationGraph(Scalar scalar, int indentationLevel) {

    String indentation = " ".repeat(indentationLevel);

    if (scalar instanceof ComputedScalar) {
      ComputedScalar computedScalar = (ComputedScalar) scalar;

      Computation computation = computedScalar.getFromComputation();

      System.out.println(String.format("%s%s %.2f = %s:",
          indentation,
          indentationLevel > 0 ? "{" : "",
          computedScalar.getValue(),
          computation.getOperation().name()));
      for (Scalar operand : computation.getOperands()) {
        printComputationGraph(operand, indentationLevel + 2);
      }

    } else {
      System.out.println(String.format("%s%s %.2f",
          indentation,
          indentationLevel > 0 ? "|" : "",
          scalar.getValue()));
    }
  }
}
