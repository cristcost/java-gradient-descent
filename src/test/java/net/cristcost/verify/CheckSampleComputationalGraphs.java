package net.cristcost.verify;

import static net.cristcost.differentiable.MathLibrary.*;
import net.cristcost.differentiable.ComputationGraphStats;
import net.cristcost.differentiable.Scalar;

public class CheckSampleComputationalGraphs {

  public static Scalar fibonacci(int n) {
    if (n > 1) {
      return sum(fibonacci(n - 1), fibonacci(n - 2));
    } else {
      return constant(n);
    }
  }

  public static Scalar factorial(int n) {
    if (n > 1) {
      return multiply(constant(n), factorial(n - 1));
    } else {
      return constant(1);
    }
  }

  public static Scalar reluQuadraticFunction(double value) {
    Scalar x = constant(value);

    return relu(sum(
        multiply((constant(-0.5)), pow(x, constant(2.0))),
        multiply(constant(2.0), x),
        constant(6.0)));
  }

  public static void main(String[] args) {
    System.out.println("### factorial: `10!`");
    Scalar factorial = factorial(10);
    ComputationGraphStats.printComputationGraphStats(factorial);
    System.out.println("## computation graph:");
    ComputationGraphStats.printComputationGraph(factorial);
    System.out.println();

    System.out.println("### fibonacci: `fib(7)`");
    Scalar fibonacci = fibonacci(7);
    ComputationGraphStats.printComputationGraphStats(fibonacci);
    System.out.println("## computation graph:");
    ComputationGraphStats.printComputationGraph(fibonacci);
    System.out.println();

    System.out.println("### rectified quadratic: `relu(0.5x^2 - 2x -6)` in x=2.0");
    Scalar reluQuadraticFunction = reluQuadraticFunction(2.0);
    ComputationGraphStats.printComputationGraphStats(reluQuadraticFunction);
    System.out.println("## computation graph:");
    ComputationGraphStats.printComputationGraph(reluQuadraticFunction);
    System.out.println();
  }

}
