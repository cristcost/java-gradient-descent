package net.cristcost.jtflow.operations.raw;

public class RawCommon {
  public static double clamp(double inputValue, double min, double max) {
    return Math.max(min, Math.min(max, inputValue));
  }

}
