package net.cristcost.jtflow.dataset;

import net.cristcost.jtflow.api.Tensor;


public interface Sample {

  int getLabel();

  Tensor getTensor();
}
