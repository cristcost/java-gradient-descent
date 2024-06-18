package net.cristcost.jtflow.api;

import java.util.Optional;
import java.util.function.Function;

public interface Projector {

  default <T> Optional<T> broadcastable(Function<Broadcastable, T> function) {
    if (this instanceof Broadcastable) {
      Broadcastable broadcastable = (Broadcastable) this;
      return Optional.of(function.apply(broadcastable));
    } else {
      return Optional.empty();
    }
  }

  default <T> Optional<T> chainable(Function<Chainable, T> function) {
    if (this instanceof Chainable) {
      Chainable broadcastable = (Chainable) this;
      return Optional.of(function.apply(broadcastable));
    } else {
      return Optional.empty();
    }
  }

  default <T> Optional<T> differentiable(Function<Differentiable, T> function) {
    if (this instanceof Differentiable) {
      Differentiable broadcastable = (Differentiable) this;
      return Optional.of(function.apply(broadcastable));
    } else {
      return Optional.empty();
    }
  }

}
