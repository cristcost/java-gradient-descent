package net.cristcost.jtflow.api;

import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Function;


/**
 * Note: experimental API to ease casting syntax;
 * 
 * - I like how it allow to write the code to apply the logic if the class implement the specific
 * type
 * 
 * - I don't like it creates a circular dependencies (Projector uses -> Broadcastable implements ->
 * Tensor implements -> Projector)
 * 
 * Maybe I'll remove this in the future
 */
public interface Projector {

  default void ifBroadcastable(Consumer<Broadcastable> function) {
    if (this instanceof Broadcastable) {
      function.accept((Broadcastable) this);
    }
  }

  default <T> Optional<T> mapBroadcastable(Function<Broadcastable, T> function) {
    if (this instanceof Broadcastable) {
      Broadcastable broadcastable = (Broadcastable) this;
      return Optional.of(function.apply(broadcastable));
    } else {
      return Optional.empty();
    }
  }

  default void ifChainable(Consumer<Chainable> function) {
    if (this instanceof Chainable) {
      function.accept((Chainable) this);
    }
  }

  default <T> Optional<T> mapChainable(Function<Chainable, T> function) {
    if (this instanceof Chainable) {
      Chainable broadcastable = (Chainable) this;
      return Optional.of(function.apply(broadcastable));
    } else {
      return Optional.empty();
    }
  }

  default void ifDifferentiable(Consumer<Differentiable> function) {
    if (this instanceof Differentiable) {
      function.accept((Differentiable) this);
    }
  }

  default <T> Optional<T> mapDifferentiable(Function<Differentiable, T> function) {
    if (this instanceof Differentiable) {
      Differentiable broadcastable = (Differentiable) this;
      return Optional.of(function.apply(broadcastable));
    } else {
      return Optional.empty();
    }
  }
}
