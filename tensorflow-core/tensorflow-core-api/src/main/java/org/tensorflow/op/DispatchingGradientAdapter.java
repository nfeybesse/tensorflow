package org.tensorflow.op;

import java.lang.reflect.Constructor;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.tensorflow.AbstractGradientAdapter;
import org.tensorflow.Graph;
import org.tensorflow.GraphOperation;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.internal.c_api.TFJ_Scope;

final class DispatchingGradientAdapter extends AbstractGradientAdapter {

  private final ConcurrentMap<String, RawCustomGradient> raw = new ConcurrentHashMap<>();
  private final ConcurrentMap<String, TypedEntry<?>> typed = new ConcurrentHashMap<>();

  static final class TypedEntry<T extends RawOpInputs<?>> {
    final CustomGradient<T> grad;
    final Class<T> inputClass;
    final Constructor<T> ctor;

    TypedEntry(CustomGradient<T> grad, Class<T> inputClass) {
      this.grad = grad;
      this.inputClass = inputClass;
      try {
        this.ctor = inputClass.getConstructor(org.tensorflow.GraphOperation.class);
      } catch (NoSuchMethodException e) {
        throw new IllegalArgumentException(
            "Inputs class " + inputClass.getName() + " must have a public ctor(GraphOperation).", e);
      }
    }
  }

  void putRaw(String opType, RawCustomGradient g) {
    raw.put(opType, g);
  }

  <T extends RawOpInputs<?>> void putTyped(String opType, CustomGradient<T> g, Class<T> inputClass) {
    typed.put(opType, new TypedEntry<>(g, inputClass));
  }

  @Override
  protected List<Operand<?>> apply(
      Graph graph, TFJ_Scope scope, GraphOperation operation, List<Output<?>> gradInputs) {

    final String opType = operation.type();

    RawCustomGradient rg = raw.get(opType);
    if (rg != null) {
      // NativeScope & Ops constructors are package-private => must be in org.tensorflow.op
      Scope nativeScope = new NativeScope(scope, graph, operation.name()).withSubScope(operation.name());
      return rg.call(new Ops(nativeScope), operation, gradInputs);
    }

    @SuppressWarnings("rawtypes")
    TypedEntry te = typed.get(opType);
    if (te != null) {
      return applyTyped(graph, scope, operation, gradInputs, te);
    }

    throw new IllegalStateException("No Java custom gradient registered for op type: " + opType);
  }

  private <T extends RawOpInputs<?>> List<Operand<?>> applyTyped(
      Graph graph, TFJ_Scope scope, GraphOperation operation, List<Output<?>> gradInputs, TypedEntry<T> te) {
    try {
      T inputs = te.ctor.newInstance(operation);
      Scope nativeScope = new NativeScope(scope, graph, operation.name()).withSubScope(operation.name());
      return te.grad.call(new Ops(nativeScope), inputs, gradInputs);
    } catch (ReflectiveOperationException e) {
      throw new RuntimeException("Failed to instantiate inputs for " + te.inputClass.getName(), e);
    }
  }
}
