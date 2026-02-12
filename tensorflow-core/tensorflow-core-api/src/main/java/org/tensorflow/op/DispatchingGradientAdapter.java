/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================
*/
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

  private static String dupMsg(String opType, String existingKind, String newKind) {
    return "A "
        + existingKind
        + " gradient is already registered for op type '"
        + opType
        + "'. Raw and typed registrations are mutually exclusive; cannot register "
        + newKind
        + ".";
  }

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
            "Inputs class " + inputClass.getName() + " must have a public ctor(GraphOperation).",
            e);
      }
    }
  }

  void putRaw(String opType, RawCustomGradient g) {
    if (typed.containsKey(opType)) {
      throw new IllegalStateException(dupMsg(opType, "typed", "raw"));
    }
    RawCustomGradient prev = raw.putIfAbsent(opType, g);
    if (prev != null) {
      throw new IllegalStateException(
          "A raw gradient is already registered for op type '" + opType + "'.");
    }
  }

  <T extends RawOpInputs<?>> void putTyped(
      String opType, CustomGradient<T> g, Class<T> inputClass) {
    if (raw.containsKey(opType)) {
      throw new IllegalStateException(dupMsg(opType, "raw", "typed"));
    }
    TypedEntry<?> prev = typed.putIfAbsent(opType, new TypedEntry<>(g, inputClass));
    if (prev != null) {
      throw new IllegalStateException(
          "A typed gradient is already registered for op type '" + opType + "'.");
    }
  }

  @Override
  protected List<Operand<?>> apply(
      Graph graph, TFJ_Scope scope, GraphOperation operation, List<Output<?>> gradInputs) {

    final String opType = operation.type();

    RawCustomGradient rg = raw.get(opType);
    if (rg != null) {
      // NativeScope & Ops constructors are package-private => must be in org.tensorflow.op
      Scope nativeScope =
          new NativeScope(scope, graph, operation.name()).withSubScope(operation.name());
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
      Graph graph,
      TFJ_Scope scope,
      GraphOperation operation,
      List<Output<?>> gradInputs,
      TypedEntry<T> te) {
    try {
      T inputs = te.ctor.newInstance(operation);
      Scope nativeScope =
          new NativeScope(scope, graph, operation.name()).withSubScope(operation.name());
      return te.grad.call(new Ops(nativeScope), inputs, gradInputs);
    } catch (ReflectiveOperationException e) {
      throw new RuntimeException("Failed to instantiate inputs for " + te.inputClass.getName(), e);
    }
  }
}
