package org.tensorflow.op;

import org.tensorflow.internal.c_api.TFJ_GradFuncAdapter;

/** Public bridge to a single native gradient adapter. */
public final class GradientDispatch {

  // package-private adapter that can access NativeScope/Ops constructors
  static final DispatchingGradientAdapter ADAPTER = new DispatchingGradientAdapter();

  private GradientDispatch() {}

  public static TFJ_GradFuncAdapter adapter() {
    return ADAPTER;
  }

  public static void putRaw(String opType, RawCustomGradient gradient) {
    ADAPTER.putRaw(opType, gradient);
  }

  public static <T extends RawOpInputs<?>> void putTyped(
      String opType, CustomGradient<T> gradient, Class<T> inputClass) {
    ADAPTER.putTyped(opType, gradient, inputClass);
  }
}
