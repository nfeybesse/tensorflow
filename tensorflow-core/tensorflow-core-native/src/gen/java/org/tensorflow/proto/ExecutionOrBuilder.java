// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/debug_event.proto

package org.tensorflow.proto;

public interface ExecutionOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.Execution)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Op type (e.g., "MatMul").
   * In the case of a Graph, this is the name of the Graph.
   * </pre>
   *
   * <code>string op_type = 1;</code>
   * @return The opType.
   */
  java.lang.String getOpType();
  /**
   * <pre>
   * Op type (e.g., "MatMul").
   * In the case of a Graph, this is the name of the Graph.
   * </pre>
   *
   * <code>string op_type = 1;</code>
   * @return The bytes for opType.
   */
  com.google.protobuf.ByteString
      getOpTypeBytes();

  /**
   * <pre>
   * Number of output tensors.
   * </pre>
   *
   * <code>int32 num_outputs = 2;</code>
   * @return The numOutputs.
   */
  int getNumOutputs();

  /**
   * <pre>
   * The graph that's executed: applicable only to the eager
   * execution of a FuncGraph.
   * </pre>
   *
   * <code>string graph_id = 3;</code>
   * @return The graphId.
   */
  java.lang.String getGraphId();
  /**
   * <pre>
   * The graph that's executed: applicable only to the eager
   * execution of a FuncGraph.
   * </pre>
   *
   * <code>string graph_id = 3;</code>
   * @return The bytes for graphId.
   */
  com.google.protobuf.ByteString
      getGraphIdBytes();

  /**
   * <pre>
   * IDs of the input tensors (if available).
   * </pre>
   *
   * <code>repeated int64 input_tensor_ids = 4;</code>
   * @return A list containing the inputTensorIds.
   */
  java.util.List<java.lang.Long> getInputTensorIdsList();
  /**
   * <pre>
   * IDs of the input tensors (if available).
   * </pre>
   *
   * <code>repeated int64 input_tensor_ids = 4;</code>
   * @return The count of inputTensorIds.
   */
  int getInputTensorIdsCount();
  /**
   * <pre>
   * IDs of the input tensors (if available).
   * </pre>
   *
   * <code>repeated int64 input_tensor_ids = 4;</code>
   * @param index The index of the element to return.
   * @return The inputTensorIds at the given index.
   */
  long getInputTensorIds(int index);

  /**
   * <pre>
   * IDs of the output tensors (if availbable).
   * If specified, must have the same length as tensor_protos.
   * </pre>
   *
   * <code>repeated int64 output_tensor_ids = 5;</code>
   * @return A list containing the outputTensorIds.
   */
  java.util.List<java.lang.Long> getOutputTensorIdsList();
  /**
   * <pre>
   * IDs of the output tensors (if availbable).
   * If specified, must have the same length as tensor_protos.
   * </pre>
   *
   * <code>repeated int64 output_tensor_ids = 5;</code>
   * @return The count of outputTensorIds.
   */
  int getOutputTensorIdsCount();
  /**
   * <pre>
   * IDs of the output tensors (if availbable).
   * If specified, must have the same length as tensor_protos.
   * </pre>
   *
   * <code>repeated int64 output_tensor_ids = 5;</code>
   * @param index The index of the element to return.
   * @return The outputTensorIds at the given index.
   */
  long getOutputTensorIds(int index);

  /**
   * <pre>
   * Type of the tensor value encapsulated in this proto.
   * </pre>
   *
   * <code>.tensorflow.TensorDebugMode tensor_debug_mode = 6;</code>
   * @return The enum numeric value on the wire for tensorDebugMode.
   */
  int getTensorDebugModeValue();
  /**
   * <pre>
   * Type of the tensor value encapsulated in this proto.
   * </pre>
   *
   * <code>.tensorflow.TensorDebugMode tensor_debug_mode = 6;</code>
   * @return The tensorDebugMode.
   */
  org.tensorflow.proto.TensorDebugMode getTensorDebugMode();

  /**
   * <pre>
   * Output Tensor values in the type described by `tensor_value_type`.
   * The length of this should match `num_outputs`.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorProto tensor_protos = 7;</code>
   */
  java.util.List<org.tensorflow.proto.TensorProto> 
      getTensorProtosList();
  /**
   * <pre>
   * Output Tensor values in the type described by `tensor_value_type`.
   * The length of this should match `num_outputs`.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorProto tensor_protos = 7;</code>
   */
  org.tensorflow.proto.TensorProto getTensorProtos(int index);
  /**
   * <pre>
   * Output Tensor values in the type described by `tensor_value_type`.
   * The length of this should match `num_outputs`.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorProto tensor_protos = 7;</code>
   */
  int getTensorProtosCount();
  /**
   * <pre>
   * Output Tensor values in the type described by `tensor_value_type`.
   * The length of this should match `num_outputs`.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorProto tensor_protos = 7;</code>
   */
  java.util.List<? extends org.tensorflow.proto.TensorProtoOrBuilder> 
      getTensorProtosOrBuilderList();
  /**
   * <pre>
   * Output Tensor values in the type described by `tensor_value_type`.
   * The length of this should match `num_outputs`.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorProto tensor_protos = 7;</code>
   */
  org.tensorflow.proto.TensorProtoOrBuilder getTensorProtosOrBuilder(
      int index);

  /**
   * <pre>
   * Stack trace of the eager execution.
   * </pre>
   *
   * <code>.tensorflow.CodeLocation code_location = 8;</code>
   * @return Whether the codeLocation field is set.
   */
  boolean hasCodeLocation();
  /**
   * <pre>
   * Stack trace of the eager execution.
   * </pre>
   *
   * <code>.tensorflow.CodeLocation code_location = 8;</code>
   * @return The codeLocation.
   */
  org.tensorflow.proto.CodeLocation getCodeLocation();
  /**
   * <pre>
   * Stack trace of the eager execution.
   * </pre>
   *
   * <code>.tensorflow.CodeLocation code_location = 8;</code>
   */
  org.tensorflow.proto.CodeLocationOrBuilder getCodeLocationOrBuilder();

  /**
   * <pre>
   * Debugged-generated IDs of the devices on which the output tensors reside.
   * To look up details about the device (e.g., name), cross-reference this
   * field with the DebuggedDevice messages.
   * </pre>
   *
   * <code>repeated int32 output_tensor_device_ids = 9;</code>
   * @return A list containing the outputTensorDeviceIds.
   */
  java.util.List<java.lang.Integer> getOutputTensorDeviceIdsList();
  /**
   * <pre>
   * Debugged-generated IDs of the devices on which the output tensors reside.
   * To look up details about the device (e.g., name), cross-reference this
   * field with the DebuggedDevice messages.
   * </pre>
   *
   * <code>repeated int32 output_tensor_device_ids = 9;</code>
   * @return The count of outputTensorDeviceIds.
   */
  int getOutputTensorDeviceIdsCount();
  /**
   * <pre>
   * Debugged-generated IDs of the devices on which the output tensors reside.
   * To look up details about the device (e.g., name), cross-reference this
   * field with the DebuggedDevice messages.
   * </pre>
   *
   * <code>repeated int32 output_tensor_device_ids = 9;</code>
   * @param index The index of the element to return.
   * @return The outputTensorDeviceIds at the given index.
   */
  int getOutputTensorDeviceIds(int index);
}