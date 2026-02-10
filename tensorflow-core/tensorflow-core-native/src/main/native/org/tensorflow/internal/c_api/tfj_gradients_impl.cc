/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef _WIN32

#include <stdio.h>
#include <stdlib.h>

// IMPORTANT: explicit STL includes (this .cc is included via tfj_gradients.h in some builds,
// so we must not rely on transitive includes from other headers).
#include <string>
#include <unordered_map>
#include <vector>

#include "tfj_graph.h"
#include "tsl/platform/errors.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/cc/framework/grad_op_registry.h"

namespace tensorflow {
namespace java {

using namespace tsl;
using namespace std;

unordered_map<string, TFJ_GradFuncAdapter> g_grad_func_adapters;

// Cast helper (inspired by TF C-API)
template <typename T, typename U>
T* struct_cast(U* ptr) {
  return static_cast<T*>(static_cast<void*>(ptr));
}

// Bridge called by TF runtime when building gradients for op
Status CustomGradFunc(const Scope& scope,
                      const Operation& op,
                      const vector<Output>& grad_inputs,
                      vector<Output>* grad_outputs) {
  const string& op_type = op.node()->type_string();
  auto found_adapter = g_grad_func_adapters.find(op_type);
  if (found_adapter == g_grad_func_adapters.end()) {
    return errors::NotFound("No gradient adapter found for operation ", op_type);
  }

  TFJ_GradFuncAdapter adapter = found_adapter->second;
  if (adapter == nullptr) {
    return errors::Unknown("Null Java gradient adapter for op ", op_type);
  }

  const int num_inputs = static_cast<int>(grad_inputs.size());

  TF_Output* inputs = nullptr;
  if (num_inputs > 0) {
    inputs = static_cast<TF_Output*>(malloc(num_inputs * sizeof(TF_Output)));
    if (inputs == nullptr) {
      return errors::ResourceExhausted(
          "Out of memory allocating inputs for custom gradient of op ", op_type);
    }
  }

  for (int i = 0; i < num_inputs; ++i) {
    const Output& grad_input = grad_inputs[i];
    inputs[i].oper = struct_cast<TF_Operation>(grad_input.node());
    inputs[i].index = grad_input.index();
  }

  TF_Output* outputs = nullptr;

  LOG(INFO) << "Calling Java gradient function for operation of type " << op_type;
  const int num_outputs = adapter(
      static_cast<TFJ_GraphId>(scope.graph()),
      struct_cast<TFJ_Scope>(const_cast<Scope*>(&scope)),
      struct_cast<TF_Operation>(op.node()),
      inputs,
      num_inputs,
      &outputs);

  if (inputs != nullptr) free(inputs);

  // Adapter contract:
  // - num_outputs < 0 indicates failure
  // - num_outputs == 0: OK, outputs may be nullptr
  // - num_outputs > 0: outputs must be non-null
  if (num_outputs < 0) {
    if (outputs != nullptr) free(outputs);
    return errors::Unknown("Java custom gradient adapter failed for op ", op_type,
                           " (num_outputs=", num_outputs, ")");
  }
  if (num_outputs > 0 && outputs == nullptr) {
    return errors::Unknown("Java custom gradient adapter returned null outputs for op ",
                           op_type, " with num_outputs=", num_outputs);
  }

  grad_outputs->reserve(grad_outputs->size() + static_cast<size_t>(num_outputs));

  for (int i = 0; i < num_outputs; ++i) {
    const TF_Output out = outputs[i];

    // Convention: out.oper == nullptr => NoGradient
    if (out.oper == nullptr) {
      grad_outputs->push_back(Output());  // TF interprets empty Output as "no grad"
      continue;
    }

    grad_outputs->push_back(Output(struct_cast<Node>(out.oper), out.index));
  }

  if (outputs != nullptr) free(outputs);  // allocated from Java via malloc
  return OkStatus();
}

}  // namespace java
}  // namespace tensorflow

using namespace tensorflow::ops;
using namespace tensorflow::java;

bool TFJ_HasGradient(const char* op_type) {
  GradFunc dummy;
  tsl::Status status = GradOpRegistry::Global()->Lookup(op_type, &dummy);
  return status.ok();
}

bool TFJ_RegisterCustomGradient(const char* op_type, TFJ_GradFuncAdapter grad_func_adapter) {
  LOG(INFO) << "TFJ_RegisterCustomGradient(" << op_type << ") adapter_ptr="
            << reinterpret_cast<void*>(grad_func_adapter);

  if (grad_func_adapter == nullptr) {
    LOG(ERROR) << "Refusing to register NULL Java gradient adapter for op " << op_type;
    return false;
  }

  if (TFJ_HasGradient(op_type)) {
    LOG(WARNING) << "Tried to register Java gradient function for operation " << op_type
                 << ", which has already a registered function";
    return false;
  }

  bool registered = GradOpRegistry::Global()->Register(op_type, CustomGradFunc);
  if (registered) {
    g_grad_func_adapters.insert({op_type, grad_func_adapter});
  }
  return registered;
}

#else  // _WIN32

bool TFJ_HasGradient(const char* op_type) { return true; }
bool TFJ_RegisterCustomGradient(const char* op_type, TFJ_GradFuncAdapter grad_func_adapter) {
  return false;
}

#endif  // _WIN32
