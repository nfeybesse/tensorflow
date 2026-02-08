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
#include <vector>
#include <unordered_map>

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

        /// This method can be used to cast a pointer to/from a C struct that contains only that pointer. It is a bit
        /// It has been "inspired" by the TensorFlow C API code, as found at this location when time of writing:
        /// https://github.com/tensorflow/tensorflow/blob/9d637f69f699c0c422716b56153a8b27b681891a/tensorflow/c/c_api.cc#L658
        template <typename T, typename U> T* struct_cast(U* ptr) {
            return static_cast<T*>(static_cast<void*>(ptr));
        }

        /// This function is called by the TensorFlow runtime when it is time to add gradient operations of `op` to the
        /// graph using the given `scope`.
        /// We use it as a bridge between the C++ signature in TensorFlow (tensorflow::op::GradFunc) and our custom
        /// "C" version (TFJ_GradFuncAdapter).
        Status CustomGradFunc(const Scope& scope,
                              const Operation& op,
                              const vector<Output>& grad_inputs,
                              vector<Output>* grad_outputs)
        {
            const string& op_type = op.node()->type_string();
            auto found_adapter = g_grad_func_adapters.find(op_type);
            if (found_adapter == g_grad_func_adapters.end()) {
                return errors::NotFound("No gradient adapter found for operation ", op_type);
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

            TFJ_GradFuncAdapter adapter = found_adapter->second;
            if (adapter == nullptr) {
                if (inputs != nullptr) free(inputs);
                return errors::Unknown("Null Java gradient adapter for op ", op_type);
            }
            LOG(INFO) << "Adapter ptr for " << op_type << " = " << reinterpret_cast<void*>(found_adapter->second);
            const int num_outputs = adapter(
                static_cast<TFJ_GraphId>(scope.graph()),
                struct_cast<TFJ_Scope>(const_cast<Scope*>(&scope)),
                struct_cast<TF_Operation>(op.node()),
                inputs,
                num_inputs,
                &outputs
            );

            // Always free inputs, even on error paths.
            if (inputs != nullptr) free(inputs);

            // Adapter contract hardening:
            // - On Java exception / failure, adapter should return negative or outputs==nullptr.
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

                // "NoGradient" sentinel from Java: TF_Output.oper == nullptr
                if (out.oper == nullptr) {
                    // Represent "no gradient" as an empty Output.
                    // TF's gradient builder should tolerate missing gradients for non-differentiable inputs.
                    grad_outputs->push_back(Output());
                    continue;
                }

                grad_outputs->push_back(Output(struct_cast<Node>(out.oper), out.index));
            }

            if (outputs != nullptr) free(outputs);
            return OkStatus();
        }
    }
}

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

    if (TFJ_HasGradient(op_type)) { // Check if gradient already exists otherwise the JVM might abort/crash
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

#else // #ifndef _WIN32

/* This extension is not available on Windows */

bool TFJ_HasGradient(const char* op_type) { return true; }
bool TFJ_RegisterCustomGradient(const char* op_type, TFJ_GradFuncAdapter grad_func_adapter) { return false; }

#endif // #ifndef _WIN32
