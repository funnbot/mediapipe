// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/calculators/tensor/inference_calculator.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/tool/subgraph_expansion.h"

namespace mediapipe {
namespace api2 {

class InferenceCalculatorSelectorImpl
    : public SubgraphImpl<InferenceCalculatorSelector,
                          InferenceCalculatorSelectorImpl> {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const CalculatorGraphConfig::Node& subgraph_node) {
    const auto& options =
        Subgraph::GetOptions<mediapipe::InferenceCalculatorOptions>(
            subgraph_node);
    std::vector<absl::string_view> impls;

    const bool should_use_gpu =
        !options.has_delegate() ||  // Use GPU delegate if not specified
        (options.has_delegate() && options.delegate().has_gpu());
    const bool should_use_onnx = 
      options.has_delegate() && options.delegate().has_gpu() && options.delegate().gpu().has_api() &&
      options.delegate().gpu().api() == mediapipe::InferenceCalculatorOptions::Delegate::Gpu::ONNX;
    if (should_use_onnx) {
      impls.emplace_back("Onnx");
    } else if (should_use_gpu) {
      impls.emplace_back("Metal");
      impls.emplace_back("Gl");
      impls.emplace_back("Cpu");
    } 
    impls.emplace_back("Cpu");
    for (const auto& suffix : impls) {
      const auto impl = absl::StrCat("InferenceCalculator", suffix);
      if (!mediapipe::CalculatorBaseRegistry::IsRegistered(impl)) continue;
      CalculatorGraphConfig::Node impl_node = subgraph_node;
      impl_node.set_calculator(impl);
      return tool::MakeSingleNodeGraph(std::move(impl_node));
    }
    return absl::UnimplementedError("no implementation available");
  }
};

absl::StatusOr<Packet<TfLiteModelPtr>> InferenceCalculator::GetModelAsPacket(
    CalculatorContext* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  if (!options.model_path().empty()) {
    return TfLiteModelLoader::LoadFromPath(options.model_path());
  }
  if (!kSideInModel(cc).IsEmpty()) return kSideInModel(cc);
  return absl::Status(mediapipe::StatusCode::kNotFound,
                      "Must specify TFLite model as path or loaded model.");
}

}  // namespace api2
}  // namespace mediapipe
