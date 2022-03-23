#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/framework/deps/file_path.h"

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

namespace mediapipe {
namespace api2 {

class InferenceCalculatorOnnxImpl
    : public NodeImpl<InferenceCalculatorOnnx, InferenceCalculatorOnnxImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  Ort::Env ort_env = Ort::Env(nullptr);
  Ort::Session ort_session = Ort::Session(nullptr);
  Ort::MemoryInfo dml_minfo = Ort::MemoryInfo(nullptr);
  Ort::MemoryInfo cpu_minfo = Ort::MemoryInfo(nullptr);
  std::wstring model_path_wide;
};

absl::Status InferenceCalculatorOnnxImpl::UpdateContract(CalculatorContract* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty())
      << "Model path in options is required.";
  return absl::OkStatus();
}

absl::Status InferenceCalculatorOnnxImpl::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  mediapipe::InferenceCalculatorOptions::Delegate delegate = options.delegate();

  const auto& model_path = options.model_path();
  this->model_path_wide = std::wstring(model_path.begin(), model_path.end());

  auto ort_opts = Ort::SessionOptions();
  auto ort_opts_ptr = (OrtSessionOptions*)ort_opts;
  ort_opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  ort_opts.DisableMemPattern();
  ort_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  
  const auto& ort_c_api = Ort::GetApi();
  const OrtDmlApi* dml_api = nullptr;
  auto ort_status = ort_c_api.GetExecutionProviderApi("DML", ORT_API_VERSION, (const void**)&dml_api);
  if (ort_status != nullptr) {
    const auto errmsg = std::string(ort_c_api.GetErrorMessage(ort_status));
    ort_c_api.ReleaseStatus(ort_status);
    RET_CHECK_FAIL() << "DML API not found: " << errmsg;
  }
  assert(dml_api != nullptr);
  
  ort_status = dml_api->SessionOptionsAppendExecutionProvider_DML(ort_opts_ptr, 0);
  if (ort_status != nullptr) {
    const auto errmsg = std::string(ort_c_api.GetErrorMessage(ort_status));
    ort_c_api.ReleaseStatus(ort_status);
    RET_CHECK_FAIL() << "DML EP not added: " << errmsg;
  }

  try {
    this->ort_env = Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, cc->NodeName().c_str());
    this->ort_session = Ort::Session(this->ort_env, this->model_path_wide.c_str(), ort_opts);
    this->dml_minfo = Ort::MemoryInfo("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    this->cpu_minfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPUOutput);
  } catch (Ort::Exception e) {
    RET_CHECK_FAIL() << "ORT create session: " << e.what();
  }

  return absl::OkStatus();
}

absl::Status InferenceCalculatorOnnxImpl::Process(CalculatorContext* cc) {
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;


  Ort::AllocatorWithDefaultOptions cpu_alloc;
  //Ort::Allocator cpu_alloc(ort_session, cpu_minfo);
  const auto& input_stream = cc->Inputs().Get("TENSORS", 0).Get<std::vector<Tensor>>();

  input_tensors.reserve(ort_session.GetInputCount());
  input_names.reserve(ort_session.GetInputCount());
  for (int in_idx = 0; in_idx < ort_session.GetInputCount(); ++in_idx) {
    const auto& input_shape = ort_session.GetInputTypeInfo(in_idx).GetTensorTypeAndShapeInfo();
    const size_t input_bytes_count = input_shape.GetElementCount() * 4;
    assert(input_shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    
    input_tensors.push_back(Ort::Value::CreateTensor(
      (const OrtMemoryInfo*)dml_minfo,
      input_stream[in_idx].GetDirectMLValueReadView().value(),
      input_bytes_count,
      input_shape.GetShape().data(),
      input_shape.GetShape().size(),
      input_shape.GetElementType()));
    input_names.push_back(ort_session.GetInputName(in_idx, cpu_alloc));
  }

  output_tensors.reserve(ort_session.GetOutputCount());
  output_names.reserve(ort_session.GetOutputCount());
  for (int out_idx = 0; out_idx < ort_session.GetOutputCount(); ++out_idx) {
    const auto& output_shape = ort_session.GetOutputTypeInfo(out_idx).GetTensorTypeAndShapeInfo();
    output_tensors.push_back(Ort::Value::CreateTensor(
      (OrtAllocator*)cpu_alloc,
      output_shape.GetShape().data(),
      output_shape.GetShape().size(),
      output_shape.GetElementType()));
    output_names.push_back(ort_session.GetInputName(out_idx, cpu_alloc));
  }

  Ort::RunOptions runOpts = Ort::RunOptions();
  runOpts.SetRunLogVerbosityLevel(3);

  ort_session.Run(runOpts, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_tensors.data(), output_tensors.size());

  // const auto& input_stream = cc->Inputs().Index(0);
  // if (input_stream.IsEmpty()) return absl::OkStatus();

  // const auto& input_image = input_stream.Get<std::vector<Tensor>>();

  // const auto detect_shapes = std::vector<Tensor::Shape>();
  // const auto detect_out = std::vector<Tensor>();

  return absl::OkStatus();
}
absl::Status InferenceCalculatorOnnxImpl::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

}
}
