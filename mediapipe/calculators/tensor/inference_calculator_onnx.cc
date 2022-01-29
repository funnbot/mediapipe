#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/framework/deps/file_path.h"

#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"

