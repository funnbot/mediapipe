# Description:
#   ONNX Runtime with DirectML for gpu inference on windows

cc_library(
    name = "onnxruntime",
    hdrs = glob(["build/native/include/*.h"]),
    includes = ["build/native/include/"],
    data = ["runtimes/win-x64/native/onnxruntime.pdb"],
    deps = [":ort_import_shared"],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "ort_import_shared",
    interface_library = "runtimes/win-x64/native/onnxruntime.lib",
    shared_library = "runtimes/win-x64/native/onnxruntime.dll",
    hdrs = glob(["build/native/include/*.h"]),
)