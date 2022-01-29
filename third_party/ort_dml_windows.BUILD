# Description:
#   ONNX Runtime with DirectML for gpu inference on windows

cc_library(
    name = "onnxruntime",
    srcs = [
        "runtimes/win-x64/native/onnxruntime.dll",
        "runtimes/win-x64/native/onnxruntime.lib",
    ],
    hdrs = glob(["build/native/include/*.h"]),
    includes = ["build/native/include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
