#pragma once
#include <Arduino.h>
#include <TensorFlowLite.h>
#include <motion_types.h>
#include <stdint.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>  // <-- NEW
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <string>
class ExcerciseDetection {
 public:
  void setup();
  void processNewData(const AccelerationData& accData, const GyroscopeData& gyroData);

 private:
  static constexpr uint8_t NUM_CLASSES = 6;

  // Exercise class names
  const std::array<std::string, NUM_CLASSES> EXERCISES = {"Bicep Curl",           "Dip",  "Lateral Raise",
                                                          "Overhead Triceps Ext", "Rest", "Two arm dumbbell Curl"};

  // Model specifics
  static constexpr size_t TARGET_SAMPLES = 150;
  static constexpr size_t NUM_FEATURES = 6;
  static constexpr size_t tensorArenaSize = 64 * 1024;

  // TFLM Globals
  const tflite::Model* tflModel = nullptr;
  tflite::MicroInterpreter* tflInterpreter = nullptr;
  tflite::AllOpsResolver* tflOpsResolver = nullptr;
  TfLiteTensor* tflInputTensor = nullptr;
  TfLiteTensor* tflOutputTensor = nullptr;
  tflite::ErrorReporter* tflErrorReporter = nullptr;

  byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

  // Tracking how full our AI buffer is
  size_t current_sample_count = 0;

  // Quantization variables
  float input_scale;
  int32_t input_zero_point;
  float output_scale;
  int32_t output_zero_point;

  void loadModel();
  void setupErrorReporter();
  void setupOpsResolver();
  void setupInterpreter();
  void setupTensors();
};
