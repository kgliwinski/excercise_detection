#pragma once
#include <Arduino.h>
#include <TensorFlowLite.h>
#include <motion_types.h>
#include <stdint.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

static constexpr const char* EXERCISE_NAMES[6] = {"Bicep Curl",           "Dip",  "Lateral Raise",
                                                  "Overhead Triceps Ext", "Rest", "Two arm dumbbell Curl"};

class ExcerciseDetection {
 public:
  void setup();
  // Call this to process any data already buffered (no-arg: uses internal buffer)
  void processNewData();
  // Buffer incoming IMU batches in their original form (unchanged)
  void bufferNewData(const AccelerationData& accData, const GyroscopeData& gyroData);
  void processWindow();

 private:
  static constexpr int NUM_CLASSES = 6;
  static constexpr int INPUT_TENSOR_INDEX = 0;
  static constexpr int OUTPUT_TENSOR_INDEX = 0;
  static constexpr int FIRST_CLASS_INDEX = 0;
  static constexpr int AXIS_COUNT = 3;
  static constexpr int ACCEL_AXIS_OFFSET = 0;
  static constexpr int GYRO_AXIS_OFFSET = 3;
  static constexpr int AXIS_X = 0;
  static constexpr float INITIAL_BEST_CONFIDENCE = -1.0f;
  static constexpr float PREDICTION_CONFIDENCE_THRESHOLD = 0.60f;
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

  // Buffers storing incoming IMU batches in their original types
  // Each element is an unchanged AccelerationData / GyroscopeData as produced by the IMU wrapper
  std::vector<AccelerationData> acc_batches;
  std::vector<GyroscopeData> gyro_batches;
  // Total number of paired samples currently buffered (sum of min(acc.samples, gyro.samples) per batch)
  size_t total_buffered_samples = 0;

  void loadModel();
  void setupErrorReporter();
  void setupOpsResolver();
  void setupInterpreter();
  void setupTensors();
};
