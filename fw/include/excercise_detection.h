#pragma once
#include <Arduino.h>
#include <TensorFlowLite.h>
#include "motion_types.h"
#include <stdint.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Make sure "Rest" aligns with how you trained it! (Index 4 here)
// static constexpr const char* EXERCISE_NAMES[6] = {
//   "Bicep Curl", "Dip", "Lateral Raise", "Overhead Triceps Ext", "Rest", "Two arm dumbbell Curl"
// };

static constexpr const char* EXERCISE_NAMES[2] = {
  "Bicep Curl", "Rest"
};

class ExcerciseDetection {
 public:
  void setup();
  
  // Now purely stateless: Takes a full 150-sample window, runs AI, returns true if valid exercise.
  bool evaluateWindow(const AccelerationData& accWindow, const GyroscopeData& gyroWindow);

 private:
  static constexpr int NUM_CLASSES = 2;
  static constexpr int REST_CLASS_INDEX = 1; // Used to tell Monitor if we should ignore it
  static constexpr float PREDICTION_CONFIDENCE_THRESHOLD = 0.60f;
  static constexpr size_t TARGET_SAMPLES = 150;
  static constexpr size_t NUM_FEATURES = 6;
  static constexpr size_t tensorArenaSize = 64 * 1024;
  
  const bool debug = true; // Set to true to enable verbose AI debugging output

  // TFLM Globals
  const tflite::Model* tflModel = nullptr;
  tflite::MicroInterpreter* tflInterpreter = nullptr;
  tflite::AllOpsResolver* tflOpsResolver = nullptr;
  TfLiteTensor* tflInputTensor = nullptr;
  TfLiteTensor* tflOutputTensor = nullptr;
  tflite::ErrorReporter* tflErrorReporter = nullptr;

  byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

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