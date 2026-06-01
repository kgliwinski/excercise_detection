#include "excercise_detection.h"

#include "model.h"

void ExcerciseDetection::setup() {
  loadModel();
  setupErrorReporter();
  setupOpsResolver();
  setupInterpreter();
  setupTensors();
}

void ExcerciseDetection::loadModel() {
  tflModel = tflite::GetModel(model_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    while (1);
  }
}

void ExcerciseDetection::setupErrorReporter() {
  static tflite::MicroErrorReporter error_reporter;
  tflErrorReporter = &error_reporter;
}

void ExcerciseDetection::setupOpsResolver() {
  static tflite::AllOpsResolver ops_resolver;
  tflOpsResolver = &ops_resolver;
}

void ExcerciseDetection::setupInterpreter() {
  static tflite::MicroInterpreter static_interpreter(tflModel, *tflOpsResolver, tensorArena, tensorArenaSize,
                                                     tflErrorReporter);
  tflInterpreter = &static_interpreter;
}

void ExcerciseDetection::setupTensors() {
  // Allocate memory
  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (1);
  }
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // Grab Quantization Parameters
  input_scale = tflInputTensor->params.scale;
  input_zero_point = tflInputTensor->params.zero_point;
  output_scale = tflOutputTensor->params.scale;
  output_zero_point = tflOutputTensor->params.zero_point;
}

bool ExcerciseDetection::evaluateWindow(const AccelerationData& accWindow, const GyroscopeData& gyroWindow) {
  // 1. Safety Check: Ensure the sliding window actually gave us 150 samples
  if (accWindow.samples.size() < TARGET_SAMPLES || gyroWindow.samples.size() < TARGET_SAMPLES) {
    Serial.println("[AI] Error: Window does not contain enough samples!");
    return false;
  }

  int8_t* input_data = tflInterpreter->typed_input_tensor<int8_t>(0);

  // 2. Flatten and Quantize
  for (size_t i = 0; i < TARGET_SAMPLES; ++i) {
    int offset = i * NUM_FEATURES;

    // Accel
    input_data[offset + 0] = static_cast<int8_t>((accWindow.samples[i].x / input_scale) + input_zero_point);
    input_data[offset + 1] = static_cast<int8_t>((accWindow.samples[i].y / input_scale) + input_zero_point);
    input_data[offset + 2] = static_cast<int8_t>((accWindow.samples[i].z / input_scale) + input_zero_point);

    // Gyro
    input_data[offset + 3] = static_cast<int8_t>((gyroWindow.samples[i].x / input_scale) + input_zero_point);
    input_data[offset + 4] = static_cast<int8_t>((gyroWindow.samples[i].y / input_scale) + input_zero_point);
    input_data[offset + 5] = static_cast<int8_t>((gyroWindow.samples[i].z / input_scale) + input_zero_point);
  }

  // 3. Run Inference
  if (tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("[AI] Inference failed!");
    return false;
  }

  // 4. Output Processing
  int8_t* output_data = tflInterpreter->typed_output_tensor<int8_t>(0);
  uint8_t best_class = 0;
  float best_confidence = -1.0f;

  for (uint8_t c = 0; c < NUM_CLASSES; ++c) {
    float confidence = (output_data[c] - output_zero_point) * output_scale;
    if (confidence > best_confidence) {
      best_confidence = confidence;
      best_class = c;
    }
  }

  // 5. Evaluate and Return
  Serial.print("[AI] Detected: ");
  if (best_confidence >= PREDICTION_CONFIDENCE_THRESHOLD) {
    Serial.print(EXERCISE_NAMES[best_class]);
    Serial.print(" (confidence: ");
    Serial.print(best_confidence * 100);
    Serial.println("%)");

    // If it's a valid exercise (not rest), tell the Monitor to flush the buffer
    if (best_class != REST_CLASS_INDEX) {
      return true;
    } else {
      return false;  // It's just rest
    }
  } else {
    Serial.print("UNKNOWN (confidence: ");
    Serial.print(best_confidence * 100);
    Serial.println("%)");
    return false;
  }
}
