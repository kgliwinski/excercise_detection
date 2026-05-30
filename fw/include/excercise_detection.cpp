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

void ExcerciseDetection::processNewData(const AccelerationData& accData, const GyroscopeData& gyroData) {
  // Ensure we have matching sample counts
  size_t num_samples = accData.samples.size();
  if (gyroData.samples.size() != num_samples) {
    Serial.println("Accel and Gyro sample counts mismatch!");
    return;
  }

  // Get pointer to input tensor data
  int8_t* input_data = tflInterpreter->typed_input_tensor<int8_t>(0);

  // Add samples to input buffer
  for (size_t i = 0; i < num_samples; ++i) {
    if (current_sample_count >= TARGET_SAMPLES) {
      // Buffer is full, run inference
      if (tflInterpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed!");
        return;
      }

      // Process output
      int8_t* output_data = tflInterpreter->typed_output_tensor<int8_t>(0);
      
      // Find the class with highest confidence
      uint8_t best_class = 0;
      float best_confidence = -1.0f;
      
      for (uint8_t c = 0; c < NUM_CLASSES; ++c) {
        float confidence = (output_data[c] - output_zero_point) * output_scale;
        if (confidence > best_confidence) {
          best_confidence = confidence;
          best_class = c;
        }
      }

      Serial.print("Detected: ");
      Serial.print(EXERCISES[best_class].c_str());
      Serial.print(" (confidence: ");
      Serial.print(best_confidence);
      Serial.println(")");

      // Reset buffer for next batch
      current_sample_count = 0;
    }

    // Quantize and store accel data
    int offset = current_sample_count * NUM_FEATURES;
    for (uint8_t j = 0; j < 3; ++j) {
      float* accel_axis = (j == 0) ? &accData.samples[i].x : (j == 1) ? &accData.samples[i].y : &accData.samples[i].z;
      input_data[offset + j] = (*accel_axis / input_scale) + input_zero_point;
    }

    // Quantize and store gyro data
    for (uint8_t j = 0; j < 3; ++j) {
      float* gyro_axis = (j == 0) ? &gyroData.samples[i].x : (j == 1) ? &gyroData.samples[i].y : &gyroData.samples[i].z;
      input_data[offset + 3 + j] = (*gyro_axis / input_scale) + input_zero_point;
    }

    current_sample_count++;
  }
}
