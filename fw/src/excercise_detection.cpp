#include "excercise_detection.h"

#include "model.h"
#include <algorithm>

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

void ExcerciseDetection::processNewData() {
  // Process any buffered data when we have a full window
  if (this->total_buffered_samples >= TARGET_SAMPLES) {
    processWindow();
  }
}

void ExcerciseDetection::bufferNewData(const AccelerationData& accData, const GyroscopeData& gyroData) {
  // Store the incoming batches unchanged
  acc_batches.push_back(accData);
  gyro_batches.push_back(gyroData);

  // Count only paired samples (min of accel/gyro per batch)
  size_t pair_count = std::min(accData.samples.size(), gyroData.samples.size());
  this->total_buffered_samples += pair_count;
}

void ExcerciseDetection::processWindow() {
  if (this->total_buffered_samples < ExcerciseDetection::TARGET_SAMPLES) return; // not enough data

  // Flatten oldest->newest paired samples into the model input (take exactly TARGET_SAMPLES)
  size_t filled = 0;
  for (size_t b = 0; b < this->acc_batches.size() && filled < TARGET_SAMPLES; ++b) {
    size_t batch_pairs = std::min(this->acc_batches[b].samples.size(), this->gyro_batches[b].samples.size());
    for (size_t i = 0; i < batch_pairs && filled < TARGET_SAMPLES; ++i) {
      size_t tensor_base = filled * ExcerciseDetection::NUM_FEATURES;
      const Axis3D& a = this->acc_batches[b].samples[i];
      const Axis3D& g = this->gyro_batches[b].samples[i];
      float vals[ExcerciseDetection::NUM_FEATURES] = {a.x, a.y, a.z, g.x, g.y, g.z};
      for (size_t f = 0; f < ExcerciseDetection::NUM_FEATURES; ++f) {
        int8_t q = static_cast<int8_t>(vals[f] / this->input_scale + this->input_zero_point);
        this->tflInputTensor->data.int8[tensor_base + f] = q;
      }
      ++filled;
    }
  }

  if (filled < TARGET_SAMPLES) {
    // Shouldn't happen, but guard just in case
    Serial.println("Not enough paired samples to fill window despite counter.");
    return;
  }

  // run inference
  if (this->tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed!");
    return;
  }

  int8_t* output_data = this->tflInterpreter->typed_output_tensor<int8_t>(ExcerciseDetection::OUTPUT_TENSOR_INDEX);
  size_t best_class = ExcerciseDetection::FIRST_CLASS_INDEX;
  float best_confidence = ExcerciseDetection::INITIAL_BEST_CONFIDENCE;
  for (size_t c = ExcerciseDetection::FIRST_CLASS_INDEX; c < ExcerciseDetection::NUM_CLASSES; ++c) {
    float confidence = (output_data[c] - this->output_zero_point) * this->output_scale;
    if (confidence > best_confidence) {
      best_confidence = confidence;
      best_class = c;
    }
  }

  Serial.print("Detected: ");
  if (best_confidence >= ExcerciseDetection::PREDICTION_CONFIDENCE_THRESHOLD) {
    Serial.print(EXERCISE_NAMES[best_class]);
    Serial.print(" (confidence: ");
    Serial.print(best_confidence);
    Serial.println(")");
  } else {
    Serial.print("UNKNOWN MOVEMENT (confidence: ");
    Serial.print(best_confidence);
    Serial.println(")");
  }

  // Trim oldest paired samples so we keep only the last TARGET_SAMPLES
  size_t excess = (this->total_buffered_samples > TARGET_SAMPLES) ? (this->total_buffered_samples - TARGET_SAMPLES) : 0;
  while (excess > 0 && !this->acc_batches.empty()) {
    size_t batch_pairs = std::min(this->acc_batches.front().samples.size(), this->gyro_batches.front().samples.size());
    if (batch_pairs <= excess) {
      excess -= batch_pairs;
      this->total_buffered_samples -= batch_pairs;
      this->acc_batches.erase(this->acc_batches.begin());
      this->gyro_batches.erase(this->gyro_batches.begin());
    } else {
      // remove 'excess' samples from the start of the first batch
      this->acc_batches.front().samples.erase(this->acc_batches.front().samples.begin(),
                                              this->acc_batches.front().samples.begin() + excess);
      this->gyro_batches.front().samples.erase(this->gyro_batches.front().samples.begin(),
                                              this->gyro_batches.front().samples.begin() + excess);
      this->total_buffered_samples -= excess;
      excess = 0;
    }
  }
}
