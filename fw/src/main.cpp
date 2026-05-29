// #include <Arduino.h>
// #include "IMU_ext.h"


// ImuExtension IMUExt(IMU);

// static constexpr bool DEBUG = false;

// void setup() {
//   Serial.begin(115200);
//   delay(1000);
//   IMUExt.begin();
// }

// void logFullImuData(const AccelerationData& accData, const GyroscopeData& gyroData) {
//   if(!DEBUG) {
//     return;
//   }
  
//   Serial.println("Accel samples: ");
//   Serial.print(accData.samples.size());
//   for(const auto& sample : accData.samples){
//     Serial.print(" | x: ");
//     Serial.print(sample.x);
//     Serial.print(" y: ");
//     Serial.print(sample.y);
//     Serial.print(" z: ");
//     Serial.println(sample.z);
//   }
//   Serial.println("Gyro samples: ");
//   Serial.println(gyroData.samples.size());
//   for(const auto& sample : gyroData.samples){
//     Serial.print(" | x: ");
//     Serial.print(sample.x);
//     Serial.print(" y: ");
//     Serial.print(sample.y);
//     Serial.print(" z: ");
//     Serial.println(sample.z);
//   }
//   Serial.println();
// }

// void loop() {
//   delay(500);
//   auto [accelData, gyroData] = IMUExt.getFullData();
//   logFullImuData(accelData, gyroData);
// }
#include <Arduino.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h> // <-- NEW
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "IMU_ext.h" // Your custom IMU wrapper
#include "model.h"   // The header file generated in Python

// Initialize your custom IMU
ImuExtension IMUExt(IMU);

// TFLM Globals
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;
tflite::ErrorReporter* tflErrorReporter = nullptr;

constexpr int tensorArenaSize = 64 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Model specifics
const int TARGET_SAMPLES = 150; 
const int NUM_FEATURES = 6;  

// Tracking how full our AI buffer is
int current_sample_count = 0;

// Quantization variables
float input_scale;
int32_t input_zero_point;
float output_scale;
int32_t output_zero_point;

const char* EXERCISES[] = {
  "Bicep Curl", "Dip", "Lateral Raise", "Overhead Triceps Ext", "Dumbbell Curl"
};

void setup() {
  Serial.begin(115200);
  delay(1000); // Give serial monitor time to open
  
  IMUExt.begin();

// Load the model
  tflModel = tflite::GetModel(model_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    while (1);
  }

  // Set up the Error Reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  tflErrorReporter = &micro_error_reporter;

  // Use the Harvard library's AllOpsResolver
  static tflite::AllOpsResolver tflOpsResolver;

  // Initialize the interpreter with the 5th argument (tflErrorReporter)
  static tflite::MicroInterpreter static_interpreter(
      tflModel, tflOpsResolver, tensorArena, tensorArenaSize, tflErrorReporter);
      
  tflInterpreter = &static_interpreter;

  // Allocate memory
  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (1);
  }

  // Map the input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // Grab Quantization Parameters
  input_scale = tflInputTensor->params.scale;
  input_zero_point = tflInputTensor->params.zero_point;
  output_scale = tflOutputTensor->params.scale;
  output_zero_point = tflOutputTensor->params.zero_point;

  Serial.println("Setup complete. Waiting for movement...");
}

void loop() {
  // 1. Grab whatever is currently in the FIFO buffer
  auto [accelData, gyroData] = IMUExt.getFullData();
  
  // Safely find out how many complete pairs of data we got
  int num_new_samples = min(accelData.samples.size(), gyroData.samples.size());

  // 2. Iterate through the new FIFO batch
  for (int i = 0; i < num_new_samples; i++) {
    
    // Calculate where in the 1D tensor array we need to put this data
    int base_index = current_sample_count * NUM_FEATURES;

    // 3. Extract, Quantize, and Insert into Tensor
    tflInputTensor->data.int8[base_index + 0] = (int8_t)(accelData.samples[i].x / input_scale + input_zero_point);
    tflInputTensor->data.int8[base_index + 1] = (int8_t)(accelData.samples[i].y / input_scale + input_zero_point);
    tflInputTensor->data.int8[base_index + 2] = (int8_t)(accelData.samples[i].z / input_scale + input_zero_point);
    tflInputTensor->data.int8[base_index + 3] = (int8_t)(gyroData.samples[i].x / input_scale + input_zero_point);
    tflInputTensor->data.int8[base_index + 4] = (int8_t)(gyroData.samples[i].y / input_scale + input_zero_point);
    tflInputTensor->data.int8[base_index + 5] = (int8_t)(gyroData.samples[i].z / input_scale + input_zero_point);

    current_sample_count++;

    // 4. Have we reached 150 samples (3 seconds of data)?
    if (current_sample_count >= TARGET_SAMPLES) {
      
      // RUN INFERENCE!
      TfLiteStatus invokeStatus = tflInterpreter->Invoke();
      if (invokeStatus != kTfLiteOk) {
        Serial.println("Inference failed!");
        current_sample_count = 0; // Reset and try again
        return;
      }

      // Read Results
      float max_score = 0.0;
      int best_class = -1;

      for (int c = 0; c < 5; c++) {
        int8_t quantized_output = tflOutputTensor->data.int8[c];
        float float_prob = (quantized_output - output_zero_point) * output_scale;

        if (float_prob > max_score) {
          max_score = float_prob;
          best_class = c;
        }
      }

      // Output Prediction
      Serial.print("\n>>> PREDICTION: ");
      if (max_score > 0.60) {
        Serial.print(EXERCISES[best_class]);
        Serial.print(" (Confidence: ");
        Serial.print(max_score * 100);
        Serial.println("%) <<<");
      } else {
        Serial.println("UNKNOWN MOVEMENT <<<");
      }

      // 5. Reset the buffer counter for the next repetition
      current_sample_count = 0; 
      
      // Optional: Break out of the loop if you want to discard the rest 
      // of the FIFO batch after a successful rep, or leave it to immediately 
      // start filling the next 3-second window!
      break; 
    }
  }

  // Small delay so we aren't polling the IMU millions of times a second 
  // while it's trying to fill the FIFO. 
  delay(10); 
}