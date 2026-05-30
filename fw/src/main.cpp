
#include <Arduino.h>

#include "excercise_detection.h"
#include "IMU_ext.h" // Your custom IMU wrapper
#include "model.h"   // The header file generated in Python

// Initialize your custom IMU
ImuExtension IMUExt(IMU);

// Initialize exercise detection
ExcerciseDetection detector;

static constexpr bool dbg = true;

void logFullImuData(const AccelerationData& accData, const GyroscopeData& gyroData) {
  if(!dbg) {
    return;
  }
  
  Serial.println("Accel samples: ");
  Serial.print(accData.samples.size());
  for(const auto& sample : accData.samples){
    Serial.print(" | x: ");
    Serial.print(sample.x);
    Serial.print(" y: ");
    Serial.print(sample.y);
    Serial.print(" z: ");
    Serial.println(sample.z);
  }
  Serial.println("Gyro samples: ");
  Serial.println(gyroData.samples.size());
  for(const auto& sample : gyroData.samples){
    Serial.print(" | x: ");
    Serial.print(sample.x);
    Serial.print(" y: ");
    Serial.print(sample.y);
    Serial.print(" z: ");
    Serial.println(sample.z);
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(1000); // Give serial monitor time to open
  
  IMUExt.begin();
  Serial.println(IMU.accelerationSampleRate());

  // Initialize exercise detection model
  detector.setup();
  
  Serial.println("Setup complete. Waiting for movement...");
}

void loop() {
  // 1. Grab whatever is currently in the FIFO buffer
  auto [accelData, gyroData] = IMUExt.getFullData();
  logFullImuData(accelData, gyroData);
  
  // Safely find out how many complete pairs of data we got
  int num_new_samples = min(accelData.samples.size(), gyroData.samples.size());
  Serial.println("FIFO batch samples: " + String(num_new_samples));
  
  // 2. Process new data through the exercise detector
  detector.processNewData(accelData, gyroData);

  // Small delay so we aren't polling the IMU millions of times a second 
  // while it's trying to fill the FIFO. 
  delay(1000); 
}