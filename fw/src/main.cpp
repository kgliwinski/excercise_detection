
#include <Arduino.h>

#include "IMU_ext.h"
#include "excercise_detection.h"
#include "excercise_monitor.h"
#include "vedba_estimation.h"

ImuExtension IMUExt(IMU);
ExcerciseDetection detector;
VedbaEstimation vedbaEstimator;
ExerciseMonitor monitor(vedbaEstimator, detector);

static constexpr bool dbg = false;

void logFullImuData(const AccelerationData& accData, const GyroscopeData& gyroData) {
  if (!dbg) {
    return;
  }

  auto uptime = millis();
  Serial.println("=== New IMU Batch @ " + String(uptime) + "ms ===");

  Serial.println("Accel samples: ");
  Serial.print(accData.samples.size());
  for (const auto& sample : accData.samples) {
    Serial.print(" | x: ");
    Serial.print(sample.x);
    Serial.print(" y: ");
    Serial.print(sample.y);
    Serial.print(" z: ");
    Serial.println(sample.z);
  }
  Serial.println("Gyro samples: ");
  Serial.println(gyroData.samples.size());
  for (const auto& sample : gyroData.samples) {
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
  delay(1000);  // Give serial monitor time to open

  IMUExt.begin();
  Serial.println(IMU.accelerationSampleRate());

  monitor.setup();
  Serial.println("Setup complete");
}

void loop() {
  // 1. Pull from hardware
  auto [accelData, gyroData] = IMUExt.getFullData();

  logFullImuData(accelData, gyroData);

  // 2. Push to the architecture
  monitor.update(accelData, gyroData);

  // 3. Yield to keep buffer from overflowing
  delay(200);
}
