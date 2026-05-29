#include <Arduino.h>
#include "IMU_ext.h"

ImuExtension IMUExt(IMU);

void setup() {
  Serial.begin(115200);
  delay(1000);
  IMUExt.begin();
}

void loop() {
  delay(500);
  auto [accelData, gyroData] = IMUExt.getFullData();
  Serial.println("Accel samples: ");
  Serial.print(accelData.samples.size());
  for(const auto& sample : accelData.samples){
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