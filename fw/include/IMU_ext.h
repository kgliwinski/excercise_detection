#pragma once
#include <Arduino.h>
#include <Arduino_BMI270_BMM150.h>

#include "motion_types.h"

class ImuExtension {
 public:
  ImuExtension(BoschSensorClass& imu) : imu(imu) {}

  void begin();

  std::tuple<AccelerationData, GyroscopeData> getFullData();

 private:
  BoschSensorClass& imu;
  static constexpr float INT16_to_G = 8192.0f;
  static constexpr float INT16_to_DPS = 16.384f;
};
