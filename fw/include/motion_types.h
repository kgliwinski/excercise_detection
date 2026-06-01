#pragma once
#include <stdint.h>
#include <vector>

// Fundamental data structure constants
namespace MotionTypes {
  static constexpr uint8_t AXIS_COUNT = 3;           // x, y, z per sensor
  static constexpr uint8_t NUM_FEATURES = 6;         // 3 accel + 3 gyro
  static constexpr uint8_t ACCEL_AXIS_OFFSET = 0;    // acceleration at indices [0-2]
  static constexpr uint8_t GYRO_AXIS_OFFSET = 3;     // gyro at indices [3-5]
  static constexpr uint16_t TARGET_SAMPLES = 150;    // 3-second window at 50Hz
}

struct Axis3D {
  float x;
  float y;
  float z;
};

struct AccelerationData {
  std::vector<Axis3D> samples;
};

struct GyroscopeData {
  std::vector<Axis3D> samples;
};