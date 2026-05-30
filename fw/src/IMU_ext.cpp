#include "IMU_ext.h"

void ImuExtension::begin() {
  imu.begin();
  imu.setContinuousMode();
}

std::tuple<AccelerationData, GyroscopeData> ImuExtension::getFullData() {
  AccelerationData accelData;
  auto convertAccel = [&](const auto& raw) {
    Axis3D s;
    s.x = raw.y / INT16_to_G * -1.f;
    s.y = raw.x / INT16_to_G * -1.f;
    s.z = raw.z / INT16_to_G;
    return s;
  };

  GyroscopeData gyroData;
  auto convertGyro = [&](const auto& raw) {
    Axis3D s;
    s.x = raw.y / INT16_to_DPS * -1.f;
    s.y = raw.x / INT16_to_DPS * -1.f;
    s.z = raw.z / INT16_to_DPS;
    return s;
  };
  bool accAvailable = imu.accelerationAvailable();
  bool gyroAvailable = imu.gyroscopeAvailable();
  while (imu.accelerationAvailable() || imu.gyroscopeAvailable()) {
    if (accAvailable) {
      const auto samplesCount = imu.getContinuousMode().getSamplesCount() - 1;
      auto accelSamples = imu.getContinuousMode().getAccelData();

      for (size_t i = 0; i < samplesCount; ++i) {
        accelData.samples.push_back(convertAccel(accelSamples[i]));
      }
      imu.getContinuousMode().subtractAccelerationAvailable();
    }

    if (gyroAvailable) {
      const auto samplesCount = imu.getContinuousMode().getSamplesCount() - 1;
      auto gyroSamples = imu.getContinuousMode().getGyroData();
      for (size_t i = 0; i < samplesCount; ++i) {
        gyroData.samples.push_back(convertGyro(gyroSamples[i]));
      }
      imu.getContinuousMode().subtractGyroscopeAvailable();
    }
  }
  return std::make_tuple(accelData, gyroData);
}
