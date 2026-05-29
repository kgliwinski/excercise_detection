#include <Arduino.h>
#include <Arduino_BMI270_BMM150.h>

#include <vector>

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

class ImuExtension {
 public:
  ImuExtension(BoschSensorClass& imu) : imu(imu) {}

  void begin() {
    imu.begin();
    imu.setContinuousMode();
  }

  std::tuple<AccelerationData, GyroscopeData> getFullData() {
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

    if (imu.accelerationAvailable()) {
      const auto samplesCount = imu.getContinuousMode().getSamplesCount() - 1;
      auto accelSamples = imu.getContinuousMode().getAccelData();

      for (size_t i = 0; i < samplesCount; ++i) {
        accelData.samples.push_back(convertAccel(accelSamples[i]));
      }
      imu.getContinuousMode().clearAccelerationAvailable();
    }
    if (imu.gyroscopeAvailable()) {
      const auto samplesCount = imu.getContinuousMode().getSamplesCount() - 1;
      auto gyroSamples = imu.getContinuousMode().getGyroData();
      for (size_t i = 0; i < samplesCount; ++i) {
        gyroData.samples.push_back(convertGyro(gyroSamples[i]));
      }
      imu.getContinuousMode().clearGyroscopeAvailable();
    }
    return std::make_tuple(accelData, gyroData);
  }

 private:
  BoschSensorClass& imu;
  static constexpr float INT16_to_G = 8192.0f;
  static constexpr float INT16_to_DPS = 16.384f;
};
