#pragma once
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