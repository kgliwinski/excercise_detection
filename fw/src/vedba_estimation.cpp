#include "vedba_estimation.h"

VedbaEstimation::VedbaEstimation(float alpha, float threshold)
    : ema_alpha(alpha),
      vedba_threshold(threshold),
      grav_x(0.0f),
      grav_y(0.0f),
      grav_z(1.0f),
      currentState(State::RESTING) {}

void VedbaEstimation::setup() {
  grav_x = 0.0f;
  grav_y = 0.0f;
  grav_z = 1.0f;
  currentState = State::RESTING;
}

bool VedbaEstimation::processBatch(const AccelerationData& accData, const GyroscopeData& gyroData) {
  bool repetition_finished = false;
  size_t num_samples = std::min(accData.samples.size(), gyroData.samples.size());

  for (size_t i = 0; i < num_samples; ++i) {
    float ax = accData.samples[i].x;
    float ay = accData.samples[i].y;
    float az = accData.samples[i].z;

    // Continuous Gravity Calibration
    grav_x = (ema_alpha * ax) + ((1.0f - ema_alpha) * grav_x);
    grav_y = (ema_alpha * ay) + ((1.0f - ema_alpha) * grav_y);
    grav_z = (ema_alpha * az) + ((1.0f - ema_alpha) * grav_z);

    // Dynamic Magnitude
    float dba_x = ax - grav_x;
    float dba_y = ay - grav_y;
    float dba_z = az - grav_z;
    float vedba = std::sqrt((dba_x * dba_x) + (dba_y * dba_y) + (dba_z * dba_z));

    // State Machine Watchdog
    if (currentState == State::RESTING) {
      if (vedba > vedba_threshold) {
        currentState = State::ACTIVE;  // You left the valley!
      }
    } else if (currentState == State::ACTIVE) {
      if (vedba < vedba_threshold) {
        currentState = State::RESTING;  // You returned to the valley!
        repetition_finished = true;     // Flag the end of the rep
      }
    }
  }

  return repetition_finished;
}
