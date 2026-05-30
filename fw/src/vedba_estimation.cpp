#include "vedba_estimation.h"

// Constructor implementation
VedbaEstimation::VedbaEstimation(float alpha, float threshold, int cooldown_ticks) 
  : ema_alpha(alpha), 
    vedba_threshold(threshold), 
    max_cooldown_ticks(cooldown_ticks),
    grav_x(0.0f), grav_y(0.0f), grav_z(1.0f), // Start assuming 1G resting flat on Z-axis
    cooldown_counter(0) {}

void VedbaEstimation::setup() {
    // Reset state variables
    grav_x = 0.0f;
    grav_y = 0.0f;
    grav_z = 1.0f; 
    cooldown_counter = 0;
}

void VedbaEstimation::resetCooldown() {
    cooldown_counter = 0;
}

VedbaLevel VedbaEstimation::processNewData(const AccelerationData& accData, const GyroscopeData& gyroData) {
    bool movement_detected = false;
    
    // Ensure we don't read out of bounds if one sensor FIFO is larger than the other
    size_t num_samples = std::min(accData.samples.size(), gyroData.samples.size());

    for (size_t i = 0; i < num_samples; ++i) {
        float ax = accData.samples[i].x;
        float ay = accData.samples[i].y;
        float az = accData.samples[i].z;

        // 1. Update Gravity Low-Pass Filter (EMA)
        grav_x = (ema_alpha * ax) + ((1.0f - ema_alpha) * grav_x);
        grav_y = (ema_alpha * ay) + ((1.0f - ema_alpha) * grav_y);
        grav_z = (ema_alpha * az) + ((1.0f - ema_alpha) * grav_z);

        // 2. Isolate pure kinetic movement (Dynamic Body Acceleration)
        float dba_x = ax - grav_x;
        float dba_y = ay - grav_y;
        float dba_z = az - grav_z;

        // 3. Calculate overall 3D Magnitude (VeDBA)
        float vedba = std::sqrt((dba_x * dba_x) + (dba_y * dba_y) + (dba_z * dba_z));

        // 4. Tick down the cooldown
        if (cooldown_counter > 0) {
            cooldown_counter--;
        }

        // 5. Peak Trigger Detection
        if (vedba > vedba_threshold && cooldown_counter == 0) {
            movement_detected = true;
            cooldown_counter = max_cooldown_ticks; // Prevent double-triggering
        }
    }

    // If any sample in this FIFO batch crossed the threshold, report MOVEMENT.
    return movement_detected ? VedbaLevel::MOVEMENT : VedbaLevel::STATIC;
}