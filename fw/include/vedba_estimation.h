#pragma once
#include "motion_types.h"
#include <cmath>
#include <algorithm>

enum class VedbaLevel { STATIC = 0, MOVEMENT };

class VedbaEstimation {
public:
    // Constructor with sensible default parameters
    // alpha: How fast the gravity filter adapts (0.1 is standard for ~50Hz)
    // threshold: G-force required to trigger a movement
    // cooldown: How many ticks to wait before allowing another trigger
    VedbaEstimation(float alpha = 0.1f, float threshold = 0.40f, int cooldown_ticks = 25);
    
    void setup();
    VedbaLevel processNewData(const AccelerationData& accData, const GyroscopeData& gyroData);
    
    // Optional utility to forcefully reset the cooldown (e.g., if AI inference fails)
    void resetCooldown();

private:
    float ema_alpha;
    float vedba_threshold;
    int max_cooldown_ticks;
    
    // Internal State
    float grav_x;
    float grav_y;
    float grav_z;
    int cooldown_counter;
};