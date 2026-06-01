#pragma once
#include "motion_types.h"
#include <cmath>
#include <algorithm>

class VedbaEstimation {
public:
    // Notice the threshold is highly sensitive (0.08g) to detect the moment you leave the "Valley"
    VedbaEstimation(float alpha = 0.1f, float threshold = 0.08f);
    
    void setup();
    
    // Returns TRUE if a physical repetition has just finished
    bool processBatch(const AccelerationData& accData, const GyroscopeData& gyroData);

private:
    float ema_alpha;
    float vedba_threshold;
    
    float grav_x, grav_y, grav_z;
    
    // Internal Watchdog State
    enum class State { RESTING, ACTIVE };
    State currentState;
};