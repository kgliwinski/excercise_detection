#pragma once
#include <Arduino.h>
#include "motion_types.h"
#include "vedba_estimation.h"
#include "excercise_detection.h"
#include "running_window_buffer.h"

class ExerciseMonitor {
public:
    // Inject our DSP and AI modules so the Monitor can orchestrate them
    ExerciseMonitor(VedbaEstimation& estimator, ExcerciseDetection& detector);
    
    void setup();
    void update(const AccelerationData& accData, const GyroscopeData& gyroData);

private:
    VedbaEstimation& vedbaEstimator;
    ExcerciseDetection& aiDetector;
    RunningWindowBuffer windowBuffer; // The sliding window lives inside the monitor
};