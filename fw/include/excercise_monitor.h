#pragma once
#include <Arduino.h>
#include "motion_types.h"
#include "vedba_estimation.h"
#include "excercise_detection.h"

enum class SystemState { IDLE, RECORDING };

class ExerciseMonitor {
public:
    // Pass our sub-modules in by reference so this class can coordinate them
    ExerciseMonitor(VedbaEstimation& estimator, ExcerciseDetection& detector);
    
    void setup();
    void update(const AccelerationData& accData, const GyroscopeData& gyroData);

private:
    VedbaEstimation& vedbaEstimator;
    ExcerciseDetection& aiDetector;
    
    SystemState currentState;
    size_t samples_recorded;
    static constexpr size_t TARGET_SAMPLES = 150;
};