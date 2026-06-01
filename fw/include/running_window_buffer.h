#pragma once
#include "motion_types.h"
#include <vector>
#include <algorithm>

class RunningWindowBuffer {
public:
    void clear();
    bool isFull() const;
    void push(const AccelerationData& accData, const GyroscopeData& gyroData);
    
    // Retrieve the cleanly formatted window
    AccelerationData getAccWindow() const;
    GyroscopeData getGyroWindow() const;

private:
    static constexpr size_t TARGET_SAMPLES = 150;
    std::vector<Axis3D> acc_buffer;
    std::vector<Axis3D> gyro_buffer;
};