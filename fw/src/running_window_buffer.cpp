#include "running_window_buffer.h"

void RunningWindowBuffer::clear() {
    acc_buffer.clear();
    gyro_buffer.clear();
}

bool RunningWindowBuffer::isFull() const {
    // Both buffers must have exactly TARGET_SAMPLES
    return (acc_buffer.size() >= TARGET_SAMPLES && gyro_buffer.size() >= TARGET_SAMPLES);
}

void RunningWindowBuffer::push(const AccelerationData& accData, const GyroscopeData& gyroData) {
    size_t num_samples = std::min(accData.samples.size(), gyroData.samples.size());
    
    // 1. Add new samples
    for (size_t i = 0; i < num_samples; ++i) {
        acc_buffer.push_back(accData.samples[i]);
        gyro_buffer.push_back(gyroData.samples[i]);
    }

    // 2. Slide the window (pop from the front if we exceed TARGET_SAMPLES)
    if (acc_buffer.size() > TARGET_SAMPLES) {
        size_t excess = acc_buffer.size() - TARGET_SAMPLES;
        acc_buffer.erase(acc_buffer.begin(), acc_buffer.begin() + excess);
        gyro_buffer.erase(gyro_buffer.begin(), gyro_buffer.begin() + excess);
    }
}

AccelerationData RunningWindowBuffer::getAccWindow() const {
    return {acc_buffer}; 
}

GyroscopeData RunningWindowBuffer::getGyroWindow() const {
    return {gyro_buffer};
}