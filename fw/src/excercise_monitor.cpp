#include "excercise_monitor.h"

ExerciseMonitor::ExerciseMonitor(VedbaEstimation& estimator, ExcerciseDetection& detector)
    : vedbaEstimator(estimator), aiDetector(detector), 
      currentState(SystemState::IDLE), samples_recorded(0) {}

void ExerciseMonitor::setup() {
    // Setup the sub-modules
    vedbaEstimator.setup();
    aiDetector.setup();
    
    // Initialize state
    currentState = SystemState::IDLE;
    samples_recorded = 0;
    
    Serial.println("[MONITOR] Setup complete. System is IDLE. Waiting for movement...");
}

void ExerciseMonitor::update(const AccelerationData& accData, const GyroscopeData& gyroData) {
    // Ensure we have balanced data
    int num_new_samples = std::min(accData.samples.size(), gyroData.samples.size());
    if (num_new_samples == 0) {
        return; 
    }

    // ==========================================
    // STATE MACHINE
    // ==========================================
    if (currentState == SystemState::IDLE) {
        
        // Let VeDBA hunt for a peak
        if (vedbaEstimator.processNewData(accData, gyroData) == VedbaLevel::MOVEMENT) {
            Serial.println("\n[SYSTEM] Peak Detected! Switching to RECORDING state...");
            
            currentState = SystemState::RECORDING;
            samples_recorded = 0;
            
            // Push the triggering batch into the AI
            aiDetector.bufferNewData(accData, gyroData);
            samples_recorded += num_new_samples;
        }
        
    } 
    else if (currentState == SystemState::RECORDING) {
        
        // Keep VeDBA's internal gravity filters updated
        vedbaEstimator.processNewData(accData, gyroData);
        
        // Feed the AI buffer
        aiDetector.bufferNewData(accData, gyroData);
        samples_recorded += num_new_samples;

        // Trigger internal AI logic 
        aiDetector.processNewData();

        // Check if the repetition window is complete
        if (samples_recorded >= TARGET_SAMPLES) {
            currentState = SystemState::IDLE;
            samples_recorded = 0;
        }
    }
}