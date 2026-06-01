#include "excercise_monitor.h"

ExerciseMonitor::ExerciseMonitor(VedbaEstimation& estimator, ExcerciseDetection& detector)
    : vedbaEstimator(estimator), aiDetector(detector) {}

void ExerciseMonitor::setup() {
  // Initialize the sub-modules
  vedbaEstimator.setup();
  aiDetector.setup();
  windowBuffer.clear();

  Serial.println("[MONITOR] Setup complete. Continuous DSP pipeline active.");
}

void ExerciseMonitor::update(const AccelerationData& accData, const GyroscopeData& gyroData) {
  // 1. Safety check
  if (accData.samples.empty() || gyroData.samples.empty()) {
    return;
  }

  // 2. Constantly feed the Sliding Window
  windowBuffer.push(accData, gyroData);

  // 3. Constantly feed VeDBA to calibrate gravity and watch for the end of a rep
  // (Ensure your VedbaEstimation::processBatch returns true when dropping back into a valley)
  bool rep_just_finished = vedbaEstimator.processBatch(accData, gyroData);

  // THE POST-FLUSH LOGIC
  if (rep_just_finished && windowBuffer.isFull()) {
    Serial.println("\n[SYSTEM] Repetition Valley Detected. Running AI Inference...");

    // Pass the window to the stateless AI
    bool valid_exercise_found = aiDetector.evaluateWindow(windowBuffer.getAccWindow(), windowBuffer.getGyroWindow());

    if (valid_exercise_found) {
      Serial.println("[SYSTEM] Exercise validated. Flushing Running Window.");
      windowBuffer.clear();
    } else {
      Serial.println("[SYSTEM] Ignored (Rest/Unknown). Keeping window continuous.");
    }
  }
}
