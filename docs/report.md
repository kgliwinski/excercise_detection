# Excercise detection on Arduino Nano 33 BLE Sense

The goal of this project was to create a simple exercise detection system that can run on the Arduino Nano 33 BLE Sense. The system uses the built-in accelerometer and gyroscope to classify whether the user is performing:
- Bicep curls
- Tricep extensions
- Shoulder presses
- Lateral raises
- etc.

The model should be able to run in real-time and provide accurate results.

There were more requirements in the original proposal, regarding the models and sensors used.

It is already worth pointing out at the beginning that the project was simplified during development, and the final version only detects bicep curls and rest. Furthermore, only the accelerometer and gyroscope sensors are used. Only a CNN model was implemented.

# Data collection

As already described in the feasibility report, data has been outsourced from a public dataset:

> Morris, D., Saponas, T. S., Guillory, A., & Kelner, I. (2014, April). RecoFit: using a wearable sensor to find, recognize, and count repetitive exercises. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (pp. 3225-3234). ACM.
> Data Source: Microsoft Exercise Recognition from Wearable Sensors dataset (https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors)

## What has changed since the feasibility review?

Opposed to the original data processing pipeline, only one thing has changed: the number of exercises has been reduced to 2 (bicep curls and rest). This was done to simplify the model and make it more feasible to run on the target device, given the limited computational resources.

Additionally, the way of splitting the recordings into repetitions has been slightly modified. It still uses a peak (valley) detection algorithm, but rather than padding the repetitions to a fixed length (if they were shorter than the target length they would be padded with zeros), they were simply truncated to the target length. This was done to avoid introducing artificial data (zeros) into the model, which could potentially affect its performance.

# Firmware development