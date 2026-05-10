#include <Arduino.h>
#include <LSM9DS1.h>

LSM9DS1Class* sensor;
void setup() {
  Serial.println("Initializing sensor...");

  sensor = new LSM9DS1Class(Wire1);
  if (!sensor->begin()) {
    Serial.println("Failed to initialize sensor!");
    while (1);
  } else {
    Serial.println("Sensor initialized successfully.");
  }
  sensor->setContinuousMode();
}

void loop() {
  delay(500);
  while(sensor->accelerationAvailable()) {
    float x, y, z;
    if (sensor->readAcceleration(x, y, z)) {
      Serial.print("Acceleration: ");
      Serial.print(x);
      Serial.print(", ");
      Serial.print(y);
      Serial.print(", ");
      Serial.println(z);
    }
  }
}