#include <WiFi.h>
#include <Wire.h>
#include <DHT.h>
#include <MPU6050.h>

#include <Firebase_ESP_Client.h>
#include <addons/TokenHelper.h>
#include <addons/RTDBHelper.h>

/********************* PIN DEFINITIONS *********************/
#define DHTPIN 4
#define DHTTYPE DHT22

#define ACS_PIN 34
#define RELAY_PIN 27

#define LED_R 19
#define LED_Y 18
#define LED_G 5

/********************* DIVIDER CONSTANTS *********************/
#define R1 10000.0
#define R2 22000.0
#define DIVIDER_SCALE (R2 / (R1 + R2))                                           

/********************* ACS712 CONSTANTS *********************/
#define ACS_SENSITIVITY 0.185
#define ADC_REF 3.3
#define ADC_MAX 4095.0

float acsZero = 0.0;
float lastKnownTemp = 25.0;

/********************* OBJECTS *********************/
DHT dht(DHTPIN, DHTTYPE);
MPU6050 mpu;

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

/********************* WIFI & FIREBASE *********************/
const char* ssid = "Nikhil";
const char* password = "Nikhil11";

String API_KEY = "AIzaSyAcZwPuoT_0e2AEpZimVnLbrMtutQHiKWI";
String DATABASE_URL = "https://prediction-system-8200f-default-rtdb.asia-southeast1.firebasedatabase.app";

/************************************************************
   FIX: Temperature retry + smoothing
*************************************************************/
float readTemp() {
  float t = dht.readTemperature();
  if (!isnan(t)) { lastKnownTemp = t; return t; }

  delay(200);
  t = dht.readTemperature();
  if (!isnan(t)) { lastKnownTemp = t; return t; }

  // If still failing, return previous good value
  return lastKnownTemp;
}

/********************* CALIBRATE ACS ZERO *********************/
void calibrateACS() {
  long sum = 0;
  for (int i = 0; i < 500; i++) {
    sum += analogRead(ACS_PIN);
    delay(2);
  }

  float raw = sum / 500.0;
  float vAdc = (raw / ADC_MAX) * ADC_REF;
  acsZero = vAdc / DIVIDER_SCALE;

  Serial.print("ACS Zero Calibrated = ");
  Serial.println(acsZero, 4);
}

/********************* RMS CURRENT READING *********************/
float readCurrentRMS(int samples = 300, int delayMicrosVal = 1000) {
  double sumSq = 0;

  for (int i = 0; i < samples; i++) {
    int raw = analogRead(ACS_PIN);
    float vAdc = (raw / ADC_MAX) * ADC_REF;
    float vSensor = vAdc / DIVIDER_SCALE;

    float vDiff = vSensor - acsZero;
    float instCurrent = vDiff / ACS_SENSITIVITY;

    sumSq += instCurrent * instCurrent;
    delayMicroseconds(delayMicrosVal);
  }

  return sqrt(sumSq / samples);
}

/********************* LED STATUS FUNCTION *********************/
void setLEDstatus(bool machineOn, bool danger, bool sensorError) {

  if (!machineOn) {
    digitalWrite(LED_R, LOW);
    digitalWrite(LED_Y, HIGH);
    digitalWrite(LED_G, LOW);
    return;
  }

  if (danger) {
    digitalWrite(LED_R, HIGH);
    digitalWrite(LED_Y, LOW);
    digitalWrite(LED_G, LOW);
    return;
  }

  if (sensorError) {
    digitalWrite(LED_R, LOW);
    digitalWrite(LED_Y, HIGH);
    digitalWrite(LED_G, LOW);
    return;
  }

  digitalWrite(LED_R, LOW);
  digitalWrite(LED_Y, LOW);
  digitalWrite(LED_G, HIGH);
}

/********************* SETUP *********************/
void setup() {
  Serial.begin(115200);

  dht.begin();
  Wire.begin(21, 22);
  mpu.initialize();

  pinMode(RELAY_PIN, OUTPUT);
  pinMode(LED_R, OUTPUT);
  pinMode(LED_Y, OUTPUT);
  pinMode(LED_G, OUTPUT);

  digitalWrite(RELAY_PIN, HIGH); // machine OFF initially

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(300);
  }
  Serial.println("\nWiFi Connected!");

  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  config.token_status_callback = tokenStatusCallback;

  Firebase.signUp(&config, &auth, "", "");
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  calibrateACS();
}

/********************* LOOP *********************/
void loop() {

  float temp = readTemp();
  float hum = dht.readHumidity();

  bool sensorError = isnan(hum);
  if (sensorError) hum = 0;

  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  float accelTotal = sqrt((double)ax*ax + ay*ay + az*az) / 16384.0;
  bool vibration = accelTotal > 1.20;

  float currentRMS = readCurrentRMS();

  Serial.print("Temp: "); Serial.print(temp);
  Serial.print(" | Current: "); Serial.print(currentRMS);
  Serial.print(" | Vib: "); Serial.println(vibration);

  bool danger = false;

  if (temp > 65 || currentRMS > 2.5 || vibration) {
    danger = true;
  }

  Firebase.RTDB.setInt(&fbdo, "machine/safe", danger ? 0 : 1);

  int userRequest = 0;
  Firebase.RTDB.getInt(&fbdo, "control/user_request", &userRequest);

  bool machineOn = false;

  if (danger) {
    digitalWrite(RELAY_PIN, HIGH);
    machineOn = false;
  }
  else if (userRequest == 0) {
    digitalWrite(RELAY_PIN, HIGH);
    machineOn = false;
  }
  else {
    digitalWrite(RELAY_PIN, LOW);
    machineOn = true;
  }

  setLEDstatus(machineOn, danger, sensorError);

  Firebase.RTDB.setFloat(&fbdo, "machine/temp", temp);
  Firebase.RTDB.setFloat(&fbdo, "machine/current", currentRMS);
  Firebase.RTDB.setInt(&fbdo, "machine/vibration", vibration);
  Firebase.RTDB.setInt(&fbdo, "machine/status", danger ? 2 : 0);

  int failureChance = danger ? 90 : 10;
  int remainingLife = 100 - failureChance;

  Firebase.RTDB.setInt(&fbdo, "predict/failure_chance", failureChance);
  Firebase.RTDB.setInt(&fbdo, "predict/remaining_life", remainingLife);

  delay(1500);
}