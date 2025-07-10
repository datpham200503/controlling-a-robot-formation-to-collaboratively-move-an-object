#include <ros.h>
#include <geometry_msgs/Twist.h>
#include <TimerOne.h>
#include <std_msgs/Int16MultiArray.h>

// PID parameters
float KP[] = { 25, 25, 20, 20 };
float KI[] = { 0, 0, 0, 0 };
float KD[] = { 0.66, 0.5, 0.5, 0.5 };

// Robot parameters
double R = 0.0485; // Wheel radius (m)
double Lx = 0.289; // Half robot length (m)
double Ly = 0.210; // Half robot width (m)
float omega[4];

// Motor pins
// Front-left (left_st_0)
int left_st_0_pwm = 8, left_st_0_dir = 24, left_st_0_enb = 22;
int left_st_0_encoder_a = 2, left_st_0_encoder_b = 40;
// Front-right (right_st_1)
int right_st_1_pwm = 7, right_st_1_dir = 28, right_st_1_enb = 26;
int right_st_1_encoder_a = 3, right_st_1_encoder_b = 42;
// Rear-left (left_nd_2)
int left_nd_2_pwm = 5, left_nd_2_dir = 34, left_nd_2_enb = 36;
int left_nd_2_encoder_a = 21, left_nd_2_encoder_b = 44;
// Rear-right (right_nd_3)
int right_nd_3_pwm = 6, right_nd_3_dir = 30, right_nd_3_enb = 32;
int right_nd_3_encoder_a = 20, right_nd_3_encoder_b = 46;

// Encoder counters
volatile long pul_left_st_0 = 0, pul_right_st_1 = 0, pul_right_nd_3 = 0, pul_left_nd_2 = 0;
long pulPre_left_st_0 = 0, pulPre_right_st_1 = 0, pulPre_right_nd_3 = 0, pulPre_left_nd_2 = 0;
int delta_tick_left_st_0 = 0, delta_tick_right_st_1 = 0, delta_tick_right_nd_3 = 0, delta_tick_left_nd_2 = 0;

// Velocity variables
float velLinear_X = 0, velLinear_Y = 0, angLinear = 0;
float desiredVel_left_st_0 = 0, desiredVel_right_st_1 = 0, desiredVel_left_nd_2 = 0, desiredVel_right_nd_3 = 0;
float vel_left_st_0 = 0, vel_right_st_1 = 0, vel_right_nd_3 = 0, vel_left_nd_2 = 0;
float errVel_left_st_0 = 0, errVel_right_st_1 = 0, errVel_right_nd_3 = 0, errVel_left_nd_2 = 0;
float errTotal_left_st_0 = 0, errTotal_right_st_1 = 0, errTotal_right_nd_3 = 0, errTotal_left_nd_2 = 0;
float errPrev_left_st_0 = 0, errPrev_right_st_1 = 0, errPrev_right_nd_3 = 0, errPrev_left_nd_2 = 0;
float errDiff_left_st_0 = 0, errDiff_right_st_1 = 0, errDiff_right_nd_3 = 0, errDiff_left_nd_2 = 0;
int pwmVal_left_st_0 = 0, pwmVal_right_st_1 = 0, pwmVal_left_nd_2 = 0, pwmVal_right_nd_3 = 0;

// Timing
const double deltaT = 0.01; // 10ms for PID
const long interval_pub = 400; // 400ms for publishing encoder data
const long interval_nh = 2; // 2ms for motor control
long currentMillis = 0, previousMillis_pub = 0, previousMillis_nh = 0;

#define PI 3.1415926535897932384626433832795

ros::NodeHandle nh;
std_msgs::Int16MultiArray encoder;
ros::Publisher pubEncoder("/robot1/encoder", &encoder);

void calculateWheelSpeeds(float vx, float vy, float wz, float lx, float ly, float r, float omega[4]) {
  float T_inv[4][3] = {
    { 1 / r, -1 / r, -(lx + ly) / r },
    { 1 / r,  1 / r,  (lx + ly) / r },
    { 1 / r,  1 / r, -(lx + ly) / r },
    { 1 / r, -1 / r,  (lx + ly) / r }
  };
  for (int i = 0; i < 4; i++) {
    omega[i] = T_inv[i][0] * vx + T_inv[i][1] * vy + T_inv[i][2] * wz;
  }
}

void commandVelocityCallback(const geometry_msgs::Twist &cmd_vel_msg) {
  velLinear_X = (float)cmd_vel_msg.linear.x;
  velLinear_Y = (float)cmd_vel_msg.linear.y;
  angLinear = (float)cmd_vel_msg.angular.z;

  velLinear_X = constrain(velLinear_X, -(R * 2 * PI * 35 / 60), R * 2 * PI * 35 / 60);
  velLinear_Y = constrain(velLinear_Y, -(R * 2 * PI * 35 / 60), R * 2 * PI * 35 / 60);
  angLinear = constrain(angLinear, -(R * 2 * PI * 35 / 60) / 0.069, (R * 2 * PI * 35 / 60) / 0.069);

  calculateWheelSpeeds(velLinear_X, velLinear_Y, angLinear, Lx, Ly, R, omega);
  desiredVel_left_st_0  = omega[0] * (30 / PI);
  desiredVel_right_st_1 = omega[1] * (30 / PI);
  desiredVel_left_nd_2  = omega[2] * (30 / PI);
  desiredVel_right_nd_3 = omega[3] * (30 / PI);

  if (desiredVel_left_st_0 > 0) {
    desiredVel_left_st_0 = constrain(desiredVel_left_st_0, 1, 20);
  } else if (desiredVel_left_st_0 < 0) {
    desiredVel_left_st_0 = constrain(desiredVel_left_st_0, -20, -1);
  }
  if (desiredVel_right_st_1 > 0) {
    desiredVel_right_st_1 = constrain(desiredVel_right_st_1, 1, 20);
  } else if (desiredVel_right_st_1 < 0) {
    desiredVel_right_st_1 = constrain(desiredVel_right_st_1, -20, -1);
  }
  if (desiredVel_left_nd_2 > 0) {
    desiredVel_left_nd_2 = constrain(desiredVel_left_nd_2, 1, 20);
  } else if (desiredVel_left_nd_2 < 0) {
    desiredVel_left_nd_2 = constrain(desiredVel_left_nd_2, -20, -1);
  }
  if (desiredVel_right_nd_3 > 0) {
    desiredVel_right_nd_3 = constrain(desiredVel_right_nd_3, 1, 20);
  } else if (desiredVel_right_nd_3 < 0) {
    desiredVel_right_nd_3 = constrain(desiredVel_right_nd_3, -20, -1);
  }
}

ros::Subscriber<geometry_msgs::Twist> subCmdVel("/robot1/cmd_vel", &commandVelocityCallback);

void controlMotor(int valPWM, int PWM, int dir_D, int enb_D) {
  if (valPWM > 0) {
    digitalWrite(enb_D, 0);
    digitalWrite(dir_D, 1);
  } else if (valPWM < 0) {
    digitalWrite(enb_D, 1);
    digitalWrite(dir_D, 0);
  } else {
    digitalWrite(enb_D, 0);
    digitalWrite(dir_D, 0);
  }
  analogWrite(PWM, abs(valPWM));
}

void readEncoder_left_st_0() {
  if (digitalRead(left_st_0_encoder_b) == digitalRead(left_st_0_encoder_a)) pul_left_st_0++;
  else pul_left_st_0--;
}

void readEncoder_right_st_1() {
  if (digitalRead(right_st_1_encoder_b) == digitalRead(right_st_1_encoder_a)) pul_right_st_1--;
  else pul_right_st_1++;
}

void readEncoder_right_nd_3() {
  if (digitalRead(right_nd_3_encoder_b) == digitalRead(right_nd_3_encoder_a)) pul_right_nd_3--;
  else pul_right_nd_3++;
}

void readEncoder_left_nd_2() {
  if (digitalRead(left_nd_2_encoder_b) == digitalRead(left_nd_2_encoder_a)) pul_left_nd_2++;
  else pul_left_nd_2--;
}

void funcVelPID() {
  delta_tick_left_st_0 = pul_left_st_0 - pulPre_left_st_0;
  delta_tick_right_st_1 = pul_right_st_1 - pulPre_right_st_1;
  delta_tick_right_nd_3 = pul_right_nd_3 - pulPre_right_nd_3;
  delta_tick_left_nd_2 = pul_left_nd_2 - pulPre_left_nd_2;

  vel_left_st_0 = ((float)delta_tick_left_st_0 / 3000) * (1 / deltaT) * 60;
  vel_right_st_1 = ((float)delta_tick_right_st_1 / 3000) * (1 / deltaT) * 60;
  vel_right_nd_3 = ((float)delta_tick_right_nd_3 / 3000) * (1 / deltaT) * 60;
  vel_left_nd_2 = ((float)delta_tick_left_nd_2 / 3000) * (1 / deltaT) * 60;

  pulPre_left_st_0 = pul_left_st_0;
  pulPre_right_st_1 = pul_right_st_1;
  pulPre_right_nd_3 = pul_right_nd_3;
  pulPre_left_nd_2 = pul_left_nd_2;

  errVel_left_st_0 = desiredVel_left_st_0 - vel_left_st_0;
  errVel_right_st_1 = desiredVel_right_st_1 - vel_right_st_1;
  errVel_right_nd_3 = desiredVel_right_nd_3 - vel_right_nd_3;
  errVel_left_nd_2 = desiredVel_left_nd_2 - vel_left_nd_2;

  errTotal_left_st_0 += errVel_left_st_0 * deltaT;
  errTotal_right_st_1 += errVel_right_st_1 * deltaT;
  errTotal_right_nd_3 += errVel_right_nd_3 * deltaT;
  errTotal_left_nd_2 += errVel_left_nd_2 * deltaT;

  errDiff_left_st_0 = (errVel_left_st_0 - errPrev_left_st_0) / deltaT;
  errDiff_right_st_1 = (errVel_right_st_1 - errPrev_right_st_1) / deltaT;
  errDiff_right_nd_3 = (errVel_right_nd_3 - errPrev_right_nd_3) / deltaT;
  errDiff_left_nd_2 = (errVel_left_nd_2 - errPrev_left_nd_2) / deltaT;

  errPrev_left_st_0 = errVel_left_st_0;
  errPrev_right_st_1 = errVel_right_st_1;
  errPrev_right_nd_3 = errVel_right_nd_3;
  errPrev_left_nd_2 = errVel_left_nd_2;

  pwmVal_left_st_0 = errVel_left_st_0 * KP[0] + errTotal_left_st_0 * KI[0] + errDiff_left_st_0 * KD[0];
  pwmVal_right_st_1 = errVel_right_st_1 * KP[1] + errTotal_right_st_1 * KI[1] + errDiff_right_st_1 * KD[1];
  pwmVal_left_nd_2 = errVel_left_nd_2 * KP[2] + errTotal_left_nd_2 * KI[2] + errDiff_left_nd_2 * KD[2];
  pwmVal_right_nd_3 = errVel_right_nd_3 * KP[3] + errTotal_right_nd_3 * KI[3] + errDiff_right_nd_3 * KD[3];

  pwmVal_left_st_0 = constrain(pwmVal_left_st_0, -255, 255);
  pwmVal_right_st_1 = constrain(pwmVal_right_st_1, -255, 255);
  pwmVal_right_nd_3 = constrain(pwmVal_right_nd_3, -255, 255);
  pwmVal_left_nd_2 = constrain(pwmVal_left_nd_2, -255, 255);
}

void publishEncoder() {
  encoder.data_length = 4;
  encoder.data = (int*)malloc(encoder.data_length * sizeof(int));
  encoder.data[0] = pul_left_st_0;  // Front-left encoder pulses
  encoder.data[1] = pul_right_st_1; // Front-right encoder pulses
  encoder.data[2] = pul_left_nd_2;  // Rear-left encoder pulses
  encoder.data[3] = pul_right_nd_3; // Rear-right encoder pulses
  pubEncoder.publish(&encoder);
  free(encoder.data);
}

void setup() {
  Serial.begin(115200);

  // Configure motor pins
  pinMode(left_st_0_pwm, OUTPUT);
  pinMode(left_st_0_dir, OUTPUT);
  pinMode(left_st_0_enb, OUTPUT);
  pinMode(right_st_1_pwm, OUTPUT);
  pinMode(right_st_1_dir, OUTPUT);
  pinMode(right_st_1_enb, OUTPUT);
  pinMode(left_nd_2_pwm, OUTPUT);
  pinMode(left_nd_2_dir, OUTPUT);
  pinMode(left_nd_2_enb, OUTPUT);
  pinMode(right_nd_3_pwm, OUTPUT);
  pinMode(right_nd_3_dir, OUTPUT);
  pinMode(right_nd_3_enb, OUTPUT);

  // Configure PWM frequency
  TCCR2B = TCCR2B & B11111000 | B00000001;

  // Configure encoder pins
  pinMode(left_st_0_encoder_a, INPUT_PULLUP);
  pinMode(left_st_0_encoder_b, INPUT);
  pinMode(right_st_1_encoder_a, INPUT_PULLUP);
  pinMode(right_st_1_encoder_b, INPUT);
  pinMode(left_nd_2_encoder_a, INPUT_PULLUP);
  pinMode(left_nd_2_encoder_b, INPUT);
  pinMode(right_nd_3_encoder_a, INPUT_PULLUP);
  pinMode(right_nd_3_encoder_b, INPUT);

  // Attach interrupts for encoders
  attachInterrupt(digitalPinToInterrupt(left_st_0_encoder_a), readEncoder_left_st_0, RISING);
  attachInterrupt(digitalPinToInterrupt(right_st_1_encoder_a), readEncoder_right_st_1, RISING);
  attachInterrupt(digitalPinToInterrupt(left_nd_2_encoder_a), readEncoder_left_nd_2, RISING);
  attachInterrupt(digitalPinToInterrupt(right_nd_3_encoder_a), readEncoder_right_nd_3, RISING);

  // Initialize ROS
  nh.getHardware()->setBaud(115200);
  nh.initNode();
  nh.advertise(pubEncoder);
  nh.subscribe(subCmdVel);

  // Initialize Timer1 for PID
  Timer1.initialize(10000); // 10ms
  Timer1.attachInterrupt(funcVelPID);

  // Initialize encoder message
  encoder.data_length = 4;
  encoder.data = (int*)malloc(encoder.data_length * sizeof(int));
}

void loop() {
  nh.spinOnce();
  currentMillis = millis();

  // Control motors every 2ms
  if (currentMillis - previousMillis_nh > interval_nh) {
    controlMotor(pwmVal_left_st_0, left_st_0_pwm, left_st_0_dir, left_st_0_enb);
    controlMotor(pwmVal_right_st_1, right_st_1_pwm, right_st_1_dir, right_st_1_enb);
    controlMotor(pwmVal_left_nd_2, left_nd_2_pwm, left_nd_2_dir, left_nd_2_enb);
    controlMotor(pwmVal_right_nd_3, right_nd_3_pwm, right_nd_3_dir, right_nd_3_enb);
    previousMillis_nh = currentMillis;
  }

  // Publish encoder data every 400ms
  if (currentMillis - previousMillis_pub > interval_pub) {
    publishEncoder();
    previousMillis_pub = currentMillis;
  }
}
