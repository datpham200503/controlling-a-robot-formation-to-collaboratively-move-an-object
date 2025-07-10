#include <ros.h>
#include <std_msgs/Bool.h>
#include <Servo.h>
#define X_STEP_PIN 2
#define X_DIR_PIN 5
#define Y_STEP_PIN 3
#define Y_DIR_PIN 6
#define Z_STEP_PIN 4
#define Z_DIR_PIN 7
#define ENABLE_PIN 8

#define end_X 9
#define end_Y 10
#define end_Z 11

Servo myservo;

// ROS NodeHandle
ros::NodeHandle nh;

// Hàm callback cho topic /setup
void setupCallback(const std_msgs::Bool& msg) {
  if (msg.data == true) {
    // Thực hiện Home cho cả ba trục
    Home_X();
    Home_YZ();

  } if (msg.data == false) {
    // Thực hiện Pos()
    Pos();
  }
}

// Hàm callback cho topic /gripper
void gripperCallback(const std_msgs::Bool& msg) {
  if (msg.data == true) {

    myservo.write(180);

  } if (msg.data == false) {
    
    myservo.write(0);
  }
}

// Subscriber cho topic /setup
ros::Subscriber<std_msgs::Bool> sub_setup("/robot1/setup", &setupCallback);
ros::Subscriber<std_msgs::Bool> sub_gripper("/robot1/gripper", &gripperCallback);

void Home_X() {
  digitalWrite(X_DIR_PIN, HIGH);
  while (digitalRead(end_X) == 1) {
    digitalWrite(X_STEP_PIN, HIGH);
    delayMicroseconds(10000);
    digitalWrite(X_STEP_PIN, LOW);
    delayMicroseconds(10000);
  }
}

void Home_Y() {
  digitalWrite(Y_DIR_PIN, HIGH);
  while (digitalRead(end_Y) == 1) {
    digitalWrite(Y_STEP_PIN, HIGH);
    delayMicroseconds(10000);
    digitalWrite(Y_STEP_PIN, LOW);
    delayMicroseconds(10000);
  }
}

void Home_Z() {
  digitalWrite(Z_DIR_PIN, LOW);
  while (digitalRead(end_Z) == 1) {
    digitalWrite(Z_STEP_PIN, HIGH);
    delayMicroseconds(10000);
    digitalWrite(Z_STEP_PIN, LOW);
    delayMicroseconds(10000);
  }
}

void Home_YZ()
{
  digitalWrite(Y_DIR_PIN, HIGH);
  digitalWrite(Z_DIR_PIN, LOW);
    while(digitalRead(end_Y)==1 and digitalRead(end_Z)==1)
  {
    digitalWrite(Y_STEP_PIN, HIGH);
    digitalWrite(Z_STEP_PIN, HIGH);
    delayMicroseconds(10000);
    digitalWrite(Y_STEP_PIN, LOW);
    digitalWrite(Z_STEP_PIN, LOW);
    delayMicroseconds(10000);
  }

    while(digitalRead(end_Y)==1)
  {
    digitalWrite(Y_STEP_PIN, HIGH);
    delayMicroseconds(10000);
    digitalWrite(Y_STEP_PIN, LOW);
    delayMicroseconds(10000);
  }

   while(digitalRead(end_Z)==1)
  {
    digitalWrite(Z_STEP_PIN, HIGH);
    delayMicroseconds(10000);
    digitalWrite(Z_STEP_PIN, LOW);
    delayMicroseconds(10000);
  }
}
void Pos() {
  digitalWrite(Y_DIR_PIN, LOW);
  digitalWrite(Z_DIR_PIN, HIGH);
    for (int i = 0; i < 60; i++) {
    digitalWrite(Z_STEP_PIN, HIGH);
    delayMicroseconds(10000);
    digitalWrite(Z_STEP_PIN, LOW);
    delayMicroseconds(10000);
  }
  for (int i = 0; i < 180; i++) {
    digitalWrite(Y_STEP_PIN, HIGH);
    digitalWrite(Z_STEP_PIN, HIGH);
    delayMicroseconds(10000);
    digitalWrite(Y_STEP_PIN, LOW);
    digitalWrite(Z_STEP_PIN, LOW);
    delayMicroseconds(10000);
  }
  digitalWrite(Z_DIR_PIN, LOW);
      for (int i = 0; i < 60; i++) {
    digitalWrite(Z_STEP_PIN, HIGH);
    delayMicroseconds(10000);
    digitalWrite(Z_STEP_PIN, LOW);
    delayMicroseconds(10000);
  }
}

void setup() {
  Serial.begin(115200);
  nh.getHardware()->setBaud(115200);
  nh.initNode();
  nh.subscribe(sub_setup);
  nh.subscribe(sub_gripper);
  // Cấu hình các chân
  pinMode(X_STEP_PIN, OUTPUT);
  pinMode(X_DIR_PIN, OUTPUT);
  pinMode(Y_STEP_PIN, OUTPUT);
  pinMode(Y_DIR_PIN, OUTPUT);
  pinMode(Z_STEP_PIN, OUTPUT);
  pinMode(Z_DIR_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(end_X, INPUT_PULLUP);
  pinMode(end_Y, INPUT_PULLUP);
  pinMode(end_Z, INPUT_PULLUP);

  // Khởi tạo trạng thái ban đầu
  digitalWrite(X_DIR_PIN, LOW);
  digitalWrite(Y_DIR_PIN, LOW);
  digitalWrite(Z_DIR_PIN, LOW);
  digitalWrite(ENABLE_PIN, LOW); // Kích hoạt động cơ

  myservo.attach(12);


}

void loop() {
  nh.spinOnce();
  delay(10); // Delay nhỏ để tránh quá tải
}
