
#if defined(ARDUINO) && ARDUINO >= 100
  #include "Arduino.h"
#else
  #include <WProgram.h>
#endif



#include "VNH3SP30.h"
#include <SPI.h>
#include <ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/String.h>

#define M1SpeedPin 3
#define M1InAPin 2
#define M1InBPin 4

#define M2SpeedPin 6
#define M2InAPin 5
#define M2InBPin 7



VNH3SP30 MotorLeft(M1SpeedPin, M1InAPin, M1InBPin);
VNH3SP30 MotorRight(M2SpeedPin, M2InAPin, M2InBPin);

ros::NodeHandle nh;

std_msgs::String str_msg;

float theta_dot = 0, x_dot = 0, omega_left = 0, omega_right = 0;
bool leftForward, rightForward;

float theta_scale = 0.4666;
float linear_scale = 0.59;
float throttle_threshold = 0.04;

void controlCallback( const geometry_msgs::Twist& twist_msg){
  //
  if(twist_msg.linear.x ==0&&twist_msg.angular.z==0){
    theta_dot = 0;
    x_dot = 0;
  }
  else{
    theta_dot = twist_msg.angular.z * theta_scale;
    x_dot = twist_msg.linear.x * linear_scale;
  }
//  
  omega_left = x_dot + theta_dot;
  omega_right = x_dot - theta_dot;
  //Serial.println("   omega_left:  "+String(omega_left)+"   omega_right:  "+String(omega_right));
  //leftForward = false; rightForward = false;
//  MotorLeft.Throttle(((omega_left)));
//  MotorRight.Throttle(((omega_right)));

  if(-throttle_threshold > omega_right || omega_right > throttle_threshold){
    MotorRight.Throttle(omega_right * 1.4);
  } else {
    MotorRight.Stop();
  }
  
  if(-throttle_threshold > omega_left || omega_left > throttle_threshold){
    MotorLeft.Throttle(omega_left);
  } else{
    MotorLeft.Stop();
  }
  

    //     leftForward = false;
  //     rightForward = false;
}


// Slave Select pins for encoders 1 and 2
// Feel free to reallocate these pins to best suit your circuit
// const int slaveSelectEnc1 = 7;
// const int slaveSelectEnc2 = 8;

// // These hold the current encoder count.
// signed long encoder1count = 0;
// signed long encoder2count = 0;

ros::Subscriber<geometry_msgs::Twist> control_sub("/chassis/cmd_vel", &controlCallback);

ros::Publisher arduino_heartbeat("/chassis/heartbeat", &str_msg);

void setup() {
  Serial.begin(57600);      // Serial com for data output
  pinMode(LED_BUILTIN, OUTPUT);
  // put your setup code here, to run once:
  MotorLeft.Stop();
  MotorRight.Stop();
  nh.initNode();
  nh.advertise(arduino_heartbeat);
  nh.subscribe(control_sub);
  // Loop Driving Forward for a small while and count ticks for the entire amount of time motor was in usefor ()
  
  
}


void loop() {
  
  // Expected inputs from ROS are vehicle speed and rotational speed
  // put your main code here, to run repeatedly:

  // Decide Direction of motors based on control inputs
  nh.spinOnce();
  char buff[50];
  sprintf(buff, "Alive theta=%d, x=%d", theta_dot*1000, x_dot*1000);
  str_msg.data = buff;
  arduino_heartbeat.publish( &str_msg );
  delay(250);
  if (theta_dot == 0 || x_dot == 0) {
    digitalWrite(LED_BUILTIN, HIGH);
  }
//  if(-throttle_threshold > omega_right || omega_right > throttle_threshold){
//    MotorRight.Throttle(omega_right);
//  } else {
//    MotorRight.Stop();
//  }
//  
//  if(-throttle_threshold > omega_left || omega_left > throttle_threshold){
//    MotorLeft.Throttle(omega_left);
//  } else{
//    MotorLeft.Stop();
//  }
  delay(250);
  digitalWrite(LED_BUILTIN, LOW);
  //Serial.println("AHHHHHHHHHHH");
  
}
