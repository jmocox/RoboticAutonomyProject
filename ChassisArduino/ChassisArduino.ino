#include "VNH3SP30.h"
#include <SPI.h>
#include <ros.h>
#include <geometry_msgs/Twist.h>

#define M1SpeedPin 3
#define M1InAPin 2
#define M1InBPin 4

#define M2SpeedPin 6
#define M2InAPin 5
#define M2InBPin 7

VNH3SP30 MotorLeft(M1SpeedPin, M1InAPin, M1InBPin);
VNH3SP30 MotorRight(M2SpeedPin, M2InAPin, M2InBPin);

ros::NodeHandle nh;

float theta_dot, x_dot, omega_left, omega_right;
bool leftForward, rightForward;

void controlCallback( const geometry_msgs::Twist& twist_msg){
  theta_dot = twist_msg.angular.z;
  x_dot = twist_msg.linear.x;
  omega_left = x_dot+theta_dot/4;
  omega_right = x_dot-theta_dot/4;
  //Serial.println("   omega_left:  "+String(omega_left)+"   omega_right:  "+String(omega_right));
  //leftForward = false; rightForward = false;
//  MotorLeft.Throttle(((omega_left)));
//  MotorRight.Throttle(((omega_right)));

  if(-0.01 > omega_right || omega_right > 0.01){
    MotorRight.Throttle(omega_right);
  } else {
    MotorRight.Stop();
  }
  
  if(-0.01 > omega_left || omega_left > 0.01){
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

ros::Subscriber<geometry_msgs::Twist> control_sub("chassis/cmd_vel", &controlCallback);

void setup() {
  Serial.begin(9600);      // Serial com for data output
  // put your setup code here, to run once:
  MotorLeft.Stop();
  MotorRight.Stop();
  nh.initNode();
  nh.subscribe(control_sub);
  // Loop Driving Forward for a small while and count ticks for the entire amount of time motor was in usefor ()
  
  
}


void loop() {
  
  // Expected inputs from ROS are vehicle speed and rotational speed
  // put your main code here, to run repeatedly:

  // Decide Direction of motors based on control inputs
  nh.spinOnce();
  delay(1);
  //Serial.println("AHHHHHHHHHHH");


  // if(thetadot != 0 || xdot != 0) {
  //   if(thetadot > 0){ //Right hand rule is positive rotation (Z is thumb upward)
  //     leftForward = false;
  //     rightForward = true;
  //   }
  //   if(thetadot <0) {
  //     leftForward = true;
  //     rightForward = false;
  //   }if(xdot < 0 && thetadot ==0){
  //     leftForward = false;
  //     rightForward = false;
  //   }
  //     if(xdot > 0 && thetadot ==0){
  //     leftForward = true;
  //     rightForward = true;
  //   }

  //   MotorLeft(toInt(xdot*255), leftForward);
  //   MotorRight(toInt(xdot*255), rightForward);
  // }
  // else() {
  //   leftForward = true;
  //   rightForward = true;
  //   MotorLeft.Stop();
  //   MotorRight.Stop();
  // }
}
