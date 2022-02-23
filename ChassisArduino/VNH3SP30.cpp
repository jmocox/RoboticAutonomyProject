#include "VNH3SP30.h"

//-- public methods --//

//<<constructor>>
VNH3SP30::VNH3SP30(int _PWMPIN, int _INAPIN, int _INBPIN){
  
 PWMPIN = _PWMPIN;
 INAPIN = _INAPIN;
 INBPIN = _INBPIN;

 
  pinMode(PWMPIN, OUTPUT);
  pinMode(INAPIN, OUTPUT);
  pinMode(INBPIN, OUTPUT);
  
   
}

VNH3SP30::~VNH3SP30(){};

void VNH3SP30::Move(byte Speed, bool Forward) {
	SetPWMA(Speed); 
	digitalWrite(INAPIN, Forward);
	digitalWrite(INBPIN, !Forward);
}

void VNH3SP30::Stop() {
	SetPWMA(255); 
	digitalWrite(INAPIN, LOW);
	digitalWrite(INBPIN, LOW);
}

void VNH3SP30::Coast(){
	
	SetPWMA(255);
	digitalWrite(INAPIN, LOW);
	digitalWrite(INBPIN, LOW);
	
}
//-- private methods --//
void VNH3SP30::SetPWMA(byte Value) {
	analogWrite(PWMPIN, Value);
  
}
