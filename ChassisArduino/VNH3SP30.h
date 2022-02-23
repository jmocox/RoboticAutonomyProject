
#ifndef VNH3SP30_H
#define VNH3SP30_H


#include <Arduino.h>

class VNH3SP30{
      
      public:
              VNH3SP30(int _PWMPIN, int _INAPIN, int _INBPIN);
              ~VNH3SP30();
      
              void Move(byte Speed, bool Forward); 
              void Stop() ;
              void Coast(); 
               
                                     
      private:
			  int PWMPIN, INAPIN, INBPIN;
              void SetPWMA(byte Value); 
      
      
};

#endif
