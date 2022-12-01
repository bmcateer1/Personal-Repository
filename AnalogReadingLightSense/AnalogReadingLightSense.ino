#include<Servo.h>
#include<Stepper.h>


int led1 = 6;
int light = A0; //reads between 0 and 5 v and returns value from 0 to 1023 (10 bits)
int servopin = 3;
int stepsperRev = 800;

Servo Servo1;
Stepper myStepper(stepsperRev,8,10,9,11);


void setup() {
  // put your setup code here, to run once:
   Serial.begin(9600);
   pinMode(led1,OUTPUT);
   
   //Servo1.attach(servopin);

   myStepper.setSpeed(20);
   
}

void loop() {
  // put your main code here, to run repeatedly:
  light = analogRead(A0);
  Serial.println(light);
  //delay(100);

//  Servo1.write(180);
//  delay(500);
//  Servo1.write(0);
//  delay(500);
 

  if(light < 750)
  {
    digitalWrite(led1,HIGH);
    myStepper.step(stepsperRev);
    myStepper.step(-stepsperRev);
  }
  
  else
  {
    digitalWrite(led1,LOW);
  }
}
