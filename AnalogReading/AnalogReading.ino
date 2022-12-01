int led1 = 9;
int x = analogRead(A0); //reads between 0 and 5 v and returns value from 0 to 1023 (10 bits)
 

void setup() {
  // put your setup code here, to run once:
   Serial.begin(9600);
   pinMode(led1,OUTPUT);
   
}

void loop() {
  // put your main code here, to run repeatedly:
  x = analogRead(A0);
  Serial.println(analogRead(A0));
  Serial.println(x);
  delay(100);
  
  if(analogRead(A0) < 750)
  {
    digitalWrite(led1,HIGH);
  }
  else
  {
    digitalWrite(led1,LOW);
  }
}
