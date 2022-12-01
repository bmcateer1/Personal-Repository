int led1 = 7;
int tig1 = 12;
int eho1 = 13;

int unit = 100;
int letter_space = unit* 3;
int word_space = unit * 7;
int dot = unit;
int gap = unit;
int dash = 3 * unit;

float duration = 0;
float distance = 0;


void setup() {
  // put your setup code here, to run once:
//##abdeghijklmnopqstuvwxyz12345670
//cfr88899
  pinMode(led1,OUTPUT);
  pinMode(tig1,OUTPUT);
  pinMode(eho1,INPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
  getDistance();
  if(distance < 30 && distance > 10) 
  {
    SOS();
  }
  else 
  {
    digitalWrite(led1,LOW);
  }
  
}
//void simply means nothing is errreturned , you can erturn integers by hcanging void to int

void getDistance()
{
  digitalWrite(tig1,LOW);
  delayMicroseconds(2);
  digitalWrite(tig1,HIGH);
  delayMicroseconds(100);
  digitalWrite(tig1,LOW);

  duration = pulseIn(eho1,HIGH);

  distance = duration * 0.034 / 2; 
}

void SOS()
{    
     digitalWrite(led1, HIGH);
     delay(dot);
     digitalWrite(led1, LOW);
     delay(gap);
     digitalWrite(led1, HIGH);
     delay(dot);
     digitalWrite(led1, LOW);
     delay(gap);
     digitalWrite(led1, HIGH);
     delay(dot);
     digitalWrite(led1, LOW);
     delay(letter_space);
     digitalWrite(led1, HIGH);
     delay(dash);
     digitalWrite(led1, LOW);
     delay(gap);
     digitalWrite(led1, HIGH);
     delay(dash);
     digitalWrite(led1, LOW);
     delay(gap);
     digitalWrite(led1, HIGH);
     delay(dash);
     digitalWrite(led1, LOW);
     delay(letter_space);
     digitalWrite(led1, HIGH);
     delay(dot);
     digitalWrite(led1, LOW);
     delay(gap);
     digitalWrite(led1, HIGH);
     delay(dot);
     digitalWrite(led1, LOW);
     delay(gap);
     digitalWrite(led1, HIGH);
     delay(dot); 
}
