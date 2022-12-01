int A = 2;
int B = 3;
int C = 4;
int D = 5;
int E = 6;
int F = 7;
int G = 8;
int H = 9;

void setup() 
{
  // put your setup code here, to run once:
  Serial.begin(9600);

  pinMode(A,OUTPUT);
  pinMode(B,OUTPUT);
  pinMode(C,OUTPUT);
  pinMode(D,OUTPUT);
  pinMode(E,OUTPUT);
  pinMode(F,OUTPUT);
  pinMode(G,OUTPUT);
  pinMode(H,OUTPUT);
  
}

void loop() 
{
  // put your main code here, to run repeatedly:int

  makezero();
  delay(1000);
  makeone();
  delay(1000);
  
}

void makezero()
{
  digitalWrite(A, LOW); //middle bar
  digitalWrite(B, HIGH); //top left side bar
  digitalWrite(C, HIGH); // top bar
  digitalWrite(D, HIGH);  //top right side bar
  digitalWrite(E, LOW);  //decimal
  digitalWrite(F, HIGH);  //bottom right bar
  digitalWrite(G, HIGH); // bottom bar
  digitalWrite(H, HIGH); // bottom left  bar
}


void makeone()
{
  digitalWrite(A, LOW); //middle bar
  digitalWrite(B, LOW); //top left side bar
  digitalWrite(C, LOW); // top bar
  digitalWrite(D, HIGH);  //top right side bar
  digitalWrite(E, LOW);  //decimal
  digitalWrite(F, HIGH);  //bottom right bar
  digitalWrite(G, LOW); // bottom bar
  digitalWrite(H, LOW); // bottom left  bar
}
