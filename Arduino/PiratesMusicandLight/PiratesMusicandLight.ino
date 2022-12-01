int A = 2;
int B = 3;
int C = 4;
int D = 5;
int E = 6;
int F = 7;
int G = 8;
int H = 9;

int LED1 = 10;
int LED2 = 11;
int LED3 = 13;

// Define pin 10 for buzzer, you can use any other digital pins (Pin 0-13)
const int buzzer = 12;

// Change to 0.5 for a slower version of the song, 1.25 for a faster version
const float songSpeed = 1.0;

// Defining frequency of each music note
#define NOTE_C4 262
#define NOTE_D4 294
#define NOTE_E4 330
#define NOTE_F4 349
#define NOTE_G4 392
#define NOTE_A4 440
#define NOTE_B4 494
#define NOTE_C5 523
#define NOTE_D5 587
#define NOTE_E5 659
#define NOTE_F5 698
#define NOTE_G5 784
#define NOTE_A5 880
#define NOTE_B5 988

// Music notes of the song, 0 is a rest/pulse
int notes[] = {
    NOTE_E4, NOTE_G4, NOTE_A4, NOTE_A4, 0,
    NOTE_A4, NOTE_B4, NOTE_C5, NOTE_C5, 0,
    NOTE_C5, NOTE_D5, NOTE_B4, NOTE_B4, 0,
    NOTE_A4, NOTE_G4, NOTE_A4, 0,

    NOTE_E4, NOTE_G4, NOTE_A4, NOTE_A4, 0,
    NOTE_A4, NOTE_B4, NOTE_C5, NOTE_C5, 0,
    NOTE_C5, NOTE_D5, NOTE_B4, NOTE_B4, 0,
    NOTE_A4, NOTE_G4, NOTE_A4, 0,

    NOTE_E4, NOTE_G4, NOTE_A4, NOTE_A4, 0,
    NOTE_A4, NOTE_C5, NOTE_D5, NOTE_D5, 0,
    NOTE_D5, NOTE_E5, NOTE_F5, NOTE_F5, 0,
    NOTE_E5, NOTE_D5, NOTE_E5, NOTE_A4, 0,

    NOTE_A4, NOTE_B4, NOTE_C5, NOTE_C5, 0,
    NOTE_D5, NOTE_E5, NOTE_A4, 0,
    NOTE_A4, NOTE_C5, NOTE_B4, NOTE_B4, 0,
    NOTE_C5, NOTE_A4, NOTE_B4, 0,

    NOTE_A4, NOTE_A4,
    //Repeat of first part
    NOTE_A4, NOTE_B4, NOTE_C5, NOTE_C5, 0,
    NOTE_C5, NOTE_D5, NOTE_B4, NOTE_B4, 0,
    NOTE_A4, NOTE_G4, NOTE_A4, 0,

    NOTE_E4, NOTE_G4, NOTE_A4, NOTE_A4, 0,
    NOTE_A4, NOTE_B4, NOTE_C5, NOTE_C5, 0,
    NOTE_C5, NOTE_D5, NOTE_B4, NOTE_B4, 0,
    NOTE_A4, NOTE_G4, NOTE_A4, 0,

    NOTE_E4, NOTE_G4, NOTE_A4, NOTE_A4, 0,
    NOTE_A4, NOTE_C5, NOTE_D5, NOTE_D5, 0,
    NOTE_D5, NOTE_E5, NOTE_F5, NOTE_F5, 0,
    NOTE_E5, NOTE_D5, NOTE_E5, NOTE_A4, 0,

    NOTE_A4, NOTE_B4, NOTE_C5, NOTE_C5, 0,
    NOTE_D5, NOTE_E5, NOTE_A4, 0,
    NOTE_A4, NOTE_C5, NOTE_B4, NOTE_B4, 0,
    NOTE_C5, NOTE_A4, NOTE_B4, 0,
    //End of Repeat

    NOTE_E5, 0, 0, NOTE_F5, 0, 0,
    NOTE_E5, NOTE_E5, 0, NOTE_G5, 0, NOTE_E5, NOTE_D5, 0, 0,
    NOTE_D5, 0, 0, NOTE_C5, 0, 0,
    NOTE_B4, NOTE_C5, 0, NOTE_B4, 0, NOTE_A4,

    NOTE_E5, 0, 0, NOTE_F5, 0, 0,
    NOTE_E5, NOTE_E5, 0, NOTE_G5, 0, NOTE_E5, NOTE_D5, 0, 0,
    NOTE_D5, 0, 0, NOTE_C5, 0, 0,
    NOTE_B4, NOTE_C5, 0, NOTE_B4, 0, NOTE_A4};

// Durations (in ms) of each music note of the song
// Quarter Note is 250 ms when songSpeed = 1.0
int durations[] = {
    125, 125, 250, 125, 125,
    125, 125, 250, 125, 125,
    125, 125, 250, 125, 125,
    125, 125, 375, 125,

    125, 125, 250, 125, 125,
    125, 125, 250, 125, 125,
    125, 125, 250, 125, 125,
    125, 125, 375, 125,

    125, 125, 250, 125, 125,
    125, 125, 250, 125, 125,
    125, 125, 250, 125, 125,
    125, 125, 125, 250, 125,

    125, 125, 250, 125, 125,
    250, 125, 250, 125,
    125, 125, 250, 125, 125,
    125, 125, 375, 375,

    250, 125,
    //Rpeat of First Part
    125, 125, 250, 125, 125,
    125, 125, 250, 125, 125,
    125, 125, 375, 125,

    125, 125, 250, 125, 125,
    125, 125, 250, 125, 125,
    125, 125, 250, 125, 125,
    125, 125, 375, 125,

    125, 125, 250, 125, 125,
    125, 125, 250, 125, 125,
    125, 125, 250, 125, 125,
    125, 125, 125, 250, 125,

    125, 125, 250, 125, 125,
    250, 125, 250, 125,
    125, 125, 250, 125, 125,
    125, 125, 375, 375,
    //End of Repeat

    250, 125, 375, 250, 125, 375,
    125, 125, 125, 125, 125, 125, 125, 125, 375,
    250, 125, 375, 250, 125, 375,
    125, 125, 125, 125, 125, 500,

    250, 125, 375, 250, 125, 375,
    125, 125, 125, 125, 125, 125, 125, 125, 375,
    250, 125, 375, 250, 125, 375,
    125, 125, 125, 125, 125, 500};

void setup()
{
    // put your setup code here, to run once:
  //Serial.begin(9600);

  pinMode(A,OUTPUT);
  pinMode(B,OUTPUT);
  pinMode(C,OUTPUT);
  pinMode(D,OUTPUT);
  pinMode(E,OUTPUT);
  pinMode(F,OUTPUT);
  pinMode(G,OUTPUT);
  pinMode(H,OUTPUT);
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT); 
  
  
}
void loop()
{

  makethree();
  delay(1000);
  maketwo();
  delay(1000);
  makeone();
  delay(1000);
  makezero();
  delay(1000);
  

  
  const int totalNotes = sizeof(notes) / sizeof(int);
  // Loop through each note
  for (int i = 0; i < totalNotes; i++)
  {
    const int currentNote = notes[i];
    float wait = durations[i] / songSpeed;
    // Play tone if currentNote is not 0 frequency, otherwise pause (noTone)
    if (currentNote != 0)
    {
      tone(buzzer, notes[i], wait); // tone(pin, frequency, duration)
      digitalWrite(LED1, HIGH); //middle bar
      digitalWrite(LED2, HIGH); //middle bar
      digitalWrite(LED3, HIGH); //middle bar
    }
    else
    {
      noTone(buzzer);
      digitalWrite(LED1, LOW); //middle bar
      digitalWrite(LED2, LOW); //middle bar
      digitalWrite(LED3, LOW); //middle bar
    }
    // delay is used to wait for tone to finish playing before moving to next loop
    delay(wait);
}

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

void maketwo()
{
  digitalWrite(A, HIGH); //middle bar
  digitalWrite(B, LOW); //top left side bar
  digitalWrite(C, HIGH); // top bar
  digitalWrite(D, HIGH);  //top right side bar
  digitalWrite(E, LOW);  //decimal
  digitalWrite(F, LOW);  //bottom right bar
  digitalWrite(G, HIGH); // bottom bar
  digitalWrite(H, HIGH); // bottom left  bar
}

void makethree()
{
  digitalWrite(A, HIGH); //middle bar
  digitalWrite(B, LOW); //top left side bar
  digitalWrite(C, HIGH); // top bar
  digitalWrite(D, HIGH);  //top right side bar
  digitalWrite(E, LOW);  //decimal
  digitalWrite(F, HIGH);  //bottom right bar
  digitalWrite(G, HIGH); // bottom bar
  digitalWrite(H, LOW); // bottom left  bar
}


void makefour()
{
  digitalWrite(A, HIGH); //middle bar
  digitalWrite(B, HIGH); //top left side bar
  digitalWrite(C, LOW); // top bar
  digitalWrite(D, HIGH);  //top right side bar
  digitalWrite(E, LOW);  //decimal
  digitalWrite(F, HIGH);  //bottom right bar
  digitalWrite(G, LOW); // bottom bar
  digitalWrite(H, LOW); // bottom left  bar
}

void makefive()
{
  digitalWrite(A, HIGH); //middle bar
  digitalWrite(B, HIGH); //top left side bar
  digitalWrite(C, HIGH); // top bar
  digitalWrite(D, LOW);  //top right side bar
  digitalWrite(E, LOW);  //decimal
  digitalWrite(F, HIGH);  //bottom right bar
  digitalWrite(G, HIGH); // bottom bar
  digitalWrite(H, LOW); // bottom left  bar
}

void makesix()
{
  digitalWrite(A, HIGH); //middle bar
  digitalWrite(B, HIGH); //top left side bar
  digitalWrite(C, HIGH); // top bar
  digitalWrite(D, LOW);  //top right side bar
  digitalWrite(E, LOW);  //decimal
  digitalWrite(F, HIGH);  //bottom right bar
  digitalWrite(G, HIGH); // bottom bar
  digitalWrite(H, HIGH); // bottom left  bar
}


void makeseven()
{
  digitalWrite(A, LOW); //middle bar
  digitalWrite(B, LOW); //top left side bar
  digitalWrite(C, HIGH); // top bar
  digitalWrite(D, HIGH);  //top right side bar
  digitalWrite(E, LOW);  //decimal
  digitalWrite(F, HIGH);  //bottom right bar
  digitalWrite(G, LOW); // bottom bar
  digitalWrite(H, LOW); // bottom left  bar
}


void makeeight()
{
  digitalWrite(A, HIGH); //middle bar
  digitalWrite(B, HIGH); //top left side bar
  digitalWrite(C, HIGH); // top bar
  digitalWrite(D, HIGH);  //top right side bar
  digitalWrite(E, LOW);  //decimal
  digitalWrite(F, HIGH);  //bottom right bar
  digitalWrite(G, HIGH); // bottom bar
  digitalWrite(H, HIGH); // bottom left  bar
}


void makenine()
{
  digitalWrite(A, HIGH); //middle bar
  digitalWrite(B, HIGH); //top left side bar
  digitalWrite(C, HIGH); // top bar
  digitalWrite(D, HIGH);  //top right side bar
  digitalWrite(E, LOW);  //decimal
  digitalWrite(F, HIGH);  //bottom right bar
  digitalWrite(G, LOW); // bottom bar
  digitalWrite(H, LOW); // bottom left  bar
}
