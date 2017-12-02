#include<Keypad.h>

const byte ROW=4;
const byte COL=4;
char keys[ROW][COL]={
  {'1','2','3','A'},
  {'4','5','6','B'},
  {'7','8','9','C'},
  {'*','0','#','D'}
};


byte rowpins[ROW]={6,7,8,9};
byte colpins[COL]={10,11,12,13};
Keypad k = Keypad(makeKeymap(keys),rowpins,colpins,ROW,COL);

void setup() {
  // put your setup code here, to run once:
Serial.begin(9600);
pinMode(2,OUTPUT);
digitalWrite(2,LOW);

}
char inByte;
char keyy;
void loop() {
  if(Serial.available()){
    // only send data back if data has been sent
inByte = Serial.read(); 
Serial.println(inByte);
 if(inByte == 1){
    digitalWrite(2,HIGH);
    }
    if(inByte == 2){
    digitalWrite(2,LOW);
    }// read the incoming data

  }
 keyy=k.getKey();

  if(keyy == 'A' ){
    digitalWrite(2,HIGH);
    }
  
 if(keyy == 'B' ){
    digitalWrite(2,LOW);
    }
  
  
}
