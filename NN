/* Red neuronal genérica para Arduino (probado en Arduino Mega)
 *  Primero realiza el entrenamiento de acuerdo a los parámetros fijados en el
 *  Network Configuration. Una vez entrenada, se mantiene en el loop donde calcula
 *  los valores de salida utilizando los pesos obtenidos durante el entrenamiento.
 *  El código se basa en aquel propuesto por Ralph Heymsfield en 6/2018 y del
 *  Makennbot disponible en: https://github.com/idlehandsproject/makennbot
 *  Adaptado por: Alejandro Von Chong 5/2021.
 */

#include <math.h>

/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

const int PatternCount = 10;
const int InputNodes = 2;
const int HiddenNodes = 5;
const int OutputNodes = 3;
const float LearningRate = 0.7;
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float Success = 0.0015;

float Input[PatternCount][InputNodes] = {
  { 0, 0},
  { 0, 0.1}, /* AI8, frente oscuro*/
  { 0.1, 0}, /* AI9, atrás*/
  { 0, 0.5}, /* AI8, frente*/
  { 0.5, 0}, /* AI9, atrás*/
  { 0, 0.8}, /* AI8, frente*/
  { 0.8, 0}, /* AI9, atrás iluminado*/
  { 0.1, 0.1}, /* Iluminación igual*/
  { 0.5, 0.5}, /* Iluminación igual*/
  { 0.8, 0.8}, /* Iluminación igual*/
};

const float Target[PatternCount][OutputNodes] = { // Tercer bit es dirección: 0 adelante 1 atrás
  { 0, 0, 0},
  { 0, 1, 0}, /*Comienza al 100%*/
  { 1, 0, 1},
  { 0, 0.5, 0},
  { 0.5, 0, 1},
  { 0, 0, 0},
  { 0, 0, 1}, /*Se detiene*/
  { 0, 0, 0}, /* Iluminación igual, no hace na*/
  { 0, 0, 0}, /* Iluminación igual, no hace na*/
  { 0, 0, 0}, /* Iluminación igual, no hace na*/
};

/******************************************************************
 * End Network Configuration
 ******************************************************************/


int i, j, p, q, r;
int ReportEvery1000;
int RandomizedIndex[PatternCount];
long  TrainingCycle;
float Rando;
float Error;
float Accum;


float Hidden[HiddenNodes];
float Output[OutputNodes];
float HiddenWeights[InputNodes+1][HiddenNodes];
float OutputWeights[HiddenNodes+1][OutputNodes];
float HiddenDelta[HiddenNodes];
float OutputDelta[OutputNodes];
float ChangeHiddenWeights[InputNodes+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][OutputNodes];

void setup(){
  Serial.begin(9600);
  randomSeed(analogRead(3));
  
  ReportEvery1000 = 1;
  for( p = 0 ; p < PatternCount ; p++ ) {    
    RandomizedIndex[p] = p ;
  }

/****************************************************************************************************
 * RED NEURONAL COMPLETA ****************************************************************************
 ****************************************************************************************************/
  /******************************************************************
* Initialize HiddenWeights and ChangeHiddenWeights 
******************************************************************/

  for( i = 0 ; i < HiddenNodes ; i++ ) {    
    for( j = 0 ; j <= InputNodes ; j++ ) { 
      ChangeHiddenWeights[j][i] = 0.0 ;
      Rando = float(random(100))/100;
      HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }
/******************************************************************
* Initialize OutputWeights and ChangeOutputWeights
******************************************************************/

  for( i = 0 ; i < OutputNodes ; i ++ ) {    
    for( j = 0 ; j <= HiddenNodes ; j++ ) {
      ChangeOutputWeights[j][i] = 0.0 ;  
      Rando = float(random(100))/100;        
      OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }
  Serial.println("Initial/Untrained Outputs: ");
  toTerminal();
/******************************************************************
* Begin training 
******************************************************************/

  for( TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++) {    

/******************************************************************
* Randomize order of training patterns
******************************************************************/

    for( p = 0 ; p < PatternCount ; p++) {
      q = random(PatternCount);
      r = RandomizedIndex[p] ; 
      RandomizedIndex[p] = RandomizedIndex[q] ; 
      RandomizedIndex[q] = r ;
    }
    Error = 0.0 ;
/******************************************************************
* Cycle through each training pattern in the randomized order
******************************************************************/
    for( q = 0 ; q < PatternCount ; q++ ) {    
      p = RandomizedIndex[q];

/******************************************************************
* Compute hidden layer activations
******************************************************************/

      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = HiddenWeights[InputNodes][i] ;
        for( j = 0 ; j < InputNodes ; j++ ) {
          Accum += Input[p][j] * HiddenWeights[j][i] ;
        }
        Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
      }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

      for( i = 0 ; i < OutputNodes ; i++ ) {    
        Accum = OutputWeights[HiddenNodes][i] ;
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          Accum += Hidden[j] * OutputWeights[j][i] ;
        }
        Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
        OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
        Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]) ;
      }

/******************************************************************
* Backpropagate errors to hidden layer
******************************************************************/

      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = 0.0 ;
        for( j = 0 ; j < OutputNodes ; j++ ) {
          Accum += OutputWeights[i][j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
      }


/******************************************************************
* Update Inner-->Hidden Weights
******************************************************************/


      for( i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
        for( j = 0 ; j < InputNodes ; j++ ) { 
          ChangeHiddenWeights[j][i] = LearningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
          HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
      }

/******************************************************************
* Update Hidden-->Output Weights
******************************************************************/

      for( i = 0 ; i < OutputNodes ; i ++ ) {    
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
          OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
      }
    }

/******************************************************************
* Every 1000 cycles send data to terminal for display
******************************************************************/
    ReportEvery1000 = ReportEvery1000 - 1;
    if (ReportEvery1000 == 0)
    {
      Serial.println(); 
      Serial.println(); 
      Serial.print ("TrainingCycle: ");
      Serial.print (TrainingCycle);
      Serial.print ("  Error = ");
      Serial.println (Error, 5);

      toTerminal();

      if (TrainingCycle==1)
      {
        ReportEvery1000 = 999;
      }
      else
      {
        ReportEvery1000 = 1000;
      }
    }    


/******************************************************************
* If error rate is less than pre-determined threshold then end
******************************************************************/

    if( Error < Success ) break ;  
  }
  Serial.println ();
  Serial.println(); 
  Serial.print ("TrainingCycle: ");
  Serial.print (TrainingCycle);
  Serial.print ("  Error = ");
  Serial.println (Error, 5);

  toTerminal();

  Serial.println ();  
  Serial.println ();
  Serial.println ("Training Set Solved! ");
  Serial.println ("--------"); 
  Serial.println ();
  Serial.println ();  
  ReportEvery1000 = 1;

/****************************************************************************************************
 * FIN  ENTRENAMIENTO RED NEURONAL ******************************************************************
 ****************************************************************************************************/
}  

void loop() {
  
  Serial.println("Corriendo la Red Neuronal");
  
  while (Error < Success) {
    float TestInput[] = {0, 0};

    int LL1 = analogRead(A8);  // Lee sensores
    int LL2 = analogRead(A9);
    int outdir = 0;
    
    LL1 = map(LL1, 200, 800, 0, 100); // Cambia los valores: 400 es 0 y 1024 es 100
    LL2 = map(LL2, 200, 800, 0, 100);

    LL1 = constrain(LL1, 0, 100); // Saturación de 0 a 100
    LL2 = constrain(LL2, 0, 100);

    TestInput[0] = float(LL1) / 100;
    TestInput[1] = float(LL2) / 100;

    InputToOutput(TestInput[0], TestInput[1]); //INPUT to ANN to obtain OUTPUT

    outdir = round(Output[2]); // Configuración de los pines digitales del enable pa que vaya alante o atrás
    if (outdir == 0){
      digitalWrite(23,LOW);
      digitalWrite(24,LOW);
      digitalWrite(25,LOW);
      digitalWrite(26,LOW);
      digitalWrite(27,LOW);
      digitalWrite(28,LOW);
      delay(5);}
      else{
      digitalWrite(23,LOW);
      digitalWrite(24,LOW);
      digitalWrite(25,LOW);
      digitalWrite(26,LOW);
      digitalWrite(27,LOW);
      digitalWrite(28,LOW);
      delay(5);
    }

    int speedA = Output[0] * 100;
    int speedB = Output[1] * 100;
    
    speedA = int(speedA);
    speedB = int(speedB);

    analogWrite(2, speedA);
    analogWrite(8, speedA);
    analogWrite(3, speedB);
    analogWrite(9, speedB);
    
    delay(50);
  }
}

void InputToOutput(float In1, float In2)    /* Función para calcular las SALIDAS una vez ya entrenada la NN*/
{
  float TestInput[] = {0, 0};
  TestInput[0] = In1;
  TestInput[1] = In2;

  /******************************************************************
    Compute hidden layer activations
  ******************************************************************/

  for ( i = 0 ; i < HiddenNodes ; i++ ) {
    Accum = HiddenWeights[InputNodes][i] ;
    for ( j = 0 ; j < InputNodes ; j++ ) {
      Accum += TestInput[j] * HiddenWeights[j][i] ;
    }
    Hidden[i] = 1.0 / (1.0 + exp(-Accum)) ;
  }

  /******************************************************************
    Compute output layer activations and calculate errors
  ******************************************************************/

  for ( i = 0 ; i < OutputNodes ; i++ ) {
    Accum = OutputWeights[HiddenNodes][i] ;
    for ( j = 0 ; j < HiddenNodes ; j++ ) {
      Accum += Hidden[j] * OutputWeights[j][i] ;
    }
    Output[i] = 1.0 / (1.0 + exp(-Accum)) ;
  }

  Serial.print ("  Output ");
  for ( i = 0 ; i < OutputNodes ; i++ ) {
    Serial.print (Output[i], 5);
    Serial.print (" ");
  }
  Serial.print(round(Output[2]));
  Serial.println();
}


void toTerminal()
{

  for( p = 0 ; p < PatternCount ; p++ ) { 
    Serial.println(); 
    Serial.print ("  Training Pattern: ");
    Serial.println (p);      
    Serial.print ("  Input ");
    for( i = 0 ; i < InputNodes ; i++ ) {
      Serial.print (Input[p][i], DEC);
      Serial.print (" ");
    }
    Serial.print ("  Target ");
    for( i = 0 ; i < OutputNodes ; i++ ) {
      Serial.print (Target[p][i], DEC);
      Serial.print (" ");
    }
/******************************************************************
* Compute hidden layer activations
******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[InputNodes][i] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += Input[p][j] * HiddenWeights[j][i] ;
      }
      Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
    }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[HiddenNodes][i] ;
      for( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum += Hidden[j] * OutputWeights[j][i] ;
      }
      Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
    }
    Serial.print ("  Output ");
    for( i = 0 ; i < OutputNodes ; i++ ) {       
      Serial.print (Output[i], 5);
      Serial.print (" ");
    }
  }


}
