// Smart Dual-Lane Adaptive Traffic Signal Controller
// End green: straight to all red (no amber). Start: all red -> amber only -> green (no red+amber together).

const int L1_RED = 2;
const int L1_YELLOW = 3;
const int L1_GREEN = 4;

const int L2_RED = 5;
const int L2_YELLOW = 6;
const int L2_GREEN = 7;

void lane1AmberOnly();
void lane2AmberOnly();

char currentCommand = '7';

void setup() {
  Serial.begin(9600);

  pinMode(L1_RED, OUTPUT);
  pinMode(L1_YELLOW, OUTPUT);
  pinMode(L1_GREEN, OUTPUT);
  pinMode(L2_RED, OUTPUT);
  pinMode(L2_YELLOW, OUTPUT);
  pinMode(L2_GREEN, OUTPUT);

  allRed();
  Serial.println("Arduino Traffic Controller Ready");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r' || c == '\n' || c == ' ') {
      continue;
    }
    if (c >= '0' && c <= '7') {
      currentCommand = c;
    }
  }

  switch (currentCommand) {
    case '0':
    case '7':
      allRed();
      break;

    case '1':
      lane1Green();
      break;

    case '2':
      lane1Yellow();
      break;

    case '3':
      lane2Green();
      break;

    case '4':
      lane2Yellow();
      break;

    case '5':
      lane1AmberOnly();
      break;

    case '6':
      lane2AmberOnly();
      break;
  }
}

void lane1Green() {
  digitalWrite(L1_GREEN, HIGH);
  digitalWrite(L1_YELLOW, LOW);
  digitalWrite(L1_RED, LOW);

  digitalWrite(L2_GREEN, LOW);
  digitalWrite(L2_YELLOW, LOW);
  digitalWrite(L2_RED, HIGH);
}

void lane1Yellow() {
  lane1AmberOnly();
}

void lane2Green() {
  digitalWrite(L1_GREEN, LOW);
  digitalWrite(L1_YELLOW, LOW);
  digitalWrite(L1_RED, HIGH);

  digitalWrite(L2_GREEN, HIGH);
  digitalWrite(L2_YELLOW, LOW);
  digitalWrite(L2_RED, LOW);
}

void lane2Yellow() {
  lane2AmberOnly();
}

void allRed() {
  digitalWrite(L1_GREEN, LOW);
  digitalWrite(L1_YELLOW, LOW);
  digitalWrite(L1_RED, HIGH);

  digitalWrite(L2_GREEN, LOW);
  digitalWrite(L2_YELLOW, LOW);
  digitalWrite(L2_RED, HIGH);
}

void lane1AmberOnly() {
  digitalWrite(L1_GREEN, LOW);
  digitalWrite(L1_YELLOW, HIGH);
  digitalWrite(L1_RED, LOW);

  digitalWrite(L2_GREEN, LOW);
  digitalWrite(L2_YELLOW, LOW);
  digitalWrite(L2_RED, HIGH);
}

void lane2AmberOnly() {
  digitalWrite(L1_GREEN, LOW);
  digitalWrite(L1_YELLOW, LOW);
  digitalWrite(L1_RED, HIGH);

  digitalWrite(L2_GREEN, LOW);
  digitalWrite(L2_YELLOW, HIGH);
  digitalWrite(L2_RED, LOW);
}