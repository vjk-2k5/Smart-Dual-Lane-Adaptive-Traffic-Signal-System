// Smart Dual-Lane Adaptive Traffic Signal Controller
// Listens to Serial commands from Python YOLOv8 Backend

// Pin Definitions for Lane 1
const int L1_RED = 2;
const int L1_YELLOW = 3;
const int L1_GREEN = 4;

// Pin Definitions for Lane 2
const int L2_RED = 5;
const int L2_YELLOW = 6;
const int L2_GREEN = 7;

void setup() {
  // Initialize Serial communication
  Serial.begin(9600);
  
  // Set all LED pins to OUTPUT
  pinMode(L1_RED, OUTPUT);
  pinMode(L1_YELLOW, OUTPUT);
  pinMode(L1_GREEN, OUTPUT);
  
  pinMode(L2_RED, OUTPUT);
  pinMode(L2_YELLOW, OUTPUT);
  pinMode(L2_GREEN, OUTPUT);

  // Default state: Lane 1 Green, Lane 2 Red
  setLights(LOW, LOW, HIGH, HIGH, LOW, LOW);
  Serial.println("Arduino Traffic Controller Ready.");
}

void loop() {
  // Drain the buffer so rapid Python writes (e.g. "2\n" then "3\n") all apply
  while (Serial.available() > 0) {
    char command = Serial.read();

    // Command Mapping:
    // '1': L1 Green, L2 Red
    // '2': L1 Yellow, L2 Red
    // '3': L1 Red, L2 Green
    // '4': L1 Red, L2 Yellow

    switch (command) {
      case '1':
        setLights(LOW, LOW, HIGH, HIGH, LOW, LOW);
        break;
      case '2':
        setLights(LOW, HIGH, LOW, HIGH, LOW, LOW);
        break;
      case '3':
        setLights(HIGH, LOW, LOW, LOW, LOW, HIGH);
        break;
      case '4':
        setLights(HIGH, LOW, LOW, LOW, HIGH, LOW);
        break;
      default:
        // Ignore '\r', '\n', or unknown characters
        break;
    }
  }
}

// Helper function to set all 6 lights at once
void setLights(int l1r, int l1y, int l1g, int l2r, int l2y, int l2g) {
  digitalWrite(L1_RED, l1r);
  digitalWrite(L1_YELLOW, l1y);
  digitalWrite(L1_GREEN, l1g);
  
  digitalWrite(L2_RED, l2r);
  digitalWrite(L2_YELLOW, l2y);
  digitalWrite(L2_GREEN, l2g);
}
