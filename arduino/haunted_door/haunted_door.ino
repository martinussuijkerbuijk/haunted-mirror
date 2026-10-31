/*
 * Haunted Door — slow open/close cycle with a linear actuator
 * ==================================================================
 * Drives a brushed-DC linear actuator through a Pololu
 * Dual G2 High-Power Motor Driver 24v14 Shield for Arduino (#2516).
 * PWM control (analogWrite) + DIR pin; no library required.
 *
 * BEHAVIOUR:
 *   * Manual mode: jog up/down with 'u'/'d', then mark the two endpoints of
 *     your cycle yourself:
 *        1) jog to the CLOSED (25%) position  -> press '1'
 *        2) jog to the OPEN   (75%) position  -> press '2'
 *   * Auto mode ('a'): slowly cycles between your two marked positions,
 *     pausing a random 10–30 s at each end.
 *   Because YOU choose the endpoints, the door never reaches the physical
 *   end stops.
 *
 * WIRING (Dual G2 24v14, on top of an Arduino Uno/Leonardo/Mega):
 *   - Plug the shield onto the Arduino (stackable headers).
 *   - Motor power:  + -> VIN,  - -> GND   (6.5–40 V, 14 A max per channel)
 *   - Actuator: one lead -> M1A, other lead -> M1B
 *   - Blue power LED on the shield = motor power connected.
 *
 * TUNING: see the CONFIGURATION block.
 */

// ============================ PINS =======================================
const uint8_t M1PWM = 9;    // speed (analogWrite 0..255)
const uint8_t M1DIR = 7;    // direction
const uint8_t M1SLP = 2;    // enable (HIGH = on, LOW = sleep/coast)
const uint8_t M1FLT = 6;    // fault flag (open-drain, active LOW)

// Which DIR level is OPEN (the 'u' jog direction). Flip to HIGH if 'u'
// moves it the wrong way.
const uint8_t OPEN_DIR  = LOW;
const uint8_t CLOSE_DIR = (OPEN_DIR == LOW) ? HIGH : LOW;
// =========================================================================

// ============================ CONFIGURATION ==============================
const int16_t TRAVEL_SPEED  = 400;    // swing speed (0..800 = 0..100%)

// Software ramp (the G2 has no built-in acceleration): the door eases toward
// its target speed so it creeps instead of snapping.
const uint8_t  RAMP_TICK_MS = 10;     // how often speed changes by one notch
const uint16_t RAMP_STEP    = 16;     // speed notch per tick (0..800 scale)

const uint32_t FULL_TRAVEL_MS = 4000; // sanity cap on manual move length

const uint32_t JOG_STEP_MS = 200;     // default manual jog step ('u'/'d')

const uint32_t IDLE_MIN_MS = 5000;   // 10 s pause at each end of the cycle
const uint32_t IDLE_MAX_MS = 15000;   // 30 s
// =========================================================================

// ---- state ----
enum State { STATE_IDLE, STATE_MOVING };
State state = STATE_IDLE;
uint32_t idleStartMs = 0;
uint32_t idleDurationMs = 0;

int16_t currentSpeed = 0;   // speed actually applied (ramps toward target)
int16_t targetSpeed  = 0;   // speed we are easing toward
uint32_t lastRampTick = 0;

// Estimated position (ms of travel at TRAVEL_SPEED). + = open, - = close.
int32_t posMs = 0;

// The two cycle endpoints, marked by the user. 25% = more closed, 75% = more
// open. (If marked in the opposite order they're swapped automatically.)
int32_t posClose = 0;       // 25% endpoint
int32_t posOpen  = 0;       // 75% endpoint
bool closeSet = false;
bool openSet  = false;

// Which end we head to on the NEXT move. Starts false so the first auto move
// goes to the close (25%) side, giving the cycle 25% -> 75% -> 25% -> ...
bool nextToOpen = false;

// The current move: target speed and how long to hold it.
int16_t  moveSpeed    = 0;
uint32_t moveDuration = 0;
uint32_t moveStartMs  = 0;

bool autoMode = false;   // start in manual mode until the endpoints are set
uint32_t lastFaultCheck = 0;
bool faultReported = false;

char serialBuf[32];
uint8_t serialLen = 0;

// -------------------------------------------------------------------------
// Map -800..800 to PWM duty and drive DIR + PWM pins.
void writeSpeed(int16_t speed) {
  if (speed > 800)  speed = 800;
  if (speed < -800) speed = -800;

  if (speed > 0) {
    digitalWrite(M1DIR, OPEN_DIR);
    analogWrite(M1PWM, (int)((long)speed * 255 / 800));
  } else if (speed < 0) {
    digitalWrite(M1DIR, CLOSE_DIR);
    analogWrite(M1PWM, (int)((long)(-speed) * 255 / 800));
  } else {
    analogWrite(M1PWM, 0);   // brake: both outputs held low
  }
}

void setSpeedInstant(int16_t s) {
  currentSpeed = s;
  targetSpeed  = s;
  writeSpeed(s);
}

void setSpeedSmooth(int16_t s) {
  targetSpeed = s;
}

void stopMotor() {
  setSpeedSmooth(0);
}

void updateSpeedRamp(uint32_t now) {
  if (now - lastRampTick < RAMP_TICK_MS) return;
  lastRampTick = now;
  if (currentSpeed == targetSpeed) return;

  if (currentSpeed < targetSpeed) {
    currentSpeed += RAMP_STEP;
    if (currentSpeed > targetSpeed) currentSpeed = targetSpeed;
  } else {
    currentSpeed -= RAMP_STEP;
    if (currentSpeed < targetSpeed) currentSpeed = targetSpeed;
  }
  writeSpeed(currentSpeed);
}

void checkFault(uint32_t now) {
  if (now - lastFaultCheck < 200) return;
  lastFaultCheck = now;
  bool fault = (digitalRead(M1FLT) == LOW);
  if (fault && !faultReported) {
    Serial.println(F("!! Motor driver FAULT (short/undervoltage/overtemp)"));
    faultReported = true;
  } else if (!fault) {
    faultReported = false;
  }
}

// Start one leg of the cycle: move to the next endpoint (alternating).
void startEpisode() {
  int32_t target = nextToOpen ? posOpen : posClose;
  nextToOpen = !nextToOpen;

  int32_t delta = target - posMs;
  moveSpeed    = (delta >= 0) ? TRAVEL_SPEED : -TRAVEL_SPEED;
  moveDuration = (uint32_t)(delta >= 0 ? delta : -delta);
  moveStartMs  = millis();

  setSpeedSmooth(moveSpeed);
  posMs = target;
  state = STATE_MOVING;

  Serial.print(F("[auto] -> "));
  Serial.print(target == posClose ? F("CLOSE (25%)") : F("OPEN (75%)"));
  Serial.print(F("  pos="));
  Serial.println(posMs);
}

void runEpisode(uint32_t now) {
  if (now - moveStartMs >= moveDuration) {
    stopMotor();
    state = STATE_IDLE;
    idleStartMs = now;
    idleDurationMs = random(IDLE_MIN_MS, IDLE_MAX_MS + 1);
    Serial.print(F("[auto] reached end. waiting "));
    Serial.print(idleDurationMs / 1000);
    Serial.println(F(" s"));
  }
}

// Blocking move for the manual (jog) mode. Free — you're watching. Updates
// the position estimate. Duration capped for sanity.
void manualMove(int16_t speed, uint32_t durationMs) {
  if (durationMs > FULL_TRAVEL_MS) durationMs = FULL_TRAVEL_MS;
  setSpeedInstant(speed);
  uint32_t start = millis();
  while (millis() - start < durationMs) {
    if (Serial.available() && Serial.read() == 'x') break;  // abort with 'x'
    delay(10);
  }
  uint32_t elapsed = millis() - start;
  setSpeedInstant(0);
  posMs += (int32_t)((long)elapsed * speed / TRAVEL_SPEED);
}

uint32_t argMs(const char *line, uint32_t defaultMs) {
  long v = atol(line + 1);
  return (v > 0) ? (uint32_t)v : defaultMs;
}

void printHelp() {
  Serial.println(F("Commands (send with a newline):"));
  Serial.println(F("  u [ms]   jog UP/open  (free, default 200 ms)"));
  Serial.println(F("  d [ms]   jog DOWN/close (free, default 200 ms)"));
  Serial.println(F("  1        mark this position as CLOSE (25%)"));
  Serial.println(F("  2        mark this position as OPEN  (75%)"));
  Serial.println(F("  s <v>    move <=1 s at speed v, -800..800 (free)"));
  Serial.println(F("  r        report position + marked endpoints"));
  Serial.println(F("  x        stop now"));
  Serial.println(F("  m        manual mode (auto paused)"));
  Serial.println(F("  a        auto mode (cycle between the two marks)"));
  Serial.println(F("  ?        this help"));
}

void parseCommand(const char *line) {
  char cmd = line[0];
  switch (cmd) {
    case 'a':
      if (!closeSet || !openSet) {
        Serial.println(F("Set both endpoints first: jog to 25% -> '1', jog to 75% -> '2'."));
        break;
      }
      if (posClose > posOpen) {   // marked in reverse order — swap so close < open
        int32_t t = posClose; posClose = posOpen; posOpen = t;
        Serial.println(F("note: endpoints swapped (close < open)"));
      }
      if (posClose == posOpen) {
        Serial.println(F("Endpoints are identical — re-mark them."));
        break;
      }
      autoMode = true; stopMotor();
      nextToOpen = false;         // first move goes to CLOSE (25%)
      state = STATE_IDLE;
      idleStartMs = millis();
      idleDurationMs = 2000;
      Serial.println(F("auto mode: cycling between your two marks"));
      break;

    case 'm': autoMode = false; stopMotor();
              Serial.println(F("manual mode"));
              break;

    case 'u': { uint32_t d = argMs(line, JOG_STEP_MS);
                manualMove( TRAVEL_SPEED, d); }
              break;
    case 'd': { uint32_t d = argMs(line, JOG_STEP_MS);
                manualMove(-TRAVEL_SPEED, d); }
              break;

    case '1': posClose = posMs; closeSet = true;
              Serial.print(F("CLOSE (25%) set at pos="));
              Serial.println(posClose);
              break;
    case '2': posOpen = posMs; openSet = true;
              Serial.print(F("OPEN (75%) set at pos="));
              Serial.println(posOpen);
              break;

    case 's': { long v = atol(line + 1);
                if (v < -800) v = -800;
                if (v >  800) v =  800;
                manualMove((int16_t)v, 1000); }
              break;

    case 'r': Serial.print(F("pos="));
              Serial.print(posMs);
              if (closeSet) { Serial.print(F("  close(25%)=")); Serial.print(posClose); }
              if (openSet)  { Serial.print(F("  open(75%)="));  Serial.print(posOpen); }
              Serial.println();
              break;

    case 'x': stopMotor(); break;
    case '?': case 'h': printHelp(); break;
    default:  Serial.println(F("? (type ? for help)"));
  }
}

void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      serialBuf[serialLen] = 0;
      if (serialLen > 0) parseCommand(serialBuf);
      serialLen = 0;
    } else if (serialLen < sizeof(serialBuf) - 1) {
      serialBuf[serialLen++] = c;
    }
  }
}

// -------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) { /* wait for native-USB boards (Leonardo/Micro) */ }
  delay(100);

  pinMode(M1PWM, OUTPUT);
  pinMode(M1DIR, OUTPUT);
  pinMode(M1SLP, OUTPUT);
  pinMode(M1FLT, INPUT_PULLUP);

  digitalWrite(M1SLP, HIGH);   // enable motor 1 (LOW = sleep/coast)
  digitalWrite(M1DIR, LOW);
  analogWrite(M1PWM, 0);

  randomSeed(analogRead(A2) + (uint32_t)micros());

  Serial.println(F("Haunted door ready (Dual G2 24v14). type ? for help."));
  Serial.println(F("1) jog with u/d to the CLOSE (25%) position -> press 1"));
  Serial.println(F("2) jog with u/d to the OPEN  (75%) position -> press 2"));
  Serial.println(F("3) press a to start the cycle."));

  state = STATE_IDLE;
  idleStartMs = millis();
  idleDurationMs = 2000;
}

void loop() {
  uint32_t now = millis();

  updateSpeedRamp(now);
  handleSerial();
  checkFault(now);

  if (autoMode) {
    switch (state) {
      case STATE_IDLE:
        if (now - idleStartMs >= idleDurationMs) startEpisode();
        break;
      case STATE_MOVING:
        runEpisode(now);
        break;
    }
  }
}
