# Haunted Door — Arduino + Pololu Dual G2 24v14 + linear actuator

A door that slowly cycles between **two positions you choose** (your own "25%"
and "75%"), pausing a random **10–30 s** at each end.

## Hardware

- Arduino (Uno / Leonardo / Mega)
- **Pololu Dual G2 High-Power Motor Driver 24v14 Shield for Arduino** (#2516)
  — *blue PCB, blue power LED*
- A 12/24 V brushed-DC **linear actuator** (2 wires)
- A matching motor power supply (6.5–40 V)

> ⚠️ This is **not** the I²C "Motoron" shield — it is **PWM-controlled** and
> needs **no library**.

## Wiring

| G2 shield | Connect to |
|---|---|
| `VIN` / `GND` (terminal block) | Motor power supply **+** / **−** |
| `M1A`, `M1B` | Actuator lead 1, actuator lead 2 |
| shield headers | Plug straight onto the Arduino |

- **Blue power LED** = motor power connected.
- If `u` moves it the wrong way, flip `OPEN_DIR` (LOW→HIGH) or swap the two
  actuator wires.

## Setting your two endpoints (once)

The sketch boots into **manual mode**. In the Serial Monitor at **115200 baud**:

1. `u` / `d` — jog to the **closed (25%)** position you want → press **`1`**.
2. `u` / `d` — jog to the **open (75%)** position you want → press **`2`**.
3. `a` — start. The door now cycles between those two exact positions, pausing
   10–30 s at each end.

> You can re-mark anytime: press `m` to pause, jog, press `1`/`2` again, then
> `a` to resume. If you mark them in the wrong order the sketch swaps them
> automatically so "close" < "open".

## How it works

- `posMs` is an **open-loop position estimate** (ms of travel). Jogging with
  `u`/`d` and the auto cycle both update it.
- Your two marks are stored as `posMs` values, and auto alternates between
  them — so the cycle is defined by *you*, not by a guessed travel distance.

Because it's open-loop (no position sensor), the estimate can drift a little
over a long night. Re-mark the two points if you ever notice the door creeping
past where you want it.

## Tuning

- `IDLE_MIN_MS` / `IDLE_MAX_MS` — pause at each end (default 10–30 s).
- `TRAVEL_SPEED` — swing speed (raise to 600 if it won't move).
- `RAMP_STEP` / `RAMP_TICK_MS` — how quickly it eases to speed (bigger
  `RAMP_STEP` = snappier; smaller = more sluggish creep).
- `JOG_STEP_MS` — default jog distance for `u`/`d` (default 200 ms).
- `OPEN_DIR` — flip if "open" (the `u` direction) is backwards.

## Commands (Serial Monitor, 115200)

| Command | Effect |
|---|---|
| `u [ms]` | jog **up**/open (free, default 200 ms) |
| `d [ms]` | jog **down**/close (free, default 200 ms) |
| `1` | mark current position as **close (25%)** |
| `2` | mark current position as **open (75%)** |
| `s <v>` | move ≤1 s at speed v (−800..800, free) |
| `r` | report position + marked endpoints |
| `x` | stop now |
| `m` | manual mode (pauses auto) |
| `a` | auto mode (cycle between the two marks) |
| `?` | help |

## Safety notes

- The G2 has **no "command timeout"** — if the sketch crashes mid-move the
  motor could keep running. Add a kill switch on the motor power line for
  unattended use.
- The G2's hardware current limiting defaults to 40 A (24v14); lower it with a
  resistor between `VREF` and `GND` (Pololu docs/0J72). The sketch prints
  `!! Motor driver FAULT` via the `M1FLT` pin on short/under-voltage/overtemp.
