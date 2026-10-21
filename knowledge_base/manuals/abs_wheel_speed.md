# ABS and Wheel Speed Sensors

> Illustrative procedure for the demo vehicle fleet. Not an OEM service manual.

## Safety note
An ABS fault disables anti-lock braking and usually stability control. Normal braking remains, but the vehicle should be repaired before regular use.

## Wheel speed sensor codes
C0035 (left front) and C0040 (right front) set when a sensor signal is missing or implausible. Compare all four wheel speeds on a scan tool or in decoded CAN data while driving slowly in a straight line: a faulty sensor reads zero or drops out while the others agree.

## Sensor and tone ring inspection
Check the wiring near the wheel, where it flexes with suspension travel. Inspect the tone ring for missing teeth, cracks or metallic debris. Excessive wheel bearing play changes the air gap and can cause intermittent signals.

## Active vs passive sensors
Passive sensors produce an AC voltage that increases with speed and can be checked with resistance measurements. Active (Hall-effect) sensors need a supply voltage from the ABS module and are tested with a scope or by watching live data.
