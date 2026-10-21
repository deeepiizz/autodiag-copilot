# CAN Bus Network Diagnostics

> Illustrative procedure for the demo vehicle fleet. Not an OEM service manual.

## Bus basics
The high-speed CAN bus runs at 500 kbit/s on a twisted pair, CAN_H and CAN_L. Each end of the bus has a 120 ohm termination resistor. Each frame carries an 11-bit identifier and up to 8 data bytes.

## Termination check
With the battery disconnected, measure resistance between CAN_H and CAN_L at the diagnostic connector (pins 6 and 14). About 60 ohm is correct (two 120 ohm resistors in parallel). About 120 ohm means one terminator or one end of the bus is open. Near 0 ohm means CAN_H and CAN_L are shorted together.

## Voltage check
With the ignition on, CAN_H idles near 2.5 V and rises to about 3.5 V when dominant; CAN_L idles near 2.5 V and falls to about 1.5 V. A line stuck at battery voltage or ground points to a short.

## Lost communication codes
U0100, U0101 and U0121 mean another module stopped hearing the named module. First check that module's power, ground and fuses, then check its CAN wiring. If several U-codes appear together, suspect the shared bus (U0001) rather than each module.

## Using frame logs
Capture traffic and confirm each module's periodic messages are present at the expected rate. A message that disappears points to the module that sends it. Decode signal values with the vehicle's signal database to check for implausible readings.
