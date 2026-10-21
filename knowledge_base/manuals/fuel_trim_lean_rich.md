# Fuel Trim and Lean/Rich Conditions

> Illustrative procedure for the demo vehicle fleet. Not an OEM service manual.

## How fuel trim works
Short-term fuel trim (STFT) reacts within seconds to the upstream O2 sensor. Long-term fuel trim (LTFT) learns the average correction. Positive trim means the ECM is adding fuel because it sees a lean mixture; negative trim means it is removing fuel.

## Diagnosing a lean code (P0171 / P0174)
If LTFT is high at idle but returns toward zero at 2500 rpm, suspect a vacuum leak, because unmetered air matters most at low airflow. If trims stay high at all speeds, suspect fuel delivery: test fuel pressure and volume, and check the fuel filter. If both banks are lean, look for a shared cause such as the MAF sensor or fuel pressure.

## Finding vacuum leaks
Use a smoke machine on the intake with the engine off. Common leak points are the PCV hose, intake manifold gaskets, the brake booster hose and the throttle body gasket.

## MAF sensor contamination
A dirty MAF under-reports airflow, so the ECM under-fuels and trims go positive. Clean the hot-wire element with dedicated MAF cleaner only. Compare grams-per-second at idle with the specification for the engine size.

## Diagnosing a rich code (P0172)
Negative trims below -10% point to excess fuel: a leaking injector, high fuel pressure from a stuck regulator, or a sensor (MAF or coolant temperature) reporting values that make the ECM over-fuel.
