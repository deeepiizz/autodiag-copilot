"""Safety guardrails: flag safety-critical systems and refuse tampering requests."""
import re

# Requests to defeat safety or emissions equipment are refused outright.
TAMPERING_PATTERNS = [
    r"\b(disable|bypass|defeat|delete|remove|turn off)\b.*\b(airbag|srs|abs|brake|seat ?belt|emission|egr|dpf|catalyst|cat|o2 sensor|immobili[sz]er)\b",
    r"\b(egr|dpf|cat|catalyst)\s*(delete|removal)\b",
]

SAFETY_RULES = [
    (r"\b(airbag|srs|b00\d\d)\b",
     "Airbag (SRS) systems can deploy unexpectedly. Disconnect the battery and wait the specified time before touching SRS parts; repairs should be done by a qualified technician."),
    (r"\b(high[- ]voltage|hv|hybrid battery|ev battery|orange cable|p0a[0-9a-f]{2})\b",
     "High-voltage systems can be lethal. Do not open the battery pack or touch orange cables without HV training, insulated PPE and the de-energising procedure."),
    (r"\b(brake|abs|c00\d\d)\b",
     "Braking-system faults affect stopping performance. Get the vehicle inspected before regular driving."),
    (r"\b(fuel leak|smell of fuel|petrol leak)\b",
     "A fuel leak is a fire risk. Do not start the engine; move the vehicle away from ignition sources."),
    (r"\bflashing (check[- ]engine|mil)\b",
     "A flashing check-engine lamp means a severe misfire that can damage the catalytic converter. Avoid driving under load."),
]


def is_tampering(text):
    t = text.lower()
    return any(re.search(p, t) for p in TAMPERING_PATTERNS)


def safety_warnings(text, severities=()):
    """Return warnings triggered by keywords in `text` or by critical DTC severities."""
    t = text.lower()
    warnings = [msg for pattern, msg in SAFETY_RULES if re.search(pattern, t)]
    if "critical" in severities:
        warnings.append("At least one fault is rated CRITICAL. Stop using the vehicle until it has been inspected.")
    return list(dict.fromkeys(warnings))


TAMPERING_REFUSAL = (
    "I can't help disable or remove safety or emissions equipment. It is illegal on road vehicles "
    "in most countries and puts people at risk. I can help diagnose why the system is reporting a fault instead."
)
