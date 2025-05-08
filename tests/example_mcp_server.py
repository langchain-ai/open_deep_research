from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("PatientData")

# Mock database
PATIENT_VITALS = {
    "P001": {
        "heart_rate": 72,
        "blood_pressure": "120/80",
        "temperature": 36.8,
        "respiratory_rate": 16,
        "oxygen_saturation": 98,
        "last_updated": "2025-05-03T08:30:00Z"
    },
    "P002": {
        "heart_rate": 86,
        "blood_pressure": "135/88",
        "temperature": 37.2,
        "respiratory_rate": 18,
        "oxygen_saturation": 96,
        "last_updated": "2025-05-03T07:45:00Z"
    },
    "P003": {
        "heart_rate": 65,
        "blood_pressure": "110/70",
        "temperature": 36.5,
        "respiratory_rate": 14,
        "oxygen_saturation": 99,
        "last_updated": "2025-05-03T09:10:00Z"
    }
}

PATIENT_CONDITIONS = {
    "P001": [
        {
            "name": "Type 2 Diabetes Mellitus",
            "icd_code": "E11.9",
            "diagnosed_date": "2023-07-15",
            "severity": "moderate",
            "status": "managed"
        },
        {
            "name": "Essential Hypertension",
            "icd_code": "I10",
            "diagnosed_date": "2022-03-22",
            "severity": "mild",
            "status": "active"
        }
    ],
}

PATIENT_CHECKS = {
    "P001": [
        {
            "check_type": "HbA1c Test",
            "scheduled_date": "2025-05-10T13:30:00Z",
            "provider": "Dr. Emily Johnson",
            "location": "Main Campus Lab, Room 302",
            "preparation": "Fast for 8 hours before the test",
            "priority": "routine"
        },
        # ...more checks
    ],
    # ...rest of the checks
}

@mcp.tool()
def get_patient_vitals(patient_id: str) -> str:
    """
    Use this tool to retrieve current vital signs for a patient for your research.
    
    Args:
        patient_id: Unique identifier for the patient
        
    Returns:
        String containing vital measurements information
    """
    vitals = PATIENT_VITALS.get(patient_id)
    if not vitals:
        return f"Patient {patient_id} not found."
    
    # Format as plain text
    result = f"Vitals for Patient {patient_id}:\n"
    result += f"- Heart Rate: {vitals['heart_rate']} bpm\n"
    result += f"- Blood Pressure: {vitals['blood_pressure']} mmHg\n"
    result += f"- Temperature: {vitals['temperature']}°C\n"
    result += f"- Respiratory Rate: {vitals['respiratory_rate']} breaths/min\n"
    result += f"- Oxygen Saturation: {vitals['oxygen_saturation']}%\n"
    result += f"- Last Updated: {vitals['last_updated']}"
    
    return result

@mcp.tool()
def get_patient_current_conditions(patient_id: str) -> str:
    """
    Use this tool to retrieve current diagnosed conditions for a patient so you can narrow your research.
    
    Args:
        patient_id: Unique identifier for the patient
        
    Returns:
        String containing a list of patient conditions
    """
    conditions = PATIENT_CONDITIONS.get(patient_id, [])
    if not conditions:
        return f"No conditions found for Patient {patient_id}."
    
    # Format as plain text
    result = f"Current Conditions for Patient {patient_id}:\n"
    for i, condition in enumerate(conditions, 1):
        result += f"{i}. {condition['name']} ({condition['icd_code']})\n"
        result += f"   - Diagnosed: {condition['diagnosed_date']}\n"
        result += f"   - Severity: {condition['severity']}\n"
        result += f"   - Status: {condition['status']}\n"
    
    return result.strip()

@mcp.tool()
def get_patient_scheduled_checks(patient_id: str) -> str:
    """
    Use this tool to retrieve upcoming scheduled health checks for a patient to understand their needs.
    
    Args:
        patient_id: Unique identifier for the patient
        
    Returns:
        String containing a list of scheduled health checks
    """
    checks = PATIENT_CHECKS.get(patient_id, [])
    if not checks:
        return f"No scheduled checks found for Patient {patient_id}."
    
    # Format as plain text
    result = f"Scheduled Checks for Patient {patient_id}:\n"
    for i, check in enumerate(checks, 1):
        result += f"{i}. {check['check_type']} - {check['scheduled_date']}\n"
        result += f"   - Provider: {check['provider']}\n"
        result += f"   - Location: {check['location']}\n"
        result += f"   - Preparation: {check['preparation']}\n"
        result += f"   - Priority: {check['priority']}\n"
    
    return result.strip()

if __name__ == "__main__":
    # Use stdio transport for integration with MCP clients
    mcp.run(transport="stdio")