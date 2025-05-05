# example_mcp_server.py
from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Any, Optional

# Create an MCP server with a clear name
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
    # ...rest of the conditions
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
def patient_vitals(patient_id: str) -> Dict[str, Any]:
    """
    Retrieve current vital signs for a patient.
    
    Args:
        patient_id: Unique identifier for the patient
        
    Returns:
        Dictionary containing vital measurements including:
        - heart_rate: Beats per minute
        - blood_pressure: Systolic/diastolic in mmHg
        - temperature: Body temperature in Celsius
        - respiratory_rate: Breaths per minute
        - oxygen_saturation: Blood oxygen percentage
    """
    return PATIENT_VITALS.get(patient_id, {"error": "Patient not found"})

@mcp.tool()
def patient_current_conditions(patient_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve current diagnosed conditions for a patient.
    
    Args:
        patient_id: Unique identifier for the patient
        
    Returns:
        List of condition dictionaries, each containing:
        - name: Name of the condition
        - icd_code: ICD-10 code
        - diagnosed_date: When the condition was diagnosed
        - severity: Severity level (mild, moderate, severe)
        - status: Current status (active, managed, resolved)
    """
    return PATIENT_CONDITIONS.get(patient_id, [])

@mcp.tool()
def patient_scheduled_checks(patient_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve upcoming scheduled health checks for a patient.
    
    Args:
        patient_id: Unique identifier for the patient
        
    Returns:
        List of scheduled check dictionaries, each containing:
        - check_type: Type of health check
        - scheduled_date: Date and time of the check
        - provider: Healthcare provider conducting the check
        - location: Where the check will take place
        - preparation: Any required preparation instructions
        - priority: Priority level (routine, follow-up, urgent)
    """
    return PATIENT_CHECKS.get(patient_id, [])

if __name__ == "__main__":
    # Use stdio transport for integration with MCP clients
    mcp.run(transport="stdio")