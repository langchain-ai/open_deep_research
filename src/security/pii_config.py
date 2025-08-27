from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Entities to detect
PII_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
    "CREDIT_CARD", "IBAN_CODE", "US_SSN",
    "IP_ADDRESS", "CRYPTO", "API_KEY"
]

def sanitize_input(text):
    results = analyzer.analyze(
        text=text,
        entities=PII_ENTITIES,
        language='en'
    )
    
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )
    
    return anonymized.text
