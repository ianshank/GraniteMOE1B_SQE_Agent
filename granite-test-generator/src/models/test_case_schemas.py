from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class TestCasePriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"

class TestCaseType(str, Enum):
    FUNCTIONAL = "functional"
    INTEGRATION = "integration"
    UNIT = "unit"
    REGRESSION = "regression"
    PERFORMANCE = "performance"

class TestStep(BaseModel):
    step_number: int = Field(..., description="Sequential step number")
    action: str = Field(..., description="Action to be performed")
    expected_result: str = Field(..., description="Expected outcome")
    data_required: Optional[Dict[str, Any]] = Field(None, description="Test data needed")

class TestCaseProvenance(BaseModel):
    """Describes the external source of a test case for traceability."""
    system: str = Field(..., description="Origin system identifier, e.g., 'jira', 'github', 'file'")
    source_id: str = Field(..., description="Identifier from the origin system (e.g., issue key/number)")
    url: Optional[str] = Field(None, description="Direct URL to the source item, if available")
    summary: Optional[str] = Field(None, description="Short source summary/title for auditing")
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional structured metadata")

class TestCase(BaseModel):
    id: str = Field(..., description="Unique test case identifier")
    summary: str = Field(..., description="Brief test case description")
    priority: TestCasePriority
    test_type: TestCaseType
    preconditions: List[str] = Field(default_factory=list)
    input_data: Dict[str, Any] = Field(default_factory=dict)
    steps: List[TestStep]
    expected_results: str = Field(..., description="Overall expected outcome")
    requirements_traced: List[str] = Field(default_factory=list)
    team_context: Optional[str] = Field(None, description="Originating team context")
    provenance: Optional[TestCaseProvenance] = Field(
        None, description="Verifiable source details for the derived test case"
    )
    
class TestSuite(BaseModel):
    name: str
    description: str
    test_cases: List[TestCase]
    coverage_metrics: Optional[Dict[str, float]] = None
