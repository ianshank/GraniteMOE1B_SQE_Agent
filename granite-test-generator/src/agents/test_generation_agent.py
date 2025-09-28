try:
    from langchain.agents import AgentType, initialize_agent  # type: ignore
    from langchain.tools import Tool  # type: ignore
    from langchain.schema import AgentAction, AgentFinish  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback for CI/test envs
    AgentType = object  # type: ignore
    AgentAction = object  # type: ignore
    AgentFinish = object  # type: ignore

    class Tool:  # type: ignore
        def __init__(self, name: str, description: str, func):
            self.name = name
            self.description = description
            self.func = func

    def initialize_agent(*args, **kwargs):  # type: ignore
        return None
from typing import List, Dict, Any, Optional
import asyncio
from typing import TYPE_CHECKING
from src.utils.constants import TEMPLATE_PATTERNS
from src.models.test_case_schemas import (
    TestCase,
    TestStep,
    TestCasePriority,
    TestCaseType,
)
# Optional MLX language model
try:
    from mlx_lm import generate  # type: ignore
    _MLX_AVAILABLE = True
except Exception:  # pragma: no cover
    _MLX_AVAILABLE = False
    def generate(*args, **kwargs):  # type: ignore
        return "[TEST_CASE][SUMMARY]fallback[/SUMMARY][INPUT_DATA]{}[/INPUT_DATA][STEPS]1. step -> ok[/STEPS][EXPECTED]ok[/EXPECTED][/TEST_CASE]"

if TYPE_CHECKING:
    from src.models.granite_moe import GraniteMoETrainer
    from src.data.rag_retriever import RAGRetriever
    from src.data.cag_cache import CAGCache
    from src.models.test_case_schemas import TestCase, TestStep, TestCasePriority, TestCaseType

class TestGenerationAgent:
    """MoE-powered agent for generating test cases from requirements"""
    
    def __init__(self, granite_trainer: 'GraniteMoETrainer', 
                 rag_retriever: 'RAGRetriever', 
                 cag_cache: 'CAGCache'):
        self.granite_model = granite_trainer
        self.rag_retriever = rag_retriever
        self.cag_cache = cag_cache
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> List[Tool]:
        """Initialize tools for the agent"""
        tools = [
            Tool(
                name="retrieve_requirements",
                description="Retrieve relevant requirements and user stories",
                func=self._retrieve_requirements
            ),
            Tool(
                name="check_cache",
                description="Check cache for similar test cases",
                func=self._check_cache
            ),
            Tool(
                name="generate_test_case",
                description="Generate new test case using Granite MoE model",
                func=self._generate_test_case
            ),
            Tool(
                name="validate_test_case",
                description="Validate generated test case structure and completeness",
                func=self._validate_test_case
            )
        ]
        return tools
    
    def _retrieve_requirements(self, query: str) -> str:
        """Tool function to retrieve requirements"""
        docs = self.rag_retriever.retrieve_relevant_context(query, k=3)
        return "\n".join([doc['content'] for doc in docs])
    
    def _check_cache(self, query: str) -> str:
        """Tool function to check cached responses"""
        cached = self.cag_cache.get_cached_response(query, "default")
        if cached:
            return f"Found cached response: {cached.get('response', '')}"
        return "No cached response found"
    
    def _generate_test_case(self, requirement: str) -> str:
        """Generate test case using MLX model or fallback templating."""
        if not getattr(self.granite_model, 'mlx_model', None) or not getattr(self.granite_model, 'mlx_tokenizer', None):
            self.granite_model.load_model_for_inference()
            # If still not available, fallback
            if not getattr(self.granite_model, 'mlx_model', None) or not getattr(self.granite_model, 'mlx_tokenizer', None):
                return self._template_generate(requirement)

        prompt = f"""<|system|>
You are a software quality engineer. Create a detailed test case for the given requirement.

<|user|>
Requirement: {requirement}
Create a comprehensive test case with summary, input data, steps, and expected results.

<|assistant|>"""

        try:
            response = generate(
                self.granite_model.mlx_model,
                self.granite_model.mlx_tokenizer,
                prompt=prompt,
                max_tokens=384,
                temperature=0.6
            )
            return response
        except TypeError:
            # Older mlx_lm signature fallback
            response = generate(
                self.granite_model.mlx_model,
                self.granite_model.mlx_tokenizer,
                prompt=prompt,
                max_tokens=384
            )
            return response
        except Exception:
            # Final fallback
            return self._template_generate(requirement)

    def _template_generate(self, requirement: str) -> str:
        """Lightweight, deterministic template-based generation when model is unavailable."""
        import textwrap, re
        # Heuristic summary extraction
        summary_line = next((ln.strip() for ln in requirement.split("\n") if ln.strip()), requirement)[:120]

        # Derive domain-specific steps
        req_lower = requirement.lower()
        if any(k in req_lower for k in ["login", "authenticate", "sign in"]):
            raw_steps = TEMPLATE_PATTERNS["login"]
        elif any(k in req_lower for k in ["upload", "ingest"]):
            raw_steps = TEMPLATE_PATTERNS["upload"]
        else:
            raw_steps = TEMPLATE_PATTERNS["default"]

        steps_block = "\n".join([f"{i+1}. {s}" for i, s in enumerate(raw_steps)])

        return textwrap.dedent(f"""
            [TEST_CASE]
            [SUMMARY]{summary_line}[/SUMMARY]

            [INPUT_DATA]
            {{}}
            [/INPUT_DATA]

            [STEPS]
            {steps_block}
            [/STEPS]

            [EXPECTED]System should satisfy the requirement without errors[/EXPECTED]
            [/TEST_CASE]
        """).strip()
        
    def _validate_test_case(self, test_case_text: str) -> str:
        """Tool function to validate test case structure"""
        required_sections = ['summary', 'steps', 'expected']
        missing_sections = []
        
        for section in required_sections:
            if section.lower() not in test_case_text.lower():
                missing_sections.append(section)
        
        if missing_sections:
            return f"Validation failed. Missing sections: {', '.join(missing_sections)}"
        else:
            return "Validation passed. Test case structure is complete."
    
    async def generate_test_cases_for_team(self, team_name: str, 
                                         requirements: List[str]) -> List['TestCase']:
        """Generate test cases for a specific team"""
        test_cases = []
        
        for requirement in requirements:
            # Agent workflow
            steps = [
                f"retrieve_requirements: {requirement}",
                f"check_cache: {requirement}",
                f"generate_test_case: {requirement}",
                "validate_test_case: generated_test_case"
            ]
            
            context = ""
            generated_text = ""
            
            for step in steps:
                action, query = step.split(": ", 1)
                
                if action == "retrieve_requirements":
                    context = self._retrieve_requirements(query)
                elif action == "check_cache":
                    cache_result = self._check_cache(query)
                    if "Found cached response" in cache_result:
                        generated_text = cache_result.split(": ", 1)[1]
                        continue  # Skip generation if cached
                elif action == "generate_test_case":
                    full_requirement = f"{query}\n\nContext: {context}"
                    generated_text = self._generate_test_case(full_requirement)
                elif action == "validate_test_case":
                    validation_result = self._validate_test_case(generated_text)
                    if "failed" in validation_result:
                        # Regenerate if validation fails
                        generated_text = self._generate_test_case(f"{requirement}\n\nContext: {context}")
            
            # Parse generated text into TestCase object
            try:
                test_case = self._parse_generated_test_case(generated_text, team_name)
                test_cases.append(test_case)
                
                # Cache the successful generation
                self.cag_cache.cache_response(
                    query=requirement,
                    response=generated_text,
                    context={'team': team_name},
                    team_context=team_name,
                    tags=['test_case', 'generated']
                )
                
            except Exception as e:
                print(f"Failed to parse test case for requirement: {requirement}. Error: {e}")
                continue
        
        return test_cases
    
    def _parse_generated_test_case(self, generated_text: str, team_name: str) -> 'TestCase':
        """Parse generated text into structured TestCase object"""
        import re, hashlib, json
        # 1. Extract tagged sections robustly (case-insensitive, multiline)
        def _extract(tag: str) -> Optional[str]:
            match = re.search(fr"\[{tag}\](.*?)\[/{tag}\]", generated_text, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else None

        summary = _extract("SUMMARY") or "Auto-generated test case"
        expected = _extract("EXPECTED") or "System should fulfill the requirement"
        steps_raw = _extract("STEPS") or ""
        input_raw = _extract("INPUT_DATA") or "{}"

        # 2. Parse steps, tolerate missing arrows
        steps: List[TestStep] = []
        for idx, line in enumerate([ln.strip() for ln in steps_raw.splitlines() if ln.strip()], start=1):
            if "->" in line:
                action, exp = [p.strip() for p in line.split("->", 1)]
            else:
                action, exp = line, "Step should succeed"
            steps.append(TestStep(step_number=idx, action=action, expected_result=exp))

        if not steps:
            # Guarantee at least one step
            steps.append(TestStep(step_number=1, action="Execute scenario", expected_result="Requirement met"))

        # 3. Parse input data JSON if present
        try:
            input_data = json.loads(input_raw) if input_raw else {}
        except json.JSONDecodeError:
            input_data = {"raw": input_raw}

        # 4. Generate deterministic yet unique ID
        hash_src = f"{summary}{expected}{len(steps)}{team_name}"
        uid = hashlib.md5(hash_src.encode()).hexdigest()[:8]

        return TestCase(
            id=f"{team_name}_{uid}",
            summary=summary,
            priority=TestCasePriority.MEDIUM,
            test_type=TestCaseType.FUNCTIONAL,
            steps=steps,
            expected_results=expected,
            input_data=input_data,
            team_context=team_name
        )
