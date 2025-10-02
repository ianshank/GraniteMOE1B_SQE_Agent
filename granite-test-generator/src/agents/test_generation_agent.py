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
from src.integration.openai_client import OpenAIClient, OpenAIIntegrationError
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
        import logging, os
        self.granite_model = granite_trainer
        self.rag_retriever = rag_retriever
        self.cag_cache = cag_cache
        self.tools = self._initialize_tools()
        # Per-instance logger
        self._logger = logging.getLogger(__name__)
        # Disallow template/dummy generation unless explicitly enabled
        self._allow_template_generation = str(os.getenv("ALLOW_TEMPLATE_GENERATION", "false")).lower() in ("1", "true", "yes")

        self._openai_client: Optional[OpenAIClient] = None
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                self._openai_client = OpenAIClient(api_key=openai_key)
                self._logger.info(
                    "OpenAI integration enabled for remote test case generation using model %s",
                    self._openai_client.default_model,
                )
            except OpenAIIntegrationError as exc:
                self._logger.warning("OpenAI integration disabled: %s", exc)
        
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
        """Generate test case using MLX model, transformers model, or fallback templating."""
        prompt = self._build_generation_prompt(requirement)

        if self._openai_client:
            try:
                return self._generate_with_openai(prompt)
            except OpenAIIntegrationError as exc:
                self._logger.warning("OpenAI generation failed, falling back to local models: %s", exc)

        # Try MLX first (optimal for Apple Silicon)
        if not getattr(self.granite_model, 'mlx_model', None) or not getattr(self.granite_model, 'mlx_tokenizer', None):
            self.granite_model.load_model_for_inference()

        # If MLX not available, try transformers model
        if not getattr(self.granite_model, 'mlx_model', None):
            return self._generate_with_transformers(requirement, prompt)

        # If no models available, only allow template generation when explicitly enabled
        if not getattr(self.granite_model, 'mlx_model', None) and not getattr(self.granite_model, 'model', None):
            if not getattr(self, '_allow_template_generation', False):
                raise RuntimeError(
                    "No generation backend available (MLX/Transformers) and template generation is disabled."
                )
            return self._template_generate(requirement)

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

    def _build_generation_prompt(self, requirement: str) -> str:
        """Construct a detailed prompt for any generation backend."""

        return f"""<|system|>
You are an expert software quality engineer. Create a comprehensive, detailed test case that thoroughly validates the system against the given requirement. Include specific validation steps, edge cases, error conditions, and acceptance criteria verification.

<|user|>
Requirement: {requirement}

Create a detailed test case that includes:
1. A clear, specific summary describing what is being tested
2. Comprehensive test steps with detailed actions and specific validation points
3. Expected results that validate all acceptance criteria
4. Input data requirements and test data specifications
5. Verification of security, performance, and functional requirements
6. Edge cases and error condition testing
7. Specific assertions and validation checks

Format your response with [TEST_CASE][SUMMARY]...[/SUMMARY][INPUT_DATA]...[/INPUT_DATA][STEPS]...[/STEPS][EXPECTED]...[/EXPECTED][/TEST_CASE]

<|assistant|>"""

    def _generate_with_openai(self, prompt: str) -> str:
        """Generate a test case via the OpenAI integration."""

        if not self._openai_client:
            raise OpenAIIntegrationError("OpenAI client not initialized")

        return self._openai_client.generate_response(
            prompt,
            temperature=0.55,
            max_output_tokens=900,
        )

    def _generate_with_transformers(self, requirement: str, prompt: Optional[str] = None) -> str:
        """Generate test case using transformers library when MLX is not available."""

        try:
            # Load transformers model if not already loaded
            if not getattr(self.granite_model, 'model', None):
                self._logger.info("Loading transformers model for test case generation")
                self.granite_model.load_model_for_training()

            if not getattr(self.granite_model, 'model', None):
                if not getattr(self, '_allow_template_generation', False):
                    raise RuntimeError("Transformers backend missing and template generation disabled")
                self._logger.warning("Transformers model not available, falling back to template generation")
                return self._template_generate(requirement)

            # Create enhanced prompt for transformers
            prompt = prompt or self._build_generation_prompt(requirement)

            # Tokenize and generate
            inputs = self.granite_model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Lazy import torch; if unavailable and templates disabled, error out
            try:
                import torch  # type: ignore
            except (ImportError, ModuleNotFoundError):
                if not getattr(self, '_allow_template_generation', False):
                    raise
                self._logger.warning("torch not available, falling back to template generation")
                return self._template_generate(requirement)

            with torch.no_grad():
                outputs = self.granite_model.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.granite_model.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.granite_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|assistant|>" in generated_text:
                response = generated_text.split("<|assistant|>")[-1].strip()
            else:
                response = generated_text.strip()
            
            self._logger.debug(f"Generated test case using transformers model: {len(response)} characters")
            return response
            
        except Exception as e:
            if not getattr(self, '_allow_template_generation', False):
                raise
            self._logger.error(f"Error generating with transformers model: {e}. Falling back to template generation.")
            return self._template_generate(requirement)

    def _template_generate(self, requirement: str) -> str:
        """Extract detailed test steps from the actual requirement content."""
        import textwrap, re
        
        # Extract meaningful summary from requirement
        lines = [line.strip() for line in requirement.split('\n') if line.strip()]
        summary = "Test case for requirement"
        
        # Look for title/header
        for line in lines:
            if line.startswith('#') or line.startswith('**') or 'title' in line.lower():
                summary = re.sub(r'^#+\s*|\*\*|\*', '', line).strip()
                break
        
        if len(summary) < 10:  # If no good summary found, use first substantial line
            for line in lines:
                if len(line) > 20 and not line.startswith('#'):
                    summary = line[:100]
                    break
        
        # Extract acceptance criteria and requirements from the text
        steps = []
        step_num = 1
        
        # Look for acceptance criteria, requirements, or test scenarios in the text
        acceptance_section = False
        requirement_section = False
        
        for line in lines:
            line_lower = line.lower()
            
            # Identify sections
            if any(keyword in line_lower for keyword in ['acceptance criteria', 'requirements', 'test', 'validate', 'verify']):
                acceptance_section = True
                continue
            elif line.startswith('#') or line.startswith('**'):
                acceptance_section = False
                requirement_section = 'requirement' in line_lower
                continue
                
            # Extract actionable items
            if acceptance_section or requirement_section:
                # Look for bullet points, numbered items, or "must/should" statements
                if (line.startswith('-') or line.startswith('*') or 
                    re.match(r'^\d+\.', line) or 
                    any(word in line_lower for word in ['must', 'should', 'shall', 'will'])):
                    
                    # Clean up the line and create a test step
                    clean_line = re.sub(r'^[-*\d\.]+\s*', '', line).strip()
                    if len(clean_line) > 10:  # Only meaningful steps
                        action = f"Verify that {clean_line.lower()}"
                        expected = "Requirement is satisfied and system behaves as specified"
                        steps.append(f"{step_num}. {action} -> {expected}")
                        step_num += 1
        
        # If no specific steps found, create generic validation steps
        if not steps:
            steps = [
                f"1. Set up test environment and prerequisites -> Environment is ready for testing",
                f"2. Execute the primary functionality described in requirement -> System responds correctly",
                f"3. Validate all acceptance criteria are met -> All criteria pass validation",
                f"4. Test edge cases and error conditions -> System handles edge cases properly",
                f"5. Verify system state and cleanup -> System returns to expected state"
            ]
        
        steps_block = "\n".join(steps)
        
        # Extract input data requirements
        input_data = "{}"
        if any(keyword in requirement.lower() for keyword in ['input', 'data', 'parameter', 'field']):
            input_data = '{"test_data": "Specific test data based on requirement specifications"}'
        
        return textwrap.dedent(f"""
            [TEST_CASE]
            [SUMMARY]{summary}[/SUMMARY]

            [INPUT_DATA]
            {input_data}
            [/INPUT_DATA]

            [STEPS]
            {steps_block}
            [/STEPS]

            [EXPECTED]All acceptance criteria are validated and the system meets the specified requirements[/EXPECTED]
            [/TEST_CASE]
        """).strip()
        
    def _validate_test_case(self, test_case_text: str) -> str:
        """Tool function to validate test case structure and quality"""
        import re
        
        required_sections = ['summary', 'steps', 'expected']
        missing_sections = []
        quality_issues = []
        
        # Check for required sections
        for section in required_sections:
            if section.lower() not in test_case_text.lower():
                missing_sections.append(section)
        
        # Check for quality indicators
        steps_match = re.search(r'\[STEPS\](.*?)\[/STEPS\]', test_case_text, re.IGNORECASE | re.DOTALL)
        if steps_match:
            steps_content = steps_match.group(1).strip()
            
            # Check if steps are too generic
            if 'stub' in steps_content.lower() or len(steps_content.split('\n')) < 3:
                quality_issues.append("Test steps are too generic or insufficient")
            
            # Check for validation keywords
            validation_keywords = ['verify', 'validate', 'check', 'confirm', 'ensure', 'assert']
            if not any(keyword in steps_content.lower() for keyword in validation_keywords):
                quality_issues.append("Test steps lack proper validation actions")
        
        # Check summary quality
        summary_match = re.search(r'\[SUMMARY\](.*?)\[/SUMMARY\]', test_case_text, re.IGNORECASE | re.DOTALL)
        if summary_match:
            summary_content = summary_match.group(1).strip()
            if 'stub' in summary_content.lower() or len(summary_content) < 20:
                quality_issues.append("Test summary is too generic or brief")
        
        # Return validation result
        if missing_sections:
            return f"Validation failed. Missing sections: {', '.join(missing_sections)}"
        elif quality_issues:
            return f"Validation warning. Quality issues: {'; '.join(quality_issues)}"
        else:
            return "Validation passed. Test case structure and quality are acceptable."
    
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
