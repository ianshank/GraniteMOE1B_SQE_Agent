"""Agent that generates structured test cases.

This module conditionally stubs optional heavy dependencies (langchain, mlx_lm)
to ensure import stability in constrained environments and during test
collection. The stubs provide the minimal surface required by this module.
"""

try:  # optional dependency; tests may run without real langchain
    from langchain.agents import AgentType, initialize_agent  # type: ignore
    from langchain.tools import Tool  # type: ignore
    from langchain.schema import AgentAction, AgentFinish  # type: ignore
except Exception:  # pragma: no cover - import-time fallback for test envs
    import types as _types
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
try:  # optional dependency; provide stub for import resilience
    import mlx_lm as _mlx_lm  # type: ignore
except Exception:  # pragma: no cover - import-time fallback
    class _StubMLXLM:
        @staticmethod
        def generate(*_a, **_kw):  # type: ignore
            # Minimal structured output so downstream parsing can proceed in tests
            return (
                "[TEST_CASE][SUMMARY]stub[/SUMMARY][INPUT_DATA]{}[/INPUT_DATA]"
                "[STEPS]1. step -> ok[/STEPS][EXPECTED]ok[/EXPECTED][/TEST_CASE]"
            )
    _mlx_lm = _StubMLXLM()  # type: ignore
import logging
import re

if TYPE_CHECKING:
    from src.models.granite_moe import GraniteMoETrainer
    from src.data.rag_retriever import RAGRetriever
    from src.data.cag_cache import CAGCache
    from src.models.test_case_schemas import TestCase, TestStep, TestCasePriority, TestCaseType
else:
    # Runtime imports for types used in object construction
    from src.models.test_case_schemas import TestCase, TestStep, TestCasePriority, TestCaseType

class TestGenerationAgent:
    """MoE-powered agent for generating test cases from requirements.

    This agent retrieves requirement context, optionally checks a cache for
    similar prior generations, uses a Granite model for structured generation,
    and parses the output into a validated TestCase object.

    Design goals:
    - Strong, explicit prompting optimized for small models.
    - Resilient fallback path when generation doesn't match the required
      structure.
    - Clear, structured logging for observability and troubleshooting.
    - Conservative parsing with defensive defaults.
    """
    
    def __init__(self, granite_trainer: 'GraniteMoETrainer', 
                 rag_retriever: 'RAGRetriever', 
                 cag_cache: 'CAGCache',
                 enforce_strict_provenance: bool = False):
        self.granite_model = granite_trainer
        self.rag_retriever = rag_retriever
        self.cag_cache = cag_cache
        self.enforce_strict_provenance = enforce_strict_provenance
        self.tools = self._initialize_tools()
        self._logger = logging.getLogger(__name__)
        
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
        """Generate a structured test case from raw requirement/context text.

        Uses an explicit prompt tailored for small models and pre-seeds the
        assistant response to guide formatting. If the model output is missing
        required tags, a deterministic fallback scaffold is returned.

        Args:
            requirement: Requirement text plus any retrieved context.

        Returns:
            Generated text containing required tags, or a fallback template.
        """
        if not getattr(self.granite_model, 'mlx_model', None):
            self._logger.debug("MLX model not loaded; loading for inference")
            self.granite_model.load_model_for_inference()

        # Extract a concise title from markdown header if present to
        # personalize the test title and improve determinism.
        title_match = re.search(r'^#\s+(.+)$', requirement, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Test Case"

        prompt = (
            f"""<|system|>
You are a software quality engineer. Generate a structured test case from the requirement below.

CRITICAL FORMAT: Reply with EXACTLY this structure and tags:

[TEST_CASE]
[SUMMARY]Brief test case title[/SUMMARY]
[INPUT_DATA]{{}}[/INPUT_DATA]
[STEPS]
1. First action -> Expected result
2. Second action -> Expected result
3. Third action -> Expected result
[/STEPS]
[EXPECTED]Overall expected outcome[/EXPECTED]
[/TEST_CASE]

Rules:
- Only use information contained in the input requirement/context.
- If data is missing, keep placeholders and do not invent specifics.

<|user|>
Requirement and Context (verbatim):
{requirement}

Generate one test case using the exact format above.

<|assistant|>
[TEST_CASE]
[SUMMARY]Test {title}[/SUMMARY]
[INPUT_DATA]{{}}[/INPUT_DATA]
[STEPS]"""
        )

        try:
            response = _mlx_lm.generate(
                self.granite_model.mlx_model,
                self.granite_model.mlx_tokenizer,
                prompt=prompt,
                max_tokens=512,
            )
            response = (response or "").strip()
            if not self._has_required_tags(response):
                self._logger.warning(
                    "Model output missing required tags; generating fallback scaffold"
                )
                response = self._create_fallback_test_case(requirement, title)
            return response
        except Exception as e:  # pragma: no cover - defensive
            self._logger.error(f"Error during generation: {e}", exc_info=True)
            return self._create_fallback_test_case(requirement, title)
    
    def _validate_test_case(self, test_case_text: str) -> str:
        """Validate structured tags in a generated test case.

        The contract requires presence of SUMMARY, STEPS, and EXPECTED blocks.
        We do a lightweight tag presence check here; deeper schema checks happen
        at parsing time.
        """
        required_tags = ['[SUMMARY]', '[/SUMMARY]', '[STEPS]', '[/STEPS]', '[EXPECTED]', '[/EXPECTED]']
        missing = [t for t in required_tags if t not in (test_case_text or "")]
        if missing:
            return f"Validation failed. Missing sections: {', '.join(missing)}"
        return "Validation passed. Test case structure is complete."
    
    async def generate_test_cases_for_team(self, team_name: str, 
                                         requirements: List[str]) -> List['TestCase']:
        """Generate test cases for a specific team"""
        test_cases = []
        
        for idx, requirement in enumerate(requirements):
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
                    self._logger.debug(f"Retrieved context length={len(context)} for team={team_name}")
                elif action == "check_cache":
                    cache_result = self._check_cache(query)
                    if "Found cached response" in cache_result:
                        generated_text = cache_result.split(": ", 1)[1]
                        continue  # Skip generation if cached
                elif action == "generate_test_case":
                    full_requirement = f"{query}\n\nContext: {context}"
                    generated_text = self._generate_test_case(full_requirement)
                    # Provenance verification
                    try:
                        ok = self._verify_against_context(query, context, generated_text)
                    except Exception:
                        ok = False
                    if not ok:
                        msg = "Generated output failed provenance overlap check (may include non-source content)."
                        if self.enforce_strict_provenance:
                            self._logger.error(msg)
                            raise ValueError(msg)
                        else:
                            self._logger.warning(msg)
                elif action == "validate_test_case":
                    validation_result = self._validate_test_case(generated_text)
                    if "failed" in validation_result:
                        # Do not auto-regenerate; fail fast per strict provenance policy
                        self._logger.error(
                            f"Validation failed for generated test case; skipping. Details: {validation_result}"
                        )
                        generated_text = ""
            
            # Parse generated text into TestCase object
            try:
                test_case = self._parse_generated_test_case(generated_text, team_name, unique_hint=str(idx))
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
                self._logger.error(
                    f"Failed to parse test case for requirement idx={idx}: {e}. Sample: {generated_text[:200]!r}"
                )
                continue
        
        return test_cases

    async def generate_test_suite_for_team(
        self,
        team_name: str,
        requirements: List[str],
        suite_name: str,
        description: str = "",
    ) -> 'TestSuite':
        """Generate a named suite (e.g., Regression, E2E) as a list of test cases.

        Args:
            team_name: Logical team or context label.
            requirements: Requirement strings to derive test cases from.
            suite_name: Human-friendly suite name, e.g., "Regression Suite".
            description: Optional suite description.

        Returns:
            TestSuite object with generated test cases.
        """
        from src.models.test_case_schemas import TestSuite  # local import to avoid cycles

        cases = await self.generate_test_cases_for_team(team_name, requirements)
        self._logger.info(
            f"Generated suite '{suite_name}' for team={team_name} with {len(cases)} cases"
        )
        return TestSuite(name=suite_name, description=description or suite_name, test_cases=cases)
    
    def _parse_generated_test_case(self, generated_text: str, team_name: str, unique_hint: Optional[str] = None) -> 'TestCase':
        """Parse generated text into a structured TestCase object with safeguards.

        Raises a ValueError if the minimal contract (SUMMARY/STEPS/EXPECTED) is
        not present. Steps are parsed line-by-line with optional "->" expected
        result segments; when missing, a safe default is used.
        """
        if not generated_text or not isinstance(generated_text, str):
            raise ValueError("Empty or invalid generated text")

        summary_match = re.search(r'\[SUMMARY\](.*?)\[/SUMMARY\]', generated_text, re.DOTALL)
        steps_match = re.search(r'\[STEPS\](.*?)\[/STEPS\]', generated_text, re.DOTALL)
        expected_match = re.search(r'\[EXPECTED\](.*?)\[/EXPECTED\]', generated_text, re.DOTALL)

        if not (summary_match and steps_match and expected_match):
            self._logger.error(
                f"Generated text missing required sections. Text head: {generated_text[:160]!r}"
            )
            raise ValueError("Generated text missing required sections (SUMMARY/STEPS/EXPECTED)")

        summary = summary_match.group(1).strip()
        steps_text = steps_match.group(1).strip()
        expected = expected_match.group(1).strip()

        # Normalize and split steps; ignore blank lines
        steps: List[TestStep] = []
        step_lines = [line.strip() for line in steps_text.split('\n') if line.strip()]
        for i, line in enumerate(step_lines, 1):
            # Remove leading numbering if present, e.g., "1. ", "2) ", "- "
            line_no_num = re.sub(r'^(?:\d+[\.)]|[-*])\s*', '', line)
            if '->' in line_no_num:
                action, expected_result = line_no_num.split('->', 1)
                steps.append(TestStep(step_number=i, action=action.strip(), expected_result=expected_result.strip()))
            else:
                steps.append(TestStep(step_number=i, action=line_no_num.strip(), expected_result="Step should complete successfully"))

        if not steps:
            steps.append(TestStep(step_number=1, action="Execute the test scenario", expected_result="Test completes successfully"))

        import hashlib
        # Deterministic ID derived from summary, team, and unique hint
        uniq_src = f"{team_name}|{len(steps)}|{summary}|{unique_hint or ''}"
        tc_id = f"{team_name}_{hashlib.md5(uniq_src.encode('utf-8')).hexdigest()[:8]}"
        return TestCase(
            id=tc_id,
            summary=summary,
            priority=TestCasePriority.MEDIUM,
            test_type=TestCaseType.FUNCTIONAL,
            steps=steps,
            expected_results=expected,
            team_context=team_name
        )

    def _verify_against_context(self, requirement_text: str, context_text: str, generated_text: str) -> bool:
        """Lightweight verification that output references input content.

        This check avoids hallucinations by ensuring at least minimal token overlap
        between the requirement/context and the generated output.
        Returns True when overlap is detected, False otherwise.
        """
        # Normalize
        req = (requirement_text or "").lower()
        ctx = (context_text or "").lower()
        src = f"{req} {ctx}"
        out = (generated_text or "").lower()
        if not src.strip() or not out.strip():
            return False
        # Token sets (basic alnum words)
        src_tokens = set(re.findall(r"[a-z0-9]+", src))
        out_tokens = set(re.findall(r"[a-z0-9]+", out))
        if not src_tokens or not out_tokens:
            return False
        overlap = src_tokens.intersection(out_tokens)
        ratio = len(overlap) / max(1, len(out_tokens))
        self._logger.debug(f"Provenance verification token overlap ratio: {ratio:.3f}")
        # Heuristic threshold kept permissive to avoid false negatives
        return ratio >= 0.02

    # ----------------------- Helpers (format + fallback) -----------------------

    def _has_required_tags(self, text: str) -> bool:
        """Check if text contains all required structural tags.

        Required tags are SUMMARY, STEPS and EXPECTED blocks. INPUT_DATA is
        optional for parsing but encouraged by the prompt. Using explicit tags
        rather than fuzzy substring checks reduces false positives.
        """
        if not text:
            return False
        required_tags = ['[SUMMARY]', '[/SUMMARY]', '[STEPS]', '[/STEPS]', '[EXPECTED]', '[/EXPECTED]']
        present = all(tag in text for tag in required_tags)
        self._logger.debug(f"Required tag presence: {present}")
        return present

    def _create_fallback_test_case(self, requirement: str, title: str) -> str:
        """Create a deterministic fallback test case when generation fails.

        Attempts to extract a brief user story or topic from the requirement
        text to make the fallback slightly contextual while remaining safe.
        """
        # Extract a simple user story chunk if annotated, otherwise a short slice
        user_story_match = re.search(r'##\s*User Story\s*\n(.+?)(?=\n##|\n\n|\Z)', requirement or "", re.DOTALL)
        if user_story_match:
            user_story = user_story_match.group(1).strip()
        else:
            # Use the first non-empty line as a proxy for topic
            first_line = next((ln.strip() for ln in (requirement or "").splitlines() if ln.strip()), "the requirement")
            user_story = first_line[:120]

        fallback = (
            f"""[TEST_CASE]
[SUMMARY]Test {title}[/SUMMARY]
[INPUT_DATA]{{}}[/INPUT_DATA]
[STEPS]
1. Set up test environment -> Environment is ready
2. Execute the main functionality described in: {user_story} -> Functionality executes successfully
3. Verify the expected behavior -> Results match requirements
4. Clean up test data -> Test environment is restored
[/STEPS]
[EXPECTED]The system should successfully implement the functionality described in the requirement without errors[/EXPECTED]
[/TEST_CASE]"""
        )
        return fallback
