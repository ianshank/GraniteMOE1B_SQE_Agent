from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.schema import AgentAction, AgentFinish
from typing import List, Dict, Any, Optional
import asyncio
from typing import TYPE_CHECKING
from mlx_lm import generate

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
        """Optimized generation for 1B model - faster inference"""
        if not getattr(self.granite_model, 'mlx_model', None):
            self.granite_model.load_model_for_inference()
        
        prompt = f"""<|system|>
You are a software quality engineer. Create a detailed test case for the given requirement.

<|user|>
Requirement: {requirement}
Create a comprehensive test case with summary, input data, steps, and expected results.

<|assistant|>"""
        
        response = generate(
            self.granite_model.mlx_model,
            self.granite_model.mlx_tokenizer,
            prompt=prompt,
            max_tokens=384,
            temp=0.6
        )
        
        return response
    
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
        # Implementation would parse the structured output
        # A simplified version
        import re
        
        summary_match = re.search(r'\[SUMMARY\](.*?)\[/SUMMARY\]', generated_text, re.DOTALL)
        steps_match = re.search(r'\[STEPS\](.*?)\[/STEPS\]', generated_text, re.DOTALL)
        expected_match = re.search(r'\[EXPECTED\](.*?)\[/EXPECTED\]', generated_text, re.DOTALL)
        
        summary = summary_match.group(1).strip() if summary_match else "Generated test case"
        steps_text = steps_match.group(1).strip() if steps_match else ""
        expected = expected_match.group(1).strip() if expected_match else "System should work as expected"
        
        # Parse steps
        steps = []
        step_lines = steps_text.split('\n')
        for i, line in enumerate(step_lines, 1):
            if line.strip():
                if '->' in line:
                    action, expected_result = line.split('->', 1)
                    steps.append(TestStep(
                        step_number=i,
                        action=action.strip(),
                        expected_result=expected_result.strip()
                    ))
                else:
                    steps.append(TestStep(
                        step_number=i,
                        action=line.strip(),
                        expected_result="Step should complete successfully"
                    ))
        
        return TestCase(
            id=f"{team_name}_{len(steps)}_{hash(summary) % 1000}",
            summary=summary,
            priority=TestCasePriority.MEDIUM,
            test_type=TestCaseType.FUNCTIONAL,
            steps=steps,
            expected_results=expected,
            team_context=team_name
        )
