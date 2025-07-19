from __future__ import annotations
import google.generativeai as genai
import re
import json
import logging
import time
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from database import PromptDatabase

@dataclass
class IntentAnalysis:
    category: str
    complexity: int
    intent: str
    tags: List[str]

class PromptEngineer:
    def __init__(self, api_key: str, model_name: str = 'gemini-1.5-flash-latest') -> None:
        """
        Initializes the PromptEngineer with enhanced error handling.
        
        Args:
            api_key: The Google Gemini API key
            model_name: The name of the Gemini model to use
        """
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.db = PromptDatabase()
            self.max_retries: int = 2
            logging.info(f"PromptEngineer initialized with model: {model_name}")
        except Exception as e:
            raise Exception(f"Failed to initialize PromptEngineer: {str(e)}") from e

    def analyze_user_intent(self, user_goal: str) -> Dict[str, Union[str, int, List[str]]]:
        """Analyze user intent and categorize the request with retry logic."""
        analysis_prompt = f'''
        Analyze this user goal and provide a structured response:
        User Goal: "{user_goal}"

        Please provide:
        1. Category (e.g., "Content Creation", "Business Strategy", "Technical", "Creative Writing", etc.)
        2. Complexity Level (1-5, where 1 is simple and 5 is complex)
        3. Key Intent (what the user really wants to achieve)
        4. Suggested Tags (3-5 relevant tags)

        Format your response as JSON:
        {{
            "category": "category_name",
            "complexity": 3,
            "intent": "clear description of intent",
            "tags": ["tag1", "tag2", "tag3"]
        }}
        '''

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(analysis_prompt)
                
                # Enhanced JSON parsing with multiple fallback methods
                if json_match := re.search(r'``````', response.text, re.DOTALL):
                    return json.loads(json_match.group(1))
                
                # Fallback to finding any JSON object
                if json_match := re.search(r'\{.*\}', response.text, re.DOTALL):
                    return json.loads(json_match.group())
                
                # Safe default if JSON parsing fails
                return {
                    "category": "General",
                    "complexity": 3,
                    "intent": user_goal,
                    "tags": ["general", "prompt", "assistance"]
                }
                
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return {
                        "category": "General",
                        "complexity": 3,
                        "intent": user_goal,
                        "tags": ["general", "prompt", "assistance"]
                    }
                time.sleep(0.5)
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Error analyzing intent: {str(e)}") from e
                time.sleep(0.5)

    def generate_prompt_variations(
        self, 
        user_goal: str, 
        similar_prompts: Optional[List[Dict]] = None
    ) -> List[str]:
        """Generate multiple enhanced prompt variations with improved error handling."""
        
        rag_context = ""
        if similar_prompts:
            rag_context = f"\n\n=== RAG CONTEXT: SUCCESSFUL PROMPTS FROM KNOWLEDGE BASE ===\n"
            rag_context += f"Found {len(similar_prompts)} similar high-rated prompts to inform generation:\n\n"
            
            for i, prompt_data in enumerate(similar_prompts, 1):
                rag_context += f"EXAMPLE {i} (Rating: {prompt_data['rating']}/5):\n"
                rag_context += f" Original Goal: \"{prompt_data['goal']}\"\n"
                rag_context += f" Category: {prompt_data.get('category', 'General')}\n"
                rag_context += f" Successful Prompt Pattern:\n {prompt_data['prompt'][:300]}...\n"
                
                if feedback := prompt_data.get('feedback'):
                    rag_context += f" User Feedback: \"{feedback}\"\n"
                rag_context += f" Tags: {', '.join(prompt_data.get('tags', []))}\n\n"
            
            rag_context += "Use these successful patterns as inspiration while adapting them to the current goal.\n"
            rag_context += "=== END RAG CONTEXT ===\n\n"

        generation_prompt = f"""
        You are an expert prompt engineer. Your task is to create 3 completely separate, highly effective prompt variations for the following user goal.

        User Goal: "{user_goal}"

        {rag_context}

        INSTRUCTIONS:
        1. Analyze the RAG context to understand successful patterns.
        2. Create 3 distinct, complete prompts someone could use immediately.
        3. Each prompt should incorporate best practices while being tailored to the goal.

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

        ===PROMPT_1===
        [Complete first prompt here]

        ===PROMPT_2===
        [Complete second prompt here]

        ===PROMPT_3===
        [Complete third prompt here]

        PROMPT QUALITY REQUIREMENTS: Be clear, specific, actionable, and standalone.

        DIFFERENTIATION STRATEGY:
        - Prompt 1: Comprehensive approach with rich context.
        - Prompt 2: Structured methodology with clear steps.
        - Prompt 3: Creative/innovative angle.

        Do NOT add explanations or extra text.
        """

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(generation_prompt)
                variations = self._parse_prompt_variations(response.text)
                
                if len(variations) < 3:
                    # Ensure we always have 3 prompts
                    variations.extend(self._generate_fallback_prompts(user_goal, len(variations)))
                
                return variations[:3]
                
            except Exception as e:
                logging.warning(f"Prompt generation attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    # Return fallback prompts if all attempts fail
                    return self._generate_fallback_prompts(user_goal, 0)
                time.sleep(0.5)

    def _parse_prompt_variations(self, text: str) -> List[str]:
        """Enhanced parsing of generated text to extract prompts."""
        try:
            # Use regex to find all content between the delimiters
            matches = re.findall(r'===\s*PROMPT_\d+\s*===(.*?)(?===|$)', text, re.DOTALL)
            parsed_prompts = [match.strip() for match in matches if match.strip()]
            
            # Filter out prompts that are too short
            valid_prompts = [prompt for prompt in parsed_prompts if len(prompt) > 20]
            
            return valid_prompts
            
        except Exception as e:
            logging.warning(f"Error parsing prompt variations: {e}")
            return []

    def _generate_fallback_prompts(self, user_goal: str, num_existing: int) -> List[str]:
        """Generate enhanced fallback prompts to ensure reliability."""
        fallbacks = [
            f"Create a detailed and comprehensive response for: {user_goal}\n\n"
            f"Provide thorough analysis, specific examples, and actionable recommendations. "
            f"Consider multiple perspectives and potential challenges.",
            
            f"Step-by-step approach for: {user_goal}\n\n"
            f"1. Analyze the requirements and context\n"
            f"2. Break down into manageable components\n"
            f"3. Provide clear implementation steps\n"
            f"4. Include examples and best practices\n"
            f"5. Address potential obstacles and solutions",
            
            f"Creative and innovative solution for: {user_goal}\n\n"
            f"Think outside the box and provide unique perspectives, creative alternatives, "
            f"and novel approaches. Consider unconventional methods and emerging trends "
            f"that could enhance the outcome."
        ]

        return fallbacks[num_existing:]
