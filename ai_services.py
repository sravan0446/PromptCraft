import google.generativeai as genai
import re
import json
from typing import List, Dict
from database import PromptDatabase

class PromptEngineer:
    def __init__(self, api_key: str, model_name: str = 'gemini-1.5-flash-latest'):
        """
        Initializes the PromptEngineer.
        
        Args:
            api_key (str): The Google Gemini API key.
            model_name (str): The name of the Gemini model to use.
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.db = PromptDatabase()
    
    def analyze_user_intent(self, user_goal: str) -> Dict:
        """Analyze user intent and categorize the request"""
        analysis_prompt = f"""
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
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            # A more robust way to find JSON in the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else: # Fallback to the original regex if the formatted one fails
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            # Return a default structure if JSON parsing fails
            return {"category": "General", "complexity": 3, "intent": user_goal, "tags": []}
        except Exception as e:
            # Propagate the exception to be handled by the UI layer
            raise Exception(f"Error analyzing intent with API: {str(e)}")
    
    def generate_prompt_variations(self, user_goal: str, similar_prompts: List[Dict] = None) -> List[str]:
        """Generate multiple enhanced prompt variations using RAG-enhanced context"""
        rag_context = ""
        if similar_prompts:
            rag_context = f"\n\n=== RAG CONTEXT: SUCCESSFUL PROMPTS FROM KNOWLEDGE BASE ===\n"
            rag_context += f"Found {len(similar_prompts)} similar high-rated prompts to inform generation:\n\n"
            for i, prompt_data in enumerate(similar_prompts, 1):
                rag_context += f"EXAMPLE {i} (Rating: {prompt_data['rating']}/5):\n"
                rag_context += f"  Original Goal: \"{prompt_data['goal']}\"\n"
                rag_context += f"  Category: {prompt_data.get('category', 'General')}\n"
                rag_context += f"  Successful Prompt Pattern:\n    {prompt_data['prompt'][:300]}...\n"
                if prompt_data.get('feedback'):
                    rag_context += f"  User Feedback: \"{prompt_data['feedback']}\"\n"
                rag_context += f"  Tags: {', '.join(prompt_data.get('tags', []))}\n\n"
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
        
        try:
            response = self.model.generate_content(generation_prompt)
            variations = self._parse_prompt_variations(response.text)
            
            if len(variations) < 3:
                # Fallback if parsing fails to get enough variations
                variations.extend(self._generate_fallback_prompts(user_goal, len(variations)))
            
            return variations[:3]
        except Exception as e:
            # Propagate for UI handling and return a safe fallback
            raise Exception(f"Error in RAG-enhanced prompt generation: {str(e)}")

    def _parse_prompt_variations(self, text: str) -> List[str]:
        """Parses the generated text to extract up to three prompts."""
        # Use regex to find all content between the delimiters, robust to extra whitespace
        matches = re.findall(r'===\s*PROMPT_\d+\s*===(.*?)(?===|$)', text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]

    def _generate_fallback_prompts(self, user_goal: str, num_existing: int) -> List[str]:
        """Generate simple fallback prompts to ensure there are always 3."""
        fallbacks = [
            f"Create a detailed and comprehensive response for: {user_goal}\n\nProvide thorough analysis, specific examples, and actionable recommendations.",
            f"Step-by-step approach for: {user_goal}\n\n1. Analyze the requirements\n2. Break down into manageable parts\n3. Provide clear implementation steps\n4. Include examples and best practices",
            f"Creative and innovative solution for: {user_goal}\n\nThink outside the box and provide unique perspectives, creative alternatives, and novel approaches to achieve this goal."
        ]
        return fallbacks[num_existing:]