import streamlit as st
from datetime import datetime
from typing import List, Dict

# Import modular components
from database import PromptDatabase
from ai_services import PromptEngineer

# --- App Configuration ---
st.set_page_config(
    page_title="PromptCraft - AI Prompt Engineering Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
def init_session_state():
    """Initialize session state variables."""
    if 'generated_prompts' not in st.session_state:
        st.session_state.generated_prompts = []
    if 'selected_prompt' not in st.session_state:
        st.session_state.selected_prompt = None
    if 'rated_prompts' not in st.session_state:
        st.session_state.rated_prompts = set()
    if 'show_rating_form' not in st.session_state:
        st.session_state.show_rating_form = False
    if 'user_goal' not in st.session_state:
        st.session_state.user_goal = ""
    if 'intent_analysis' not in st.session_state:
        st.session_state.intent_analysis = {}
    if 'similar_prompts_used' not in st.session_state:
        st.session_state.similar_prompts_used = []

# --- Main Application ---
def main():
    init_session_state()

    # --- Sidebar ---
    st.sidebar.title("🎯 PromptCraft")
    st.sidebar.markdown("*AI Prompt Engineering Assistant*")
    
    api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")
    
    # UPDATED: Added a custom model input option
    predefined_models = (
        "gemini-1.5-pro-latest",    # Top-tier GA model
        "gemini-1.5-flash-latest",  # Fast and efficient GA model
        "gemini-1.0-pro",           # Stable older GA model
        "gemini-2.0-flash",
        "gemini-2.0-pro",
        "gemini-2.5-flash",
        "gemini-2.5-pro"
    )
    custom_option = "Enter Custom Model..."
    model_options = predefined_models + (custom_option,)

    selected_option = st.sidebar.selectbox(
        "Choose a Gemini Model",
        options=model_options,
        index=1, # Default to gemini-1.5-flash-latest
        help="Select a standard model or choose 'Custom' to enter a specific one (e.g., a preview model)."
    )

    model_name = ""
    if selected_option == custom_option:
        model_name = st.sidebar.text_input(
            "Enter Custom Model Name:",
            value="gemini-1.5-pro-latest",
            help="Enter the exact model ID from Google AI's documentation."
        ).strip()
    else:
        model_name = selected_option

    if not api_key:
        st.sidebar.warning("Please enter your Gemini API key to continue.")
        st.sidebar.markdown("Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
        return
    
    # Ensure a model name is available before proceeding
    if not model_name:
        st.sidebar.error("Please enter a custom model name.")
        return

    try:
        db = PromptDatabase()
        engineer = PromptEngineer(api_key=api_key, model_name=model_name)
    except Exception as e:
        st.error(f"Error initializing PromptCraft: {e}")
        st.info("Please ensure your API key is correct and that the selected model is valid and available for your account.")
        return

    # Stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 PromptCraft Stats")
    try:
        stats = db.get_stats()
        st.sidebar.metric("Total Prompts", stats.get('total_prompts', 0))
        st.sidebar.metric("Rated Prompts", stats.get('rated_prompts', 0))
        st.sidebar.metric("Average Rating", f"{stats.get('avg_rating', 0)}/5")
    except Exception as e:
        st.sidebar.error(f"Could not load stats: {e}")

    # --- Main Interface ---
    st.title("🎯 PromptCraft")
    st.markdown("### Transform your ideas into powerful AI prompts")
    
    user_goal = st.text_area(
        "What do you want to create a prompt for?",
        placeholder="e.g., 'Create a weekly social media content plan for my coffee shop' or 'Write a professional email to decline a job offer'",
        height=100
    )
    
    if st.button("🚀 Generate Prompts", type="primary"):
        if user_goal:
            with st.spinner(f"Analyzing your goal and crafting prompts using `{model_name}`..."):
                try:
                    # Reset state for new generation
                    st.session_state.generated_prompts = []
                    st.session_state.rated_prompts = set()
                    st.session_state.show_rating_form = False

                    intent_analysis = engineer.analyze_user_intent(user_goal)
                    similar_prompts = db.get_similar_prompts(user_goal)
                    variations = engineer.generate_prompt_variations(user_goal, similar_prompts)
                    
                    st.session_state.user_goal = user_goal
                    st.session_state.generated_prompts = variations
                    st.session_state.intent_analysis = intent_analysis
                    st.session_state.similar_prompts_used = similar_prompts
                    
                    db.log_usage(user_goal, len(variations))
                except Exception as e:
                    st.error(f"An error occurred during prompt generation: {e}")
                    st.info("This could be due to an invalid custom model name, API key issues, or content safety filters.")
        else:
            st.warning("Please enter a goal to generate prompts.")

    # --- Display Results ---
    if st.session_state.generated_prompts:
        st.markdown("---")
        st.subheader("✨ Generated Prompt Variations")
        
        display_analysis_and_rag()
        display_prompt_tabs()
        display_rating_form(db)
    
    # --- Footer ---
    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit and Google Gemini*")

def display_analysis_and_rag():
    """Displays the expander with intent analysis and RAG context."""
    with st.expander("🔍 Intent Analysis & RAG Context", expanded=True):
        analysis = st.session_state.intent_analysis
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Category", analysis.get('category', 'N/A'))
        col2.metric("Complexity", f"{analysis.get('complexity', 'N/A')}/5")
        col3.write(f"**Tags:** {', '.join(analysis.get('tags', []))}")
        
        st.markdown("### 🧠 RAG Knowledge Base Context")
        if st.session_state.similar_prompts_used:
            st.success(f"Found {len(st.session_state.similar_prompts_used)} similar high-rated prompts to enhance generation.")
            for i, p in enumerate(st.session_state.similar_prompts_used, 1):
                st.write(f"**Reference {i}** (⭐{p['rating']}/5): *{p['goal']}*")
        else:
            st.info("No highly-rated similar prompts found. Your feedback will help build our knowledge base!")

def display_prompt_tabs():
    """Displays the generated prompts in separate tabs."""
    tabs = st.tabs([f"Variation {i+1}" for i in range(len(st.session_state.generated_prompts))])
    for i, (tab, prompt) in enumerate(zip(tabs, st.session_state.generated_prompts)):
        with tab:
            st.text_area(f"Prompt {i+1}", value=prompt, height=250, key=f"prompt_{i}", label_visibility="collapsed")
            
            cols = st.columns(3)
            if cols[0].button(f"📋 Copy", key=f"copy_{i}"):
                st.code(prompt, language="text")
                st.success("Prompt displayed above for easy copying!")
            
            if i in st.session_state.rated_prompts:
                cols[1].success("✅ Rated")
            elif cols[1].button(f"⭐ Rate", key=f"rate_{i}"):
                st.session_state.selected_prompt = i
                st.session_state.show_rating_form = True
                st.rerun()
            
            if cols[2].button(f"🧪 Test", key=f"test_{i}"):
                st.info("💡 Copy the prompt and test it with your preferred AI tool!")

def display_rating_form(db: PromptDatabase):
    """Displays the rating form when a prompt is selected for rating."""
    if st.session_state.show_rating_form and st.session_state.selected_prompt is not None:
        idx = st.session_state.selected_prompt
        prompt_to_rate = st.session_state.generated_prompts[idx]

        with st.form(key=f"rating_form_{idx}"):
            st.subheader(f"⭐ Rate Variation {idx + 1}")
            st.text_area("Selected Prompt", value=prompt_to_rate, height=150, disabled=True)
            
            rating = st.selectbox("Rating (1-5 stars)", options=[5, 4, 3, 2, 1], format_func=lambda x: f"{'⭐' * x}")
            feedback = st.text_area("Optional feedback", placeholder="What worked well? What could be improved?")
            
            submitted = st.form_submit_button("✅ Submit Rating", type="primary")
            
            if submitted:
                try:
                    db.save_prompt(
                        user_goal=st.session_state.user_goal,
                        generated_prompt=prompt_to_rate,
                        rating=rating,
                        feedback=feedback,
                        category=st.session_state.intent_analysis.get('category'),
                        tags=st.session_state.intent_analysis.get('tags')
                    )
                    st.session_state.rated_prompts.add(idx)
                    st.session_state.show_rating_form = False
                    st.session_state.selected_prompt = None
                    st.success("🎉 Thank you for your feedback!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save rating: {e}")

if __name__ == "__main__":
    main()