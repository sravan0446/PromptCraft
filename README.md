# 🎯 PromptCraft: AI Prompt Engineering Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-red.svg)](https://streamlit.io)

PromptCraft is an intelligent, web-based assistant designed to help you transform your ideas into powerful, effective, and well-structured prompts for generative AI models. It bridges the gap between a simple goal and a high-quality prompt that yields superior results.

The application leverages a Retrieval-Augmented Generation (RAG) system, which learns from user-rated prompts stored in a local knowledge base. Over time, as more prompts are generated and rated, PromptCraft becomes smarter and provides even better suggestions tailored to similar goals.

## ✨ Key Features

*   **Intelligent Prompt Generation:** Describe your goal in plain English, and PromptCraft will generate three distinct, high-quality prompt variations.
*   **Flexible AI Model Selection:** Choose from a curated list of the latest stable Gemini models or enter a custom model ID to experiment with preview or unlisted versions.
*   **Retrieval-Augmented Generation (RAG):** Uses a knowledge base of previously successful, high-rated prompts to inform the generation of new ones.
*   **User Intent Analysis:** Automatically analyzes your goal to determine its category, complexity, and key tags.
*   **Feedback Loop:** Rate the generated prompts to help the system learn. Your feedback directly improves future suggestions for everyone.
*   **Local Knowledge Base:** All prompts and ratings are stored locally in a SQLite database (`promptcraft.db`), giving you full control over your data.

## 🚀 Getting Started

Follow these instructions to set up and run PromptCraft on your local machine.

### Prerequisites

*   Python 3.10
*   A Google Gemini API Key. You can get a free key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/promptcraft.git
    cd promptcraft
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```

5.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## 💻 Usage

1.  **Enter API Key:** Paste your Google Gemini API Key into the input field in the sidebar.
2.  **Select a Model:** Choose your desired Gemini model from the dropdown.
    *   **Standard Models:** Select from the list for a balance of power and stability (e.g., `gemini-1.5-pro-latest`, `gemini-1.5-flash-latest`).
    *   **Custom Models:** To use a preview or unlisted model, select `"Enter Custom Model..."` and type the exact model ID into the text field that appears.
3.  **Define Your Goal:** In the main text area, describe what you want the AI to do.
4.  **Generate Prompts:** Click the `🚀 Generate Prompts` button.
5.  **Review Variations:** PromptCraft will produce three distinct prompt variations.
6.  **Copy and Test:** Use the `📋 Copy` button to copy a prompt.
7.  **Rate the Prompts:** Click the `⭐ Rate` button and provide feedback to improve the system.

## 🤔 FAQ

**Q: Why would I use a custom model name?**
**A:** Google often releases new models in a "preview" state before they are generally available. These models might have new features or improved performance. The custom model input allows you to experiment with these new models (e.g., `gemini-2.5-pro-preview-08-01`) as soon as they are announced, without waiting for an application update.

**Q: What's the difference between the listed Gemini models?**
**A:** The models offer a trade-off between performance, speed, and cost. `gemini-2.0-pro` is highly capable for complex tasks, while `gemini-2.0-flash` is optimized for speed and cost-efficiency.

**Q: Where is my data (prompts, ratings) stored?**
**A:** All data is stored locally in a file named `promptcraft.db` in the project's root directory. You have full ownership and control over this data.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  **Fork the Project**
2.  **Create your Feature Branch** (`git checkout -b feature/AmazingFeature`)
3.  **Commit your Changes** (`git commit -m 'Add some AmazingFeature'`)
4.  **Push to the Branch** (`git push origin feature/AmazingFeature`)
5.  **Open a Pull Request**

## 📄 License

This project is distributed under the MIT License.
