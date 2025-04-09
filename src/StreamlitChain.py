from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap
import streamlit as st 

# === Initialize Ollama LLM with gemma3:1b model ===
llm = OllamaLLM(model="gemma3:1b", temperature=0.7)

# Define the first prompt for generating the story
def generate_lullaby(location, name):
    template = """ 
        As a children's book writer, please come up with a simple and short (90 words)
        lullaby based on the location
        {location}
        and the main character {name}
        
        STORY:
    """

    # Create the prompt template
    prompt = PromptTemplate(input_variables=["location", "name"], template=template)

    # Generate the lullaby using the model
    story_runnable = prompt | llm
    story = story_runnable.invoke({"location": location, "name": name})
    
    return story

# Define the second prompt for translating the story
def translate_lullaby(story, language):
    template_update = """
    Translate the {story} into {language}. Make sure 
    the language is simple and fun.

    TRANSLATION:
    """

    # Create the translation prompt template
    prompt_translate = PromptTemplate(input_variables=["story", "language"], template=template_update)

    # Translate the story using the model
    translation_runnable = prompt_translate | llm
    translated = translation_runnable.invoke({"story": story, "language": language})
    
    return translated

# Main function to tie everything together and interact with Streamlit
def main():
    st.set_page_config(page_title="Generate Children's Lullaby", layout="centered")
    st.title("Let AI Write and Translate a Lullaby for You 📖")
    st.header("Get Started...")

    location_input = st.text_input(label="Where is the story set?")
    main_character_input = st.text_input(label="What's the main character's name?")
    language_input = st.text_input(label="Translate the story into...")

    submit_button = st.button("Submit")
    
    if location_input and main_character_input and language_input:
        if submit_button:
            with st.spinner("Generating lullaby..."):
                # Step 1: Generate the lullaby story
                story = generate_lullaby(location=location_input, name=main_character_input)
                
                # Step 2: Translate the lullaby story
                translated = translate_lullaby(story=story, language=language_input)

                # Display the results
                with st.expander("English Version"):
                    st.write(story)
                with st.expander(f"{language_input} Version"):
                    st.write(translated)
                
            st.success("Lullaby Successfully Generated!")

# Invoke the main function to run the Streamlit app
if __name__ == '__main__':
    main()
