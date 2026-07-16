import os
import streamlit as st
from groq import Groq
import json
import io
from tavily import TavilyClient
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from gtts import gTTS
from moviepy.editor import *
from fpdf import FPDF # Added for PDF conversion

# Imports for Hugging Face image generation
from diffusers import DiffusionPipeline
import torch # Assuming torch is available in Colab

# Ensure ffmpeg is available (usually pre-installed in Colab)
# Or you might need to install it: !apt-get update && !apt-get install -y ffmpeg

st.markdown(
    """
<style>
h1 {
    color: #FF4B4B; /* Change the title color */
    font-size: 3em; /* Increase font size */
    text-align: center;
}
</style>
""",
    unsafe_allow_html=True
)

# Get API keys from environment variables
groq_api_key = os.environ.get('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY environment variable not found.")
    st.stop()  # Stop the app if API key is missing
client = Groq(api_key=groq_api_key)

tavily_api_key = os.environ.get('TAVILY_API_KEY')
if not tavily_api_key:
    st.error("TAVILY_API_KEY environment variable not found.")
    st.stop()  # Stop the app if API key is missing

st.markdown(
    """
<style>
h1 {
    color: #FF4B4B; /* Change the title color */
    font-size: 3em; /* Increase font size */
    text-align: center;
}
body {
    background-color: #f0f2f6; /* Example: light grey background */
}
</style>
""",
    unsafe_allow_html=True
)

selected_page = st.sidebar.selectbox(
    "Choose a page",
    ['Research Assistant', 'StudyMate', 'Video Generation Engine', 'Voice to Text', 'Text to PDF'] # Added 'Video Generation Engine' back

)



if selected_page == 'Research Assistant':
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add an initial system message to set the persona for the assistant
        st.session_state.messages.append(
            {"role": "system", "content": "You are a helpful assistant."}
        )

    def chat_with_groq(messages):
        # Call the API with stream=True to get a streaming response
        return client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,  # Pass the entire message history for context
            temperature=0.7,
            stream=True
        )

    def stream_response(current_messages):
        full_response = ""
        placeholder = st.empty()
        for chunk in chat_with_groq(current_messages):
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                # Using st.markdown for potential rich text and blinking cursor
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)  # Display final response
        return full_response  # Return the full response for logging

    st.set_page_config(
        page_title="Research Assistance with Lion AI",
        page_icon="🦁",
        layout="wide"
    )
    st.title("🦁Chat with Lion AI Assistant")

    st.write(
        "This is Lion AI research that provides fast and accurate "
        "solutions to your questions."
    )
    st.write("___")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # Don't display the system message directly to the user
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_input = st.chat_input("Here is Lion AI, ask anything")

    with st.spinner("Generating response..."):
        if user_input:
            # Add user message to chat history and display it
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate and stream assistant response
            with st.chat_message("assistant"):
                assistant_response = stream_response(
                    st.session_state.messages
                )
                # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )
        else:
            # Only show this message if no user input has been provided yet
            if (
                len(st.session_state.messages) == 1 and
                    st.session_state.messages[0]["role"] == "system"):
                st.write("Please enter a message to start the conversation with Lion.")

elif selected_page == 'StudyMate':
    # ---------- clients (built once, using secrets) ----------
    tavily = TavilyClient(api_key=tavily_api_key)

    st.set_page_config(
        page_title="Study With Lion AI",
        page_icon="🦁",
        layout="centered"
    )
    st.title("📚 Lion AI is Your StudyMate")
    st.caption(
        "Your research & homework assistant - "
        "upload your notes, then ask anything."
    )
    # Groq-hosted model used for the agent's reasoning + tool calling.
    # openai/gpt-oss-120b is Groq's current recommended model for strong
    # tool-use quality.
    MODEL_NAME = "openai/gpt-oss-120b"

    SYSTEM_PROMPT = (
        "You are StudyMate, a helpful research and homework assistant. "
        "Users will upload their notes via a sidebar, and these notes will be "
        "indexed and available through the 'search_my_notes' tool. "
        "Always use the 'search_my_notes' tool first for anything "
        "that could plausibly be covered in their uploaded (and already "
        "indexed) document. Be clear and concise. If neither tool has the "
        "answer, say so honestly instead of guessing. DO NOT ask the user "
        "to re-upload documents if they have already been indexed. The "
        "'search_my_notes' tool already has access to indexed documents." 
          "' if anyone ask about you name, tell them that your name is Lion AI."
    )


    @st.cache_resource
    def get_collection():
        chroma_client = chromadb.Client()
        embed_fn = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=("all-MiniLM-L6-v2")
            )
        )
        return chroma_client.get_or_create_collection(
            name="my_notes",
            embedding_function=embed_fn
        )


    collection = get_collection()


    # ---------- session state ----------
    # The conversation list IS the agent's memory, and its first entry is the
    # system prompt (Groq/OpenAI-style messages put the system prompt in the
    # messages list itself, rather than passing it as a separate argument).
    if "conversation" not in st.session_state:
        st.session_state.conversation = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    if "doc_indexed" not in st.session_state:
        st.session_state.doc_indexed = False


    # ---------- document ingestion ----------
    def extract_text(uploaded_file):
        if uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        return (uploaded_file.read().decode("utf-8", errors="ignore"))


    def chunk_text(text, chunk_size=800, overlap=100):
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return [c.strip() for c in chunks if c.strip()]


    with st.sidebar:
        st.header("Your documents")
        uploaded_file = st.file_uploader(
            "Upload notes (PDF or .txt)",
            type=["pdf", "txt"]
        )
        if uploaded_file is not None and st.button("Index this document"):
            with st.spinner("Reading and indexing..."):
                text = extract_text(uploaded_file)
                chunks = chunk_text(text)
                existing = collection.count()
                collection.add(
                    documents=chunks,
                    ids=[
                        f"chunk_{existing + i}"
                        for i in range(len(chunks))
                    ],
                )
                st.session_state.doc_indexed = True
            st.success(
                f"Indexed {len(chunks)} chunks from {uploaded_file.name}."
            )
        if st.session_state.doc_indexed:
            st.info(
                "StudyMate will check your notes first before searching the web."
            )
        if st.button("Clear conversation"):
            st.session_state.conversation = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            st.rerun()



    # ---------- tools ----------
    def search_my_notes(query: str) -> str:
        if collection.count() == 0:
            return "No documents have been uploaded yet."
        results = collection.query(query_texts=[query], n_results=3)
        if not results["documents"][0]:
            return "No relevant content found in the uploaded document."
        return "\n\n".join(results["documents"][0])


    def web_search(query: str) -> str:
        results = tavily.search(query=query, max_results=3)
        formatted_results = []
        for r in results["results"]:
            title_part = f"- {r['title']}: "
            content_part = f"{r['content'][:200]}"
            formatted_results.append(
                title_part + content_part
            )
        return "\n".join(formatted_results)


    def calculator(expression: str) -> str:
        try:
            # Using eval is dangerous. For a real app, a safer math parser
            # should be used. For this example, it's sufficient.
            safe_globals = {"__builtins__": {}}
            return str(eval(expression, safe_globals))
        except Exception as e:
            return (
f"Error evaluating expression: {e}")


    # Groq's API is OpenAI-compatible, so tools are described in OpenAI's
    # "function calling" schema: each tool is wrapped in a
    # {"type": "function", "function": {...}} object,
    # with parameters as JSON Schema.
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_my_notes",
                "description": (
                    "Search the user's own uploaded document/notes for "
                    "relevant content. ALWAYS try this first for anything "
                    "that could be covered in the user's material before "
                    "searching the open web."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look for in the notes."
                        }
                    },
                    "required": ["query"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the open web for current facts, definitions, "
                    "or general knowledge not found in the user's own notes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query."
                        }
                    },
                    "required": ["query"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": (
                    "Evaluate a mathematical expression, e.g. for grade "
                    "averages, percentages, or unit conversions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A Python-evaluable math "
                                           "expression."
                        }
                    },
                    "required": ["expression"],
                },
            }
        }
    ]


    def run_tool(name, tool_input):
        if name == "search_my_notes":
            return search_my_notes(**tool_input)
        elif name == "web_search":
            return web_search(**tool_input)
        elif name == "calculator":
            return calculator(**tool_input)
        return f"Error: tool '{name}' does not exist."


    def run_agent(messages, max_iterations=6):
        for _ in range(max_iterations):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=1024,
                tools=TOOLS,
                messages=messages,
            )
            message = response.choices[0].message
            # Groq's SDK wants the assistant message appended as a plain dict.
            messages.append(
                message.model_dump(
                    exclude_none=True
                )
            )

            if not message.tool_calls:
                return message.content, messages

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)
                result = run_tool(tool_name, tool_input)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })
        return (
            "Sorry, I couldn't finish that in time.",
            messages
        )


    # ---------- chat UI ----------
    for msg in st.session_state.conversation:
        if (
            msg["role"] in ("user", "assistant") and
                isinstance(msg.get("content"), str) and msg["content"]):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    user_input = st.chat_input("Ask about your notes, or anything else...")
    if user_input:
        st.session_state.conversation.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, st.session_state.conversation = \
                    run_agent(st.session_state.conversation)
                st.write(answer)

elif selected_page == 'Video Generation Engine':
    st.set_page_config(
        page_title="Video Generation with Lion AI",
        page_icon="🎥",
        layout="wide"
    )
    st.title("🎥 Lion AI Video Generation")
    st.write("Turn your text into a captivating video!")

    story_text = st.text_area("Enter your story or text here:", height=300)

    if st.button("Generate Video"):
        if not story_text:
            st.warning("Please enter some text to generate a video.")
        else:
            with st.spinner("Processing story and generating video..."):
                # Load the Hugging Face image generation model
                st.info("Loading image generation model from Hugging Face...")
                try:
                    hf_model_name = "runwayml/stable-diffusion-v1-5"
                    # Using DiffusionPipeline for text-to-image.
                    # Ensure you have a GPU enabled for performance (`Runtime` -> `Change runtime type`).
                    pipeline = DiffusionPipeline.from_pretrained(hf_model_name, torch_dtype=torch.float16)
                    pipeline.to("cuda")
                    st.success("Image generation model loaded!")
                except Exception as e:
                    st.error(f"Error loading Hugging Face model: {e}. Please ensure you have a GPU runtime enabled (Runtime -> Change runtime type).", icon="🚨")
                    st.stop()

                # 1. Split text into scenes/paragraphs
                scenes = [s.strip() for s in story_text.split('\n\n') if s.strip()]
                if not scenes:
                    st.warning("Could not split text into scenes. Please check formatting.")
                else:
                    video_clips = []
                    audio_files = []

                    for i, scene in enumerate(scenes):
                        st.info(f"Processing scene {i+1}/{len(scenes)}: {scene[:50]}...")

                        # Use Groq to generate image description for the scene
                        prompt_for_image = f"Describe a visual scene that would represent the following text for a video: '{scene}'. Be concise and vivid, focusing on key visual elements. For example, if the text is 'A knight rode through a dark forest', respond with 'A lone knight on horseback in a moonlit, dense forest, mist swirling.'"
                        image_description_response = client.chat.completions.create(
                            model="meta-llama/llama-4-scout-17b-16e-instruct", # Using your existing Groq model
                            messages=[
                                {"role": "system", "content": "You are a creative visual storyteller."},
                                {"role": "user", "content": prompt_for_image}
                            ],
                            max_tokens=100
                        )
                        image_description = image_description_response.choices[0].message.content
                        st.write(f"Scene {i+1} visual idea: {image_description}")

                        # Generate audio for the scene
                        try:
                            tts = gTTS(scene, lang='en')
                            audio_filename = f"scene_audio_{i}.mp3"
                            tts.save(audio_filename)
                            audio_files.append(audio_filename)

                            audio_clip = AudioFileClip(audio_filename)

                            # Create a text clip for the scene text
                            txt_clip = TextClip(scene, fontsize=24, color='white', bg_color='black', size=(1280, 720)).set_duration(audio_clip.duration)

                            # Generate image using Hugging Face model
                            st.info(f"Generating image for scene {i+1} with description: {image_description[:100]}...")
                            try:
                                generated_image = pipeline(prompt=image_description).images[0]
                                image_filename = f"scene_image_{i}.png"
                                generated_image.save(image_filename)
                                st.success(f"Image generated and saved as {image_filename}")

                                img_clip = ImageClip(image_filename).set_duration(audio_clip.duration)
                                # Resize image to fit video dimensions (e.g., 1280x720)
                                # The resize method in moviepy takes newsize=(width, height)
                                img_clip = img_clip.resize(newsize=(1280, 720))

                                # Composite the image clip and the text clip
                                final_clip = CompositeVideoClip([img_clip.set_position(('center', 'center')), txt_clip.set_position(('center', 'bottom'))], size=(1280, 720)).set_audio(audio_clip)
                                video_clips.append(final_clip)
                                os.remove(image_filename) # Clean up generated image
                            except Exception as e:
                                st.error(f"Error generating or integrating image for scene {i+1}: {e}. Falling back to text description image.")
                                # Fallback to TextClip if image generation fails
                                desc_text_clip = TextClip(image_description, fontsize=30, color='white', bg_color='darkblue', size=(1280, 720)).set_duration(audio_clip.duration)
                                final_clip = CompositeVideoClip([desc_text_clip.set_position(('center', 'center')), txt_clip.set_position(('center', 'bottom'))], size=(1280, 720)).set_audio(audio_clip)
                                video_clips.append(final_clip)
                            finally:
                                # Ensure audio files are cleaned up even if image generation fails
                                if os.path.exists(audio_filename):
                                    os.remove(audio_filename)


                        except Exception as e:
                            st.error(f"Error processing scene {i+1} (audio or general): {e}")
                            if os.path.exists(audio_filename):
                                os.remove(audio_filename)
                            continue

                    if video_clips:
                        st.info("Concatenating video clips and writing final video...")
                        final_video = concatenate_videoclips(video_clips, method="compose")
                        output_filename = "generated_story_video.mp4"
                        try:
                            final_video.write_videofile(output_filename, codec="libx264", audio_codec="aac", fps=24)
                            st.success("Video generated successfully!")
                            st.video(output_filename)

                            # Clean up final video file after display
                            os.remove(output_filename)
                        except Exception as e:
                            st.error(f"Error writing final video file: {e}. This might be due to ImageMagick policy issues. Please check the console for details.")
                    else:
                        st.error("No video clips were generated. Please check for errors in your story and ensure all models loaded correctly.")
elif selected_page == 'Voice to Text':
    st.set_page_config(
        page_title="Voice to Text with Lion AI",
        page_icon="🎤",
        layout="centered"
    )
    st.title("🎤 Lion AI Voice to Text")
    st.write("Upload an audio file to get its transcription.")

    uploaded_audio_file = st.file_uploader("Upload an audio file (e.g., .m4a, .mp3, .wav)", type=["m4a", "mp3", "wav"])

    if uploaded_audio_file is not None:
        st.audio(uploaded_audio_file, format=uploaded_audio_file.type) # Display the uploaded audio

        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing audio..."):
                try:
                    # Groq's transcription API expects a file-like object
                    # uploaded_audio_file is a BytesIO object by default
                    transcription = client.audio.transcriptions.create(
                      file=("uploaded_audio.m4a", uploaded_audio_file.getvalue(), uploaded_audio_file.type),
                      model="whisper-large-v3",
                      temperature=0,
                      response_format="verbose_json",
                    )
                    st.success("Transcription Complete!")
                    st.write("### Transcription:")
                    st.info(transcription.text)
                except Exception as e:
                    st.error(f"Error during transcription: {e}")
elif selected_page == 'Text to PDF': # New page for PDF conversion
    st.set_page_config(
        page_title="Text to PDF with Lion AI",
        page_icon="📄",
        layout="centered"
    )
    st.title("📄 Lion AI Text to PDF Converter")
    st.write("Enter your text below to convert it into a PDF file.")

    text_to_convert = st.text_area("Enter text here:", height=300)

    if st.button("Generate PDF"):
        if not text_to_convert:
            st.warning("Please enter some text to convert to PDF.")
        else:
            with st.spinner("Generating PDF..."):
                # PDF conversion logic using fpdf2
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                # Use multi_cell to handle line breaks automatically
                pdf.multi_cell(0, 10, text_to_convert)

                # Save PDF to a bytes buffer to allow download in Streamlit
                pdf_output = bytes(pdf.output(dest='S')) # Explicitly convert to bytes

                st.success("PDF generated successfully!")
                st.download_button(
                    label="Download PDF",
                    data=pdf_output,
                    file_name="generated_document.pdf",
                    mime="application/pdf"
                )
