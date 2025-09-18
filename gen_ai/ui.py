import streamlit as st
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor

# Consolidate all imports from TruthLens into a single try-except block
try:
    from TruthLens import (
        TruthLensDatabase,
        TruthLensEmbeddings,
        TruthLensAnalytics,
        configure_genai,
        enhanced_analyze_statement,
        fetch_fact_check_references,
        analyze_image,
        get_truthlens_insights,
    )
except ImportError as e:
    raise ImportError(
        "Required classes/functions are missing from TruthLens. "
        "Please ensure TruthLens is installed and up to date."
    ) from e

def render_css():
    """Renders the custom CSS for the application."""
    st.markdown("""
    <style>
        :root {
            --primary: #4f46e5;
            --primary-light: #818cf8;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --text: #1f2937;
            --text-light: #6b7280;
            --bg: #f9fafb;
            --card-bg: #ffffff;
            --border: #e5e7eb;
        }
        
        body { font-family: 'Inter', system-ui, sans-serif; }
        
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea {
            border-radius: 8px !important;
            border: 2px solid var(--border) !important;
            padding: 0.75rem 1rem !important;
        }
        
        .stButton>button {
            background: var(--primary);
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background: var(--primary-light);
        }
        
        .stExpander {
            border-color: var(--border) !important;
        }
        
        .result-card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: all 0.2s ease-in-out;
            border: 1px solid var(--border);
        }
        
        .status-true { color: var(--success); }
        .status-false { color: var(--danger); }
        .status-misleading { color: var(--warning); }

        .fact-check-card {
            background-color: #f8f9fa;
            border-left: 5px solid var(--primary-light);
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
        }
        .fact-check-publisher {
            font-weight: 600;
            color: var(--text);
        }
        .fact-check-rating {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 500;
            font-size: 0.85rem;
            margin-left: 0.5rem;
        }
        .rating-false, .rating-inaccurate, .rating-distorted { background-color: #fee2e2; color: #991b1b; }
        .rating-true, .rating-accurate { background-color: #dcfce7; color: #166534; }
        .rating-misleading, .rating-partially-true, .rating-unproven { background-color: #ffedd5; color: #9a3412; }

        .section-container {
            background-color: var(--card-bg);
            padding: 2rem;
            border-radius: 12px;
            margin-top: 2rem;
            border: 1px solid var(--border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.5rem;
            color: var(--primary);
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Renders the main header of the application."""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 3rem; font-weight: 700; margin: 0; color: var(--primary);">🔍 TruthLens</h1>
        <p style="color: var(--text-light); margin: 0.5rem 0 0; font-size: 1.1rem;">
            AI-Powered Fact Checking
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Renders the footer of the application."""
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 1.5rem 0; color: var(--text-light); border-top: 1px solid var(--border);">
        <p style="margin: 0; font-size: 0.9rem;">© 2023 TruthLens. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
    
def initialize_app():
    """Sets page config and renders initial CSS and scripts."""
    st.set_page_config( 
        page_title="TruthLens", 
        page_icon="🔍", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    render_css()

    # Add Lenis for smooth scrolling
    st.markdown("""
    <script src="https://unpkg.com/@studio-freight/lenis@1.0.42/dist/lenis.min.js"></script>
    <script>
        const lenis = new Lenis()

        function raf(time) {
          lenis.raf(time)
          requestAnimationFrame(raf)
        }

        requestAnimationFrame(raf)
        
        // Stop Lenis on Streamlit script rerun/widget interaction
        const observer = new MutationObserver((mutations) => {
            lenis.stop();
        });
        observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)

def setup_session_state():
    """Handles API key configuration and initializes session state."""
    try:
        api_key = st.secrets["gemini"]["api_key"]
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            st.error("""
            ## ❌ Configuration Required
            **Gemini API key not configured!** 
            To use TruthLens, you need to add your Google Gemini API key to the configuration file.
            **Steps to fix:**
            1. Create a file called `.streamlit/secrets.toml` in your project directory.
            2. Add your API key in this format:
            ```toml
            [gemini]
            api_key = "your_actual_api_key_here"
            ```
            3. Get your API key from: https://makersuite.google.com/app/apikey
            **Without this key, the AI analysis features will not work.**
            """)
            st.stop()

        factcheck_api_key = st.secrets.get("google", {}).get("factcheck_api_key")
        if not factcheck_api_key:
            st.warning(
                "**Fact Check API key not configured.**\n"
                "Fact-checking features will be disabled, but you can still use AI analysis.\n"
                "To enable fact-checking, add your Fact Check API key to `.streamlit/secrets.toml`:\n"
                "```\n"
                "[google]\n"
                "factcheck_api_key = \"your_factcheck_api_key_here\"\n"
                "```"
            )

        st.session_state.factcheck_api_key = factcheck_api_key
        # Initialize models and database in session state if not already present
        if 'model' not in st.session_state:
            st.session_state.model = configure_genai(api_key)
        if 'truthlens_db' not in st.session_state:
            st.session_state.truthlens_db = TruthLensDatabase()
        if 'truthlens_embeddings' not in st.session_state:
            st.session_state.truthlens_embeddings = TruthLensEmbeddings()
        if 'truthlens_analytics' not in st.session_state:
            st.session_state.truthlens_analytics = TruthLensAnalytics()
    except (KeyError, AttributeError):
        st.error("❌ Configuration error. Please check your `.streamlit/secrets.toml` file.")
        st.code("""
    [gemini]
    api_key = "your_gemini_api_key_here"

    [google]
    factcheck_api_key = "your_factcheck_api_key_here"
    """)
        st.stop()

def render_main_ui():
    """Renders the main UI components like title and input area."""
    # Show loading status for models
    if st.session_state.get('truthlens_embeddings') is None:
        st.info("🔄 AI models are loading in the background... The app is ready to use!")

    render_header()

    st.info("""
    **Welcome to TruthLens!** This tool helps you analyze claims, articles, and images to detect potential misinformation.
    Use the sections below to check a statement, analyze an image, or chat with our AI assistant.
    """)

    with st.container(border=True):
        st.markdown('<div class="section-header"><span>📝</span><h3>Analyze a Claim or Article</h3></div>', unsafe_allow_html=True)
        user_input = st.text_area(
            "Enter your claim or statement",
            key="claim_input",
            placeholder="Paste an article link or type a claim here...",
            height=120,
            help="Enter any claim, statement, or news headline you'd like to analyze for misinformation",
            label_visibility="collapsed"
        )
        handle_text_analysis(user_input)

def handle_text_analysis(user_input):
    """Handles the logic for text analysis when the button is clicked."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Check Truth", type="primary", use_container_width=True):
            st.session_state.analyze_clicked = True
            st.session_state.feedback_submitted = False # Reset feedback state

    st.markdown("""
    <div style="text-align: center; margin: 1rem 0; color: #6c757d; font-size: 0.9rem;">
        <em>Powered by Google Generative AI & Fact-Check APIs</em>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get('analyze_clicked', False):
        if not user_input.strip():
            st.session_state.analyze_clicked = False # Reset on empty input
            st.warning("Please enter a statement to check.")
        else:
            with st.spinner("Analyzing... This may take a moment."):
                with ThreadPoolExecutor() as executor:
                    # Use enhanced analysis with database storage
                    gemini_future = executor.submit(
                        enhanced_analyze_statement,
                        st.session_state.model,
                        user_input,
                        st.session_state.truthlens_db,
                        st.session_state.truthlens_embeddings
                    )
                    fact_check_api_key = st.session_state.get('factcheck_api_key')
                    fact_check_api_key_safe = fact_check_api_key if fact_check_api_key is not None else ""
                    fact_check_future = executor.submit(fetch_fact_check_references, fact_check_api_key_safe, user_input)

                    ai_analysis = gemini_future.result()
                    fact_response = fact_check_future.result()

                if ai_analysis:
                    classification = ai_analysis.get("classification", "N/A")
                    confidence = ai_analysis.get("confidence_score", 0)
                    explanation = ai_analysis.get("explanation", "No explanation provided.")

                    st.subheader("AI Analysis Results")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)

                    if "True" in classification:
                        status_icon = "✅"
                        status_class = "status-true"
                    elif "False" in classification or "Misleading" in classification:
                        status_icon = "❌"
                        status_class = "status-false"
                    else:
                        status_icon = "⚠️"
                        status_class = "status-misleading"

                    st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 2rem;">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">{status_icon}</div>
                        <h2 class="{status_class}">{classification}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="margin: 2rem 0;">
                        <h4 style="color: #2c3e50; margin-bottom: 1rem;">📊 Credibility Score: {confidence}%</h4>
                        <div class="credibility-score" style="background: #e9ecef; border-radius: 10px; padding: 4px;">
                            <div class="score-fill" style="width: {confidence}%; background: var(--primary-gradient); height: 20px; border-radius: 6px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="margin: 2rem 0;">
                        <h4 style="color: #2c3e50; margin-bottom: 1rem;">💡 AI Explanation</h4>
                        <p style="font-size: 1.1rem; line-height: 1.6; color: #495057;">{explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("AI analysis failed. Please try again later.")

                st.subheader("🔗 Fact-Check Sources")
                if not st.session_state.get('factcheck_api_key'):
                    st.info("Fact-checking is disabled. Please configure your Fact Check API key to enable this feature.")
                elif fact_response and "claims" in fact_response and fact_response["claims"]:
                    st.info(f"Found {len(fact_response['claims'])} related fact-check articles:")
                    for i, claim in enumerate(fact_response["claims"]):
                        claim_text = claim.get('text', 'No claim text')
                        with st.expander(f"📄 **Reference {i+1}:** {claim_text[:80]}...", expanded=i < 2):
                            if "claimReview" in claim and claim["claimReview"]:
                                review = claim["claimReview"][0]
                                rating = review.get('textualRating', 'N/A')
                                rating_class = f"rating-{rating.lower().replace(' ', '-')}"

                                st.markdown(f"""
                                <div class="fact-check-card">
                                    <div>
                                        <span class="fact-check-publisher">📰 {review.get('publisher', {}).get('name', 'N/A')}</span>
                                        <span class="fact-check-rating {rating_class}">{rating}</span>
                                    </div>
                                    <p style="margin: 0.75rem 0;"><strong>Claim:</strong> "{claim_text}"</p>
                                    <p style="margin: 0.5rem 0;"><strong>Article:</strong> <a href="{review.get('url', '#')}" target="_blank">{review.get('title', 'Read more...')}</a></p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.write("📋 Detailed review not available for this claim.")
                else:
                    st.info("No matching articles found in the fact-check database.")

                st.subheader("Was this analysis helpful?")
                if 'feedback_submitted' not in st.session_state:
                    st.session_state.feedback_submitted = False

                if not st.session_state.feedback_submitted:
                    col1, col2, col3 = st.columns([1, 1, 5])
                    if col1.button("👍 Yes", key="feedback_yes"):
                        st.session_state.feedback_submitted = True
                        st.toast("Thank you for your feedback!", icon="✅")
                    if col2.button("👎 No", key="feedback_no"):
                        st.session_state.feedback_submitted = True
                        st.toast("Thank you for your feedback!", icon="✅")

def render_image_analysis_section():
    """Renders the UI and logic for the image analysis feature."""
    with st.container(border=True):
        st.markdown('<div class="section-header"><span>🖼️</span><h3>Analyze an Image</h3></div>', unsafe_allow_html=True)
        st.markdown("Upload an image to detect manipulation, misleading visuals, and fake content using AI vision technology.")
        
        uploaded_file = st.file_uploader(
            "Upload an image to check for misinformation",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            help="Supported formats: PNG, JPG, JPEG, GIF, WEBP",
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        st.markdown("### 📸 Uploaded Image")
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension not in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
                st.error("❌ Unsupported file type. Please upload a PNG, JPG, JPEG, GIF, or WEBP file.")
                st.stop()

            image_bytes = uploaded_file.getvalue()
            if len(image_bytes) > 10 * 1024 * 1024:
                st.error("❌ Image file too large. Please upload an image smaller than 10MB.")
                st.stop()

            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.stop()

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                progress_bar = st.progress(0, "Starting image analysis...")
                with st.spinner("Analyzing image..."):
                    image_analysis_result = analyze_image(st.session_state.model, image_bytes)
                    progress_bar.progress(100, "Analysis complete!")

                st.markdown("### 🔍 Image Analysis Results")
                if image_analysis_result:
                    st.markdown(f"""
                    <div style="background: #d1ecf1; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #17a2b8; margin: 1rem 0;">
                        <h4 style="color: #0c5460; margin-top: 0;">📋 Analysis Report</h4>
                        <p style="margin-bottom: 0;">{image_analysis_result}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("💡 **Pro Tip:** For additional verification, try a reverse image search using Google Images or TinEye to find where else this image has been used.")
                else:
                    st.error("Image analysis failed. Please try again or upload a different image.")

def render_chatbot_section():
    """Renders the UI and logic for the interactive chatbot."""
    with st.container(border=True):
        st.markdown('<div class="section-header"><span>💬</span><h3>Chat with TruthLens AI</h3></div>', unsafe_allow_html=True)
        st.markdown('Ask me about any claim, conspiracy theory, or suspicious information. I\'ll help you understand the facts!')

    with st.expander("❓ Example Questions"):
        st.markdown("""
        - Is it true that drinking coffee stunts your growth?
        - What's the real story behind the latest viral video?
        """)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role_class = "chat-user" if message["role"] == "user" else "chat-assistant"
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message {role_class}"><strong>{"You" if message["role"] == "user" else "AI"}:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Enhanced AI message display with better visibility
            st.markdown(f'''
            <div class="chat-message {role_class}">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.2rem; margin-right: 0.5rem;">🤖</span>
                    <strong>TruthLens AI:</strong>
                </div>
                <div style="font-size: 1.1rem; line-height: 1.6;">
                    {message["content"]}
                </div>
            </div>
            ''', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask me anything about misinformation, facts, or claims...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🤔 Thinking..."):
            try:
                if st.session_state.model is None:
                    st.error("AI model is not initialized. Please check your API key configuration.")
                    return
                chatbot_prompt = f'You are a helpful AI assistant that debunks misinformation. The user asked: "{prompt}". Provide a clear, factual, and concise response.'
                response = st.session_state.model.generate_content(chatbot_prompt)
                ai_response = response.text
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                error_message = f"Sorry, I'm having trouble responding right now. Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.rerun()

    if st.session_state.messages:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

def render_feedback_section():
    """Renders the community feedback submission form."""
    with st.container(border=True):
        st.markdown('<div class="section-header"><span>📝</span><h3>Community Feedback</h3></div>', unsafe_allow_html=True)
        st.markdown('Help us improve by reporting suspicious content you\'ve encountered.')

    with st.form("simple_feedback_form"):
        flag_description = st.text_area(
            "Flag suspicious content",
            placeholder="Describe the suspicious content you encountered...",
            height=100,
            help="Please provide as much detail as possible to help us understand the issue",
        )
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button("🚩 Submit Report", type="primary", use_container_width=True)
        if submitted:
            if not flag_description.strip():
                st.error("❌ Please provide a description of the suspicious content.")
            else:
                st.success("✅ Thank you for helping fight misinformation!")
                st.info("**Note:** This is a demo. In a production app, this data would be stored securely and reviewed by our team.")
                with st.expander("📋 Report Details (Demo)", expanded=False):
                    st.json({
                        "description": flag_description,
                        "timestamp": "2024-01-01T00:00:00Z",
                        "status": "Submitted"
                    })

def render_analytics_section():
    """Renders the analytics dashboard section."""
    with st.container(border=True):
        st.markdown('<div class="section-header"><span>📊</span><h3>Analytics Dashboard</h3></div>', unsafe_allow_html=True)
        st.markdown("View insights and statistics about the claims analyzed by TruthLens.")
        try:
            insights = get_truthlens_insights()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📋 Statistics")
                stats = insights['statistics']
                st.metric("Total Claims Analyzed", stats['total_claims'])
                
                if stats['confidence_stats']:
                    st.metric("Average Confidence", f"{stats['confidence_stats']['mean']:.1f}%")
                
                if stats['accuracy_distribution']:
                    st.markdown("#### Classification Distribution")
                    for classification, count in stats['accuracy_distribution'].items():
                        st.write(f"**{classification}**: {count}")

            with col2:
                st.markdown("### 🎯 Confidence Analysis")
                conf_analysis = insights['confidence_analysis']
                if conf_analysis:
                    st.metric("Mean Confidence", f"{conf_analysis['mean_confidence']:.1f}%")
                    st.metric("High Confidence Claims", conf_analysis['high_confidence_claims'])
                    st.metric("Low Confidence Claims", conf_analysis['low_confidence_claims'])

            if insights['recommendations']:
                st.markdown("### 💡 Recommendations")
                for recommendation in insights['recommendations']:
                    st.info(f"💡 {recommendation}")

        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")

def render_similar_claims_section():
    """Renders the similar claims search section."""
    with st.container(border=True):
        st.markdown('<div class="section-header"><span>🔍</span><h3>Similar Claims Search</h3></div>', unsafe_allow_html=True)
        st.markdown("Search our database for previously analyzed claims that are similar to your query.")
        search_query = st.text_input("Search for similar claims in our database", placeholder="Enter a claim to find similar ones...", label_visibility="collapsed")
    if st.button("🔍 Search Similar Claims", use_container_width=True) and search_query:
        try:
            similar_claims = st.session_state.truthlens_db.get_similar_claims(search_query)
            if similar_claims:
                st.success(f"Found {len(similar_claims)} similar claims!")
                for i, claim in enumerate(similar_claims):
                    with st.expander(f"Claim {i+1}: {claim['claim_text'][:100]}...", expanded=False):
                        st.write(f"**Classification**: {claim['classification']}")
                        st.write(f"**Confidence**: {claim['confidence_score']}%")
                        st.write(f"**Explanation**: {claim['explanation']}")
                        st.write(f"**Date**: {claim['timestamp']}")
            else:
                st.info("No similar claims found in the database.")
                
        except Exception as e:
            st.error(f"Error searching similar claims: {str(e)}")

def main():
    """Main function to run the Streamlit application."""
    initialize_app()
    setup_session_state()
    render_main_ui()
    render_image_analysis_section()
    render_chatbot_section()
    render_feedback_section()
    render_analytics_section()
    render_similar_claims_section()

    render_footer()

if __name__ == "__main__":
    main()