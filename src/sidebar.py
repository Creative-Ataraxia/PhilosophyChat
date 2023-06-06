import streamlit as st

def record_feedback(feedback: str):
    st.session_state.feedback.append(feedback)
    print(feedback)
    
def sidebar():
    if "feedback" not in st.session_state:
        st.session_state.feedback = ""
            
    with st.sidebar:

        st.subheader("Chat with over 250,000 pages of modern & historical philosophy texts")
        st.subheader("与超过25万页古典和现代哲学典籍对话")
        st.subheader("25万ページ以上の現代・歴史哲学のテキストを使ったチャット")
        st.divider()
        st.markdown(
            "## How to Use\n"
            "1. Submit a philosophical inquiry through the text box💬\n" 
            "2. Click the button underneath for some new ideas💡\n"
            "3. Chat with over 250,000 pages of philosophy texts🧙\n"
        )

        # TODO: connect this to notion
        quick_feedback = st.text_input(
            "What else should this app do?",
            placeholder="type any additional features you'd like here...",
            help="Thanks for your feedback!",
        )

        if quick_feedback:
            record_feedback(quick_feedback)

        st.divider()
        st.markdown("# About")
        st.markdown(
            "This conversational AI is embedded with over 250,000 pages of modern & traditional philosophical texts. "
            "Given the right prompts, this Chatbot should be able to engage in in-depth philosophical discussions."
        )
        # TODO: update repo address
        st.markdown(
            "This tool is a work in progress. "
            "Contributions are welcome on [GitHub](https://github.com/Creative-Ataraxia/PhilosophyChat). "
            "Bug reports and feature requests can also be submitted through Github."
        )
        st.divider()
        st.markdown("Made with :heart: by [Creative_Ataraxia](<https://github.com/Creative-Ataraxia>)")
        