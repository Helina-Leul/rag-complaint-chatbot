import streamlit as st

from src.rag import ask_rag

st.title("CrediTrust Complaint Chatbot")

question = st.text_input("Ask a question")

if st.button("Ask"):

    answer, sources = ask_rag(question)

    st.subheader("Answer")

    st.write(answer)

    st.subheader("Sources")

    for source in sources:

        st.write(source["metadata"])

        st.write(source["text"][:400])

        st.divider()
