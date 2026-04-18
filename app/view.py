import streamlit as st
st.title('Dr. Akinator')

# initialize
st.session_state.current_question = None

def showQuestion():
    if st.session_state.current_question is None:
        st.text('No question yet (question here)')
    else:
        st.text(st.session_state.current_question)


def answer(id:int):
    print(id)


showQuestion()

c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])

with c1:
    if st.button('Yes', width='stretch'): answer(4)
with c2:
    if st.button('Probably', width='stretch'): answer(3)
with c3:
    if st.button('Maybe', width='stretch'): answer(2)
with c4:
    if st.button('Probably Not', width='stretch'): answer(1)
with c5:
    if st.button('No', width='stretch'): answer(0)
