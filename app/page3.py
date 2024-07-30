import streamlit as st

st.header('Contacts:telephone:')

with st.container(border= True):
    col1, col2, col3, col4 = st.columns(4)
    with col2:
        st.write("""
                    Claudia Sagini :sunglasses: \n
                    [Github Link](https://github.com/saginiclaudia) """)
    with col3:
        st.write("""
                    Ivy Atieng :goggles:  \n
                    [Github Link](https://github.com/Atieng)""")
    with col4:
        st.write("""
                    George Kariuki :goggles: \n
                    [GitHub Link](https://github.com/ggeorgekkariuki)""")
    with col1:
        st.write("""
                    Bradley Ouko :sunglasses:  \n
                    [GitHub Link](https://github.com/Misfit911)   """)
