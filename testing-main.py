# import streamlit as st
# import time  # Simulating a loading process

# def load_vector_database():
#     # Simulate a loading process (e.g., loading a vector database)
#     time.sleep(5)  # Replace this with your actual loading logic

# # Main Streamlit app
# def main():
#     st.title("Vector Database Loader")

#     with st.spinner("Loading the Vector Database..."):
#         load_vector_database()  # Call your loading function here

#     st.success("Vector Database Loaded Successfully!")

# if __name__ == "__main__":
#     main()


import streamlit as st
import time
import keyboard
import os
import psutil

exit_app = st.sidebar.button("Shut Down")

if exit_app:
    time.sleep(2)  # Delay for user experience
    keyboard.press_and_release('ctrl+w')  # Close the browser tab
    pid = os.getpid()  # Get current process ID
    p = psutil.Process(pid)  # Create a process object
    p.terminate()  # Terminate the process