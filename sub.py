import subprocess
import streamlit as st
def start_app(app_name, port):
    # Start the Streamlit app on a specific port
    command = ["streamlit", "run", app_name, "--server.port", str(port)]
    subprocess.Popen(command)
def start_app2(app_name, port):
    # Start the Streamlit app on a specific port
    command = ["flask", "--app", app_name ,"run"]
    subprocess.Popen(command)
     

start_app("streamlit_app.py", 8502)
start_app("main.py", 8503)
start_app2("app.py",8504)
start_app("recipe_recommend.py",8505)