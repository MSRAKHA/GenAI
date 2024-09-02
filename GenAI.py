import streamlit as st
from streamlit import set_page_config
import time
from dotenv import load_dotenv
import requests
import io
from PIL import Image
import os
import google.generativeai as genai
import PIL
import numpy as np
import bcrypt
st.set_page_config(page_title="Login", page_icon=":tada:", layout="wide")
users={
    "user1" :{
        "username" : "user1",
        "password" : bcrypt.hashpw("password1".encode('utf-8'), bcrypt.gensalt())
    },
    "user2" :{
        "username" : "user2",
        "password" : bcrypt.hashpw("password2".encode('utf-8'), bcrypt.gensalt())
    }
}

#check if user is logged in
if "logged_in" not in st.session_state:
  st.session_state["logged_in"] = False
if not st.session_state["logged_in"]:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username]["password"]):
            st.session_state["logged_in"] = True
            st.success("Logged in successfully")
            st.rerun()
        else:
            st.error("Invalid username or password")
if st.session_state["logged_in"]:
    st.sidebar.title("GenAI App")
    #st.sidebar.title("This is a simple GenAI app using Streamlit and Google's Gemini Pro model")
    st.sidebar.title("Developer - Rakha Shaik")
    st.sidebar.markdown("-----")

    #create a sidebar for navigation
    options =["Get Started" ,"Text2Image","Image2Caption","Text Generation","Image Analysis" , "Video Insights", "ChatBot"]
    option = st.sidebar.selectbox("Go to", options)
    #Logout Button
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] =None
        st.success("Logged out successfully")
        st.rerun()
    # create a function to displap welcome page
    def display_welcome_page():
        st.title("Welcome to GenAI App LLM-Powered Insights")
        st.write("This is a simple GenAI app using Streamlit and Google's Gemini Pro model")
        st.header("Features")
        st.write("- Text2Image: Generate images from text descriptions")
        st.write("- Image2Caption: Generate captions for images")
        st.write("- Text Generation: Generate text based on a prompt")
        st.write("- Image Analysis: Analyze images and extract relevant information")
        st.write("- Video Insights: Extract insights from videos")
        st.write("- ChatBot: Engage in a conversation with the app")
        st.header("Developer - Rakha Shaik")
    def display_Text2Image():
        API_URL ="https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
        headers = {"Authorization": "Bearer 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"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            content_type = response.headers.get('Content-Type')

            if "image" in content_type:
                return response.content
            else:
                st.error(f"received non-image response  : {response.content}")
                return None
        st.title("Text2Image Generator")
        user_input = st.text_input("Enter a prompt :", "Astronaut riding a horse")
        if st.button("Generate Image"):
            with st.spinner("Generating Image..."):
                response_content = query({"inputs": user_input,
                                          })
                if response_content:
                    try:
                        image = Image.open(io.BytesIO(response_content))
                        caption=f"Generated Image for '{user_input}'"
                        st.image(image, caption=caption, use_column_width=True)
                        st.success("Image generated successfully!")
                        img_buffer = io.BytesIO()
                        image.save(img_buffer, format="PNG")
                        img_buffer.seek(0)
                        st.download_button(
                                label="Download Image",
                                data=img_buffer,
                                file_name=f"{caption}.png",
                                mime="image/png"

                            )
                    except Exception as e:
                        st.error(f"Error generating image : {e}")
    def display_Image2Caption():
        os.environ['GOOGLE_API_KEY'] = "AIzaSyCu5LyOa_tUH1suyjJG-ToTFRsOA9sakbo"
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        def get_image_caption(image):
            vision_model = genai.GenerativeModel('gemini-1.5-flash')
            response = vision_model.generate_content(["Generate a good caption for this image ",image])
            return response.text
        
           
        st.title("Image Caption Generator")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            with st.spinner("Generating Caption..."):
                caption = get_image_caption(image)
                st.success(f"Caption: {caption}")
                st.header("Image Caption Generator")
                st.info(caption)
    def display_TextGeneration():
        os.environ['GOOGLE_API_KEY'] = "AIzaSYBrCihLeiptofBDzgIdG8U6ypyo8aBFWew"
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        st.title("Text Generation")

        def generate_text(prompt):
            text_model = genai.GenerativeModel("gemini-pro")
            response = text_model.generate_content(prompt)
            return response.text
        
        user_prompt = st.text_input("Enter a prompt:")
        with st.spinner("Generating Text..."):
            
            if st.button("Generate Text"):
                result = generate_text(user_prompt)
                st.write(result)
    def display_ImageAnalysis():
        os.environ['GOOGLE_API_KEY'] = "AIzaSYBrCihLeiptofBDzgIdG8U6ypyo8aBFWew"
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        

        st.title("Image Analysis ")
        def analyze_image(image,prompt):
            vision_model = genai.GenerativeModel('gemini-1.5-flash')
            response = vision_model.generate_content([prompt,image])
            return response.text
        prompt = st.text_input("Enter a prompt for image analysis:")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            #st.image(image, caption="Uploaded Image", use_column_width=True)
            with st.spinner("Analyzing Image..."):
                if st.button("Analyze Image"):
                    result = analyze_image(image, prompt)
                    st.write(result)
    MEDIA_FOLDER ='medias'
    my_api_key = "AIzaSYBrCihLeiptofBDzgIdG8U6ypyo8aBFWew"
    

    def display_videoGeneration():
        if not os.path.exists(MEDIA_FOLDER):
            os.makedirs(MEDIA_FOLDER)
        load_dotenv()
        genai.configure(api_key=my_api_key)
    
    def save_uploaded_file(uploaded_file):
        """save the uploaded file to to the media folder and return the file path."""
        file_path = os.path.join(MEDIA_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        return file_path
    def get_insights(video_path):
        """Extra insights from thr video using gemini flash"""
        st.write(f"Processing Video: {video_path}")
        st.write("This may take a while...")
        st.write("uploading file....")
        video_file = genai.upload_file(path = video_path)
        st.write("Completeyl Uploaded:{video_file_url}") 

        while video_file.status.name == "PROCESSING":
            st.write('waiting for video to be processed')
            time.sleep(10)
            video_file = genai.get_file(video_file.name)
        if video_file.state.name =="FAILED":
            raise ValueError(video_file.status.name)
        prompt = "Generate a detailed report on the video"
        model = genai.GenerativeModel("gemini-1.5-flash")
        st.write("Generating Insights....")
        response = model.generate_content([prompt,video_file],requests_options={"timeout": 600})
        st.write("Video Processing Complete")
        st.subheader("Video Insights")
        st.write(response.text)
        genai.delete_file(video_file.name)
    def app():
        st.title("Video Insights Generator")
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            file_path = save_uploaded_file(uploaded_file)
            st.video(file_path)
            if st.button("Analyze Video"):
                get_insights(file_path)
            if os.path.exists(file_path):
                os.remove(file_path)
                st.warning("File deleted successfully")
    def display_chatBot():
        os.environ['GOOGLE_API_KEY'] = "AIzaSYBrCihLeiptofBDzgIdG8U6ypyo8aBFWew"
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        st.title("ChatBot")
        model = genai.GenerativeModel("gemini-pro")
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Ask me Anything."}
            ]
        for messages in st.session_state.messages:
            with st.chat_message(messages["role"]):
                st.markdown(messages["content"])
        def llm_function(query):
            response = model.generate_content(query)
            with st.chat_message("assistant"):
                st.markdown(response.text)
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        query = st.chat_input("Ask me Anything", key="user_input")
        if query:
          with st.chat_message("user"):
            st.markdown(query)
          llm_function(query)




    if option == "Get Started":
        display_welcome_page()
    elif option == "Text2Image":
        display_Text2Image()
    elif option == "Image2Caption":
        display_Image2Caption()
    elif option == "Text Generation":
        display_TextGeneration()
    elif option == "Image Analysis":
        display_ImageAnalysis()
    elif option == "Video Insights":
        app()
    elif option == "ChatBot":
        display_chatBot()
      
