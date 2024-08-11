import base64
from threading import Lock, Thread
import cv2
import os
import time
import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner.script_run_context import add_script_run_ctx, get_script_run_ctx
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class WebcamStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        self.frame = None
        if self.stream.isOpened():
            _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            if self.stream.isOpened():
                _, frame = self.stream.read()
                if frame is not None:
                    with self.lock:
                        self.frame = frame

    def read(self, encode=False):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        if frame is not None and encode:
            _, buffer = cv2.imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)
        self.image_count = 0

    def answer(self, image):
        prompt = "Detect real fire in the image but ignore if fire in photo or in video."
        image_base64 = ""
        if image:
            image_base64 = image.decode()
            self.image_count += 1

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image_base64},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        return response

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a fire detection assistant. Analyze the provided image to determine
        if there is a real fire present. Respond with a Fire Detected or NO Fire detected.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()
        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

webcam_stream = WebcamStream().start()
model = ChatGoogleGenerativeAI(google_api_key='AIzaSyAw1KjtzXP6Dj6C3800Rx9Gv8jpjOj32gE',model="gemini-1.5-flash-latest")
assistant = Assistant(model)

# Streamlit app

st.title("Fire Detection")

st.markdown("""
<style>
.st-emotion-cache-1jzia57{
    margin-left: -117px;
    margin-top: -107px;
}
.st-emotion-cache-h4xjwg {
    display:none;
}
.st-emotion-cache-1v0mbdj.e115fcil1{
    max-width: 161%;
    margin-left: -228px;
    margin-top: -47px;
}
#detection-result{
    margin-right:-55;
}
#detection-result{
    margin-right: -135px;
    padding-left: 146px;
    margin-left: 0px;
    margin-top: -18px;
    padding-top: 0px;
}
.st-emotion-cache-y4bq5x{
    width: 446px;
}
.st-emotion-cache-183lzff{
    margin-left: 32px;
    font-size: 1.5rem;
    font-weight: bolder;
    margin-right: -267px;
}
.row-widget.stButton{
    width: 762px;
    margin-top: -262px;  
    margin-left: 81px;  
}
.st-emotion-cache-15hul6a{
    margin-left: 496px;
}
.st-emotion-cache-15hul6a.ef3psqc12 {
    margin-top: 20px;
}
.st-emotion-cache-bcargt{
    display:none;
}
.st-emotion-cache-13n2bn5 {
    margin-left: -214px;
    width: 150%;
    hight: 150%;
}
.st-emotion-cache-1qg05tj.e1y5xkzn3{
    display: none;
}
.st-emotion-cache-1l6h8p9{
    width: 476px;
    position: relative;
    left: -23px;
}
</style>
""", unsafe_allow_html=True)

# Start and stop detection buttons
if 'detection_running' not in st.session_state:
    st.session_state.detection_running = False

col1, col2 = st.columns([3, 1])

with col1:
    cam_placeholder = st.empty()
with col2:
    st.header("Detection Result")
    result_placeholder = st.empty()

def start_detection():
    st.session_state.detection_running = True

def stop_detection():
    st.session_state.detection_running = False

st.button("Start Detection", on_click=start_detection)
st.header("        ")
st.header("        ")
st.button("Stop Detection", on_click=stop_detection)

def capture_and_display():
    ctx = get_script_run_ctx()
    add_script_run_ctx(ctx)
    try:
        while True:
            frame = webcam_stream.read()
            if frame is not None:
                cam_placeholder.image(frame, channels="BGR", use_column_width=True)
            time.sleep(0.003)  # Adjust sleep time as necessary

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        webcam_stream.stop()
        cv2.destroyAllWindows()

def detect_fire():
    ctx = get_script_run_ctx()
    add_script_run_ctx(ctx)
    try:
        while True:
            if st.session_state.detection_running:
                encoded_frame = webcam_stream.read(encode=True)
                if encoded_frame is not None:
                    response = assistant.answer(encoded_frame)
                    result_placeholder.text(f"Response: {response}")

                    if response == "Fire Detected":
                        st.warning("Fire detected!")

            time.sleep(3)  # Send frames to assistant every 3 seconds

    except Exception as e:
        st.error(f"An error occurred: {e}")

capture_thread = Thread(target=capture_and_display)
detection_thread = Thread(target=detect_fire)

add_script_run_ctx(capture_thread)
add_script_run_ctx(detection_thread)

capture_thread.start()
detection_thread.start()

capture_thread.join()
detection_thread.join()
