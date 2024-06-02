import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Define cool_header function
def cool_header():
# Define the display_chat function
    def display_chat(chats):
        for chat in chats:
            st.write(chat)



# Import remaining modules
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit.components.v1 import html
#from st_on_hover_tabs import on_hover_tabs
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import streamlit_analytics
import base64
from streamlit_extras.mention import mention
from streamlit_extras.app_logo import add_logo
import sqlite3
#from bs4 import BeautifulSoup
from streamlit_extras.echo_expander import echo_expander

# Set page config
st.set_page_config(page_title="Aniket Maheshwari", page_icon="desktop_computer", layout="wide", initial_sidebar_state="auto")

# Load local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")


# Use the following line to include your style.css file
st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def render_lottie(url, width, height):
    lottie_html = f"""
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.7.14/lottie.min.js"></script>
    </head>
    <body>
        <div id="lottie-container" style="width: {width}; height: {height};"></div>
        <script>
            var animation = lottie.loadAnimation({{
                container: document.getElementById('lottie-container'),
                renderer: 'svg',
                loop: true,
                autoplay: true,
                path: '{url}'
            }});
            animation.setRendererSettings({{
                preserveAspectRatio: 'xMidYMid slice',
                clearCanvas: true,
                progressiveLoad: false,
                hideOnTransparent: true
            }});
        </script>
    </body>
    </html>
    """
    return lottie_html

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

footer = """
footer{
    visibility:visible;
}
footer:after{
    content:'Copyright © 2023 Aniket';
    position:relative;
    color:black;
}
"""
# PDF functions
def show_pdf(file_path):
        with open(file_path,"rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

def pdf_link(pdf_url, link_text="Click here to view PDF"):
    href = f'<a href="{pdf_url}" target="_blank">{link_text}</a>'
    return href

# Load assets
#lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
# Assets for about me
img_utown = Image.open("images/a1.JPG")
img_lh = Image.open("images/po.jpg")
img_ifg = Image.open("images/image_6483441.jpg")

# Assets for education
img_pondicherry = Image.open("images/a5.jpg")
img_iimt = Image.open("images/a6.jpg")
img_cityconvent = Image.open("images/a7.jpg")
# Assets for experiences

img_bharatintern = Image.open("images/bharat_intern_logo.jpg")
img_techietripper = Image.open("images/techietripper.png")
# Assets for projects
image_names_projects = ["cv", "dr1", "c19", "health", 
                         "biopics", "anime", "word2vec", "cellphone", 
                         "spotify", "map", "gephi", "fob", "get", "ty",
                         "qw"]
images_projects = [Image.open(f"images/{name}.{'jpg' if name not in ('map', 'gephi', 'health') else 'png'}") for name in image_names_projects]


# Assets for blog
img_qb = Image.open("images/qb.jpg")
# Assets for contact
lottie_coding = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_abqysclq.json")

img_linkedin = Image.open("images/linkedin.png")
img_github = Image.open("images/github.png")
img_email = Image.open("images/email.png")

def social_icons(width=24, height=24, **kwargs):
        icon_template = '''
        <a href="{url}" target="_blank" style="margin-right: 20px;">
            <img src="{icon_src}" alt="{alt_text}" width="{width}" height="{height}">
        </a>
        '''

        icons_html = ""
        for name, url in kwargs.items():
            icon_src = {
                "youtube": "https://img.icons8.com/ios-filled/100/ff8c00/youtube-play.png",
                "linkedin": "https://img.icons8.com/ios-filled/100/ff8c00/linkedin.png",
                "github": "https://img.icons8.com/ios-filled/100/ff8c00/github--v2.png",
                "wordpress": "https://img.icons8.com/ios-filled/100/ff8c00/wordpress--v1.png",
                "email": "https://img.icons8.com/ios-filled/100/ff8c00/filled-message.png"
            }.get(name.lower())

            if icon_src:
                icons_html += icon_template.format(url=url, icon_src=icon_src, alt_text=name.capitalize(), width=width, height=height)

        return icons_html
#####################
# Custom function for printing text
def txt(a, b):
  col1, col2 = st.columns([4,1])
  with col1:
    st.markdown(a)
  with col2:
    st.markdown(b)

def txt2(a, b):
  col1, col2 = st.columns([1,4])
  with col1:
    st.markdown(f'`{a}`')
  with col2:
    st.markdown(b)

#def txt3(a, b):
  #col1, col2 = st.columns([1,2])
  #with col1:
    #st.markdown(f'<p style="font-size: 20px;">{a}</p>', unsafe_allow_html=True)
  #with col2:
    # Split the text at the comma and wrap each part in backticks separately
    #b_parts = b.split(',')
    #b_formatted = '`' + ''.join(b_parts) + '`'
    #st.markdown(f'<p style="font-size: 20px; font-family: monospace;">{b_formatted}</p>', unsafe_allow_html=True)
    #st.markdown(f'<p style="font-size: 20px; color: red;"></code>{b}</code></p>', unsafe_allow_html=True)

def txt3(a, b):
  col1, col2 = st.columns([1,4])
  with col1:
    st.markdown(f'<p style="font-size: 20px;">{a}</p>', unsafe_allow_html=True)
  with col2:
    b_no_commas = b.replace(',', '')
    st.markdown(b_no_commas)

def txt4(a, b):
  col1, col2 = st.columns([1.5,2])
  with col1:
    st.markdown(f'<p style="font-size: 25px; color: white;">{a}</p>', unsafe_allow_html=True)
  with col2: #can't seem to change color besides green
    st.markdown(f'<p style="font-size: 25px; color: red;"><code>{b}</code></p>', unsafe_allow_html=True)

#####################

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bg.png')   


# Sidebar: If using streamlit_option_menu
with st.sidebar:
    with st.container():
        l, m, r = st.columns((1,3,1))
        with l:
            st.empty()
        with m:
            st.image(img_lh, width=175)
        with r:
            st.empty()
    
    choose = option_menu(
                        "Aniket Maheshwari", 
                        ["About Me", "Site Overview","Personal Chatbot", "Experience", "Technical Skills", "Education", "Projects", "Blog", "Gallery", "Resume", "Credentials", "Contact"],
                         icons=['person fill','globe' , 'star fill', 'clock history', 'tools', 'book half', 'clipboard', 'pencil square', 'heart','image','trophy fill',    'envelope','paperclip'],
                         menu_icon="mortarboard", 
                         default_index=0,
                         styles={
        "container": {"padding": "0!important", "background-color": "#f5f5dc"},
        "icon": {"color": "darkorange", "font-size": "20px"}, 
        "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#cfcfb4"},
    }
    )
    youtube_url = "https://www.youtube.com/@aniketmaheshwari7224/"
    linkedin_url = "https://www.linkedin.com/in/aniket-maheshwari-9a1568220/"
    github_url = "https://github.com/aniket072"
    email_url = "aniketmaheshwari44@gmail.com"
    with st.container():
        l, m, r = st.columns((0.11,2,0.1))
        with l:
            st.empty()

        with r:
            st.empty()

st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.title("Aniket Maheshwari")
# Create header
if choose == "About Me":
    #aboutme.createPage()
    with st.container():
        left_column, middle_column, right_column = st.columns((1,0.2,0.5))
        with left_column:
            st.header("About Me")
            st.subheader("Aspiring Data Analyst/Product Manager")
            st.write("👋🏻Hi, I'm Aniket, a final-year M.Sc. Computer Science postgraduate student at Pondicherry University.")
            st.write("💼 With the COVID-19 pandemic behind us, I believe there is potential for data science to be applied in the retail industry. In response to the increasing demand for data analytics from both online and brick-and-mortar sales, I am thus aiming to enter this industry for my first full-time job.")
            st.write("👨🏼‍💻 Academic interests: Data Visualization, Machine Learning, Database Management System")
            st.write("💭 Ideal Career Prospects: Data Analyst, Data Scientist,Machine Learning Engineer,AI/ML Consultant") 
            st.write("🏋🏻 In addition, I like to Travel,exercise in the gym, run, play cricket and... enjoy eating good food in my free time!")
            st.write("📄 [Resume (1 page)](https://drive.google.com/file/d/1AWTfQPUHizRHCvTeS7wCe5KqjA4KHITO/view?usp=drive_link)")
        with middle_column:
            st.empty()
        with right_column:
            st.image(img_utown)
elif choose == "Site Overview":   
    #overview.createPage()
    st.header("Site Overview")
    st.markdown("""
    Initally creating this as a portfolio website in the form of an extended resume, I came to discover the uniqueness of Streamlit as compared to typical front-end frameworks such as Angular and Bootstrap. Even though Streamlit is primarily used as a web application for dashboarding, its extensive features make it more aesthetically appealing to explore with as compared to alternatives such as Plotly and Shiny.
     documenting key moments and achievements.
    With the convenience of using Python as a beginner-friendly programming language, I have now decided to evolve this personal project into a time capsule -

    A video will also be embedded in this section to provide a detailed tour of this entire web application and its features.

    """)
    with st.container():
            col1, col2, col3 = st.columns((1,3,1))
            with col1:
                st.empty()
            with col2:
                st.video("https://www.youtube.com/@aniketmaheshwari7224/")
            with col3:
                st.empty()
    st.markdown("""
    *For an optimal experience, do browse this site on desktop!*

    Updated May 5, 2024
    """)

elif choose == "Personal Chatbot":
    st.header("Personal Chatbot")
    import os
    from turtle import st
    from PyPDF2 import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from dotenv import load_dotenv

    from app import cool_header, display_chat, user_input

    # Load environment variables
    load_dotenv()

    # Get Google API Key
    GOOGLE_API_KEY = os.getenv("API_KEY")

    # Check if GOOGLE_API_KEY is loaded, if not raise an error
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

    # Set the PDF directory
    pdf_destination = r"D:\Aniket Digital CV\pdf"

    # Get all PDF files from the specified directory
    pdf_docs = [
        os.path.join(pdf_destination, pdf_file) 
        for pdf_file in os.listdir(pdf_destination) 
        if pdf_file.endswith(".pdf")
    ]

    # Extract text from PDFs
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure the page has text
                text += page_text

    print("Wait!!! VectorDB in making")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=10000)
    text_chunks = text_splitter.split_text(text)

    # Initialize embeddings and create a vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the vector store locally
    vector_store.save_local("chatbot")
    print("VectorDb made!!")

    cool_header()

    user_question = st.text_input("Enter your question here")
    chats = st.session_state.get('chats', [])

    if st.button("Ask", key="ask_button"):
        with st.spinner("Amy is thinking..."):
            if user_question:
                chats = user_input(user_question, chats)
                st.session_state['chats'] = chats
                user_question = ""  # Clear input field after asking

    display_chat(chats)




# Create section for Work Experience
elif choose == "Experience":
    st.header("Experience")
    with st.container():
        image_column, text_column = st.columns((1,5))
        with image_column:
            st.image(img_bharatintern)  # Placeholder image for Bharat Intern
        with text_column:
            st.subheader("Data Science Intern, [Bharat Intern](https://bharatintern.live/)")
            st.write("*Month Year to Month Year*")
            st.markdown("""
            - Brief description of responsibilities and achievements
            - List of relevant skills
            
            `Skill1` `Skill2` `Skill3`
            """)
    with st.container():
        image_column, text_column = st.columns((1,5))
        with image_column:
            st.image(img_techietripper)  # Placeholder image for Techie Tripper
        with text_column:
            st.subheader("Database Intern, [Techie Tripper](https://www.techietripper.com/)")
            st.write("*Month Year to Month Year*")
            st.markdown("""
            - Brief description of responsibilities and achievements
            - List of relevant skills
            
            `Skill1` `Skill2` `Skill3`
            """)

#st.write("##")

# Create section for Technical Skills
elif choose == "Technical Skills":
    #st.write("---")
    st.header("Technical Skills")
    txt3("Programming Languages"," `Python`, `SQL`")
    txt3("Academic Interests","`Data Visualization`, `Recommendation Systems`, `Natural Language Processing`")
    txt3("Data Visualization", " `matplotlib`, `seaborn`, `Plotly`,  `Folium`")
    txt3("Database Systems", "`MySQL`,  `MongoDB`")
    # txt3("Cloud Platforms", "`Google Cloud Platform`, `Amazon Web Services`, `Heroku`, `Streamlit Cloud`, `Render`, `Hugging Face`, `Minio`")
    # txt3("Natural Language Processing", "`NLTK`, `Word2Vec`, `TF-IDF`, `TextStat`")
    # txt3("Version Control", "`Git`, `Docker`, `MLFlow`")
    txt3("Design and Front-end Development", "`Canva`, `HTML`, `CSS`, `Streamlit`")
    txt3("Data Science Techniques", "`Regression`,  `Decison Trees`, `Sentiment Analysis`")
    txt3("Machine Learning Frameworks", "`Numpy`, `Pandas`, `Scikit-Learn`, `TensorFlow`, `Keras`")
    txt3("Task Management Tools", " `Notion`, `Slack`")
    # txt3("Miscellaneous", " `Microsoft Office'")

#create section for education 
#st.write("---")
elif choose == "Education":
    st.header("Education")
    selected_options = ["Summary"]
    selected = st.selectbox("Which section would you like to read?", options=selected_options)
    st.write("Current selection:", selected)
    if selected == "Summary":
        st.subheader("Summary")
        st.write("*Summary of education from primary school till university*")
        with st.container():
            image_column, text_column = st.columns((1,2.5))
            with image_column:
                st.image(img_pondicherry)  # Placeholder image for Pondicherry University
            with text_column:
                st.subheader("Master of Science - Computer Science, [Pondicherry University](https://www.pondiuni.edu.in/) (December 2022 - May 2024)")
                st.write("Relevant Coursework: List of relevant coursework")
                st.markdown("""
                - [Optional] Extracurricular activities or achievements related to education
                """)
        with st.container():
            image_column, text_column = st.columns((1,2.5))
            with image_column:
                st.image(img_iimt)  # Placeholder image for IIMT Aligarh
            with text_column:
                st.subheader("Bachelor of Computer Applications - [IIMT Aligarh](https://iimtaligarh.com/) (2019 - 2022)")
                st.write("Relevant Coursework: List of relevant coursework")
                st.markdown("""
                - [Optional] Extracurricular activities or achievements related to education
                """)
        with st.container():
            image_column, text_column = st.columns((1,2.5))
            with image_column:
                st.image(img_cityconvent)  # Placeholder image for City Convent Sr. Sec. School
            with text_column:
                st.subheader("School Education - [City Convent Sr. Sec. School Atrauli](https://cityconventatrauli.blogspot.com/) (Year - Year)")
                st.write("Relevant Coursework: List of relevant coursework")
                st.markdown("""
                - [Optional] Extracurricular activities or achievements related to education
                """)


elif choose == "Projects":
    # Create section for Projects
    #st.write("---")
    st.header("Projects")
    with st.container():
        text_column, image_column = st.columns((3,1))
        with text_column:
            st.subheader("Registration form via PHP")
            st.write("*Self-initiated project*")
            st.markdown("""
            - PHP connects to a database (e.g., MySQL) to securely store user registration data, managing tasks like insertion, 
                        error handling, and security measures like password hashing.
            -  PHP ensures that user input is valid (e.g., email format, password strength) before submission, preventing errors and enhancing security.
            """)
       
            mention(label="Github Repo", icon="github", url="https://github.com/aniket072/Registration-Form-via-PHP",)
        with image_column:
            st.image(images_projects[14])
    with st.container():
        text_column, image_column = st.columns((3,1))
        with text_column:
            st.subheader("Stock Prediction using Python")
            st.write("*Self-initiated project*")
            st.markdown("""
            - Developed a stock prediction model in Python leveraging libraries such as NumPy, Pandas, and Matplotlib for data manipulation, visualization, and analysis.
            - Implemented a deep learning approach using TensorFlow's Keras API, employing LSTM (Long Short-Term Memory) neural networks
                         for time-series forecasting, with features like dropout layers for regularization.
            """)
            # st.write("[Github Repo](https://github.com/harrychangjr/sales-prediction)")
            mention(label="Github Repo", icon="github", url="https://github.com/aniket072/BharatIntern/tree/main/stockpredict",)
        with image_column:
            st.image(images_projects[13])
    with st.container():
        text_column, image_column = st.columns((3,1))
        with text_column:
            st.subheader("Titanic Survivour Prediction")
            st.write("*Self-initiated project based on e-commerce case study*")
            st.markdown("""
            - Utilized Pandas and NumPy for data manipulation, alongside Scikit-learn for preprocessing
                         tasks such as label encoding and splitting the dataset into training and testing sets.
            - Employed a Random Forest classifier from Scikit-learn to predict survival outcomes, achieving accuracy
                         assessment through metrics like accuracy score and confusion matrix visualization with Matplotlib and Seaborn
            """)
            # st.write("[Github Repo](https://github.com/harrychangjr/sales-prediction)")
            mention(label="Github Repo", icon="github", url="https://github.com/aniket072/BharatIntern/tree/main/titanic%20surv",)
        with image_column:
            st.image(images_projects[0])
    with st.container():
        text_column, image_column = st.columns((3,1))
        with text_column:
            st.subheader("Number Recognition System")
            st.write("*Self-initiated project*")
            st.markdown("""
            - Implemented a convolutional neural network (CNN) for number recognition using TensorFlow and Keras, leveraging the MNIST dataset for training and testing.
            - "Designed a Sequential model architecture with Conv2D and MaxPooling2D layers for feature extraction, followed by Flatten and Dense layers for classification.
            """)
            #st.write("[Github Repo](https://github.com/harrychangjr/sp1541-nlp)")
            mention(label="Github Repo", icon="github", url="https://github.com/aniket072/BharatIntern/blob/main/Number%20Recognition.ipynb",)
        with image_column:
            st.image(images_projects[1])
    with st.container():
        text_column, image_column = st.columns((3,1))
        with text_column:
            st.subheader("COVID-19 Data Analysis")
            st.write("*Academic Year 2022/23 Semester 2*")
            #st.write("Methods performed on [Kaggle dataset](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings):")
            st.markdown("""
            - Developed interactive visualizations using libraries like Matplotlib and Seaborn in Python, showcasing trends, distribution, and impact of COVID-19 cases across regions.
            - Utilized data from reliable sources such as Johns Hopkins University and WHO, enhancing understanding of pandemic dynamics through insightful plots, heatmaps, and time series analyses.
            """)
            #st.write("[Github Repo](https://github.com/harrychangjr/st4248-termpaper) | [Term Paper](https://github.com/harrychangjr/st4248-termpaper/blob/main/ST4248%20Term%20Paper%20(A0201825N)%20v5.pdf)")
            mention(label="Github Repo", icon="github", url="https://github.com/aniket072",)
        with image_column:
            st.image(images_projects[2])

    
elif choose == "Blog":
    st.header("Blog")

elif choose == "Gallery":
    st.header("Gallery")
    st.subheader("Some of my favorite photos!")
    import streamlit as st
    import os

    def show_photos(directory):
        photos = [f"p{i}.jpg" for i in range(1, 21)]  # Adjust the range to include all photos
        num_photos = len(photos)
        cols = st.columns(2)  # Adjust the number of columns as needed

        for i, photo_name in enumerate(photos):
            with cols[i % 2]:
                photo_path = os.path.join(directory, photo_name)
                if os.path.exists(photo_path):
                    st.image(photo_path, caption=f"Photo {photo_name}", width=400)

    photo_directory = "gallery"
    show_photos(photo_directory)






elif choose == "Resume":   
    resume_url = "https://drive.google.com/file/d/1AWTfQPUHizRHCvTeS7wCe5KqjA4KHITO/view?usp=drive_link"
    st.header("Resume")
    st.write("*In case your current browser cannot display the PDF documents, do refer to the hyperlink below!*")

    st.markdown(pdf_link(resume_url, "**Resume (1 page)**"), unsafe_allow_html=True)
    show_pdf("Aniket_Resume.pdf")
    with open("Aniket_Resume.pdf", "rb") as file:
        btn = st.download_button(
            label="Download Resume (1 page)",
            data=file,
            file_name="Aniket_Resume.pdf",
            mime="application/pdf"
        )
elif choose == "Credentials": 
    test_url = "https://drive.google.com/file/d/1xBdumO4z4l09Ta6masVy-actGocaidOl/view?usp=drive_link"  
    st.header("Credentials")
    st.subheader("Some appraisals from my past referees!")
    st.markdown(pdf_link(test_url, "**Compiled Credentials**"), unsafe_allow_html=True)  
    import streamlit as st

    with st.container():
     col1, col2 = st.columns(2)

    with col1:
        st.subheader("MongoDB")
        show_pdf("c1.pdf")

        st.subheader("Infosys")
        show_pdf("c3.pdf")

    with col2:
        st.subheader("Lyft,Forage")
        show_pdf("c2.pdf")

        st.subheader("HackerRank")
        show_pdf("c4.pdf")

elif choose == "Contact":
# Create section for Contact
    #st.write("---")
    st.header("Contact")
    def social_icons(width=24, height=24, **kwargs):
        icon_template = '''
        <a href="{url}" target="_blank" style="margin-right: 10px;">
            <img src="{icon_src}" alt="{alt_text}" width="{width}" height="{height}">
        </a>
        '''

        icons_html = ""
        for name, url in kwargs.items():
            icon_src = {
                "linkedin": "https://cdn-icons-png.flaticon.com/512/174/174857.png",
                "github": "https://cdn-icons-png.flaticon.com/512/25/25231.png",
                "email": "https://cdn-icons-png.flaticon.com/512/561/561127.png"
            }.get(name.lower())

            if icon_src:
                icons_html += icon_template.format(url=url, icon_src=icon_src, alt_text=name.capitalize(), width=width, height=height)

        return icons_html
    with st.container():
        text_column, mid, image_column = st.columns((1,0.2,0.5))
        with text_column:
            st.write("Let's connect! You may either reach out to me at aniketmaheshwari44@gmail.com or use the form below!")
            
            def create_database_and_table():
                conn = sqlite3.connect('contact_form.db')
                c = conn.cursor()
                c.execute('''CREATE TABLE IF NOT EXISTS contacts
                            (name TEXT, email TEXT, message TEXT)''')
                conn.commit()
                conn.close()
            create_database_and_table()

            st.subheader("Contact Form")
            if "name" not in st.session_state:
                st.session_state["name"] = ""
            if "email" not in st.session_state:
                st.session_state["email"] = ""
            if "message" not in st.session_state:
                st.session_state["message"] = ""
            st.session_state["name"] = st.text_input("Name", st.session_state["name"])
            st.session_state["email"] = st.text_input("Email", st.session_state["email"])
            st.session_state["message"] = st.text_area("Message", st.session_state["message"])


            column1, column2= st.columns([1,5])
            if column1.button("Submit"):
                conn = sqlite3.connect('contact_form.db')
                c = conn.cursor()
                c.execute("INSERT INTO contacts (name, email, message) VALUES (?, ?, ?)",
                        (st.session_state["name"], st.session_state["email"], st.session_state["message"]))
                conn.commit()
                conn.close()
                st.success("Your message has been sent!")
                # Clear the input fields
                st.session_state["name"] = ""
                st.session_state["email"] = ""
                st.session_state["message"] = ""
            def fetch_all_contacts():
                conn = sqlite3.connect('contact_form.db')
                c = conn.cursor()
                c.execute("SELECT * FROM contacts")
                rows = c.fetchall()
                conn.close()
                return rows
            
            if "show_contacts" not in st.session_state:
                st.session_state["show_contacts"] = False
            if column2.button("View Submitted Forms"):
                st.session_state["show_contacts"] = not st.session_state["show_contacts"]
            
            if st.session_state["show_contacts"]:
                all_contacts = fetch_all_contacts()
                if len(all_contacts) > 0:
                    table_header = "| Name | Email | Message |\n| --- | --- | --- |\n"
                    table_rows = "".join([f"| {contact[0]} | {contact[1]} | {contact[2]} |\n" for contact in all_contacts])
                    markdown_table = f"**All Contact Form Details:**\n\n{table_header}{table_rows}"
                    st.markdown(markdown_table)
                else:
                    st.write("No contacts found.")


            st.write("Alternatively, feel free to check out my social accounts below!")
            linkedin_url = "https://www.linkedin.com/in/aniket-maheshwari-9a1568220/"
            github_url = "https://github.com/aniket072"
            email_url = "aniketmaheshwari44@gmail.com"
            st.markdown(
                social_icons(32, 32, LinkedIn=linkedin_url, GitHub=github_url, Email=email_url),
                unsafe_allow_html=True)
            st.markdown("")
            #st.write("© 2024 Aniket Maheshwari")
            #st.write("[LinkedIn](https://linkedin.com/in/harrychangjr) | [Github](https://github.com/harrychangjr) | [Linktree](https://linktr.ee/harrychangjr)")
        with mid:
            st.empty()
        with image_column:
            st.image(img_ifg)
st.markdown("*Copyright © 2024 Aniket Maheshwari*")

