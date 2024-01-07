# Import necessary libraries.

import pickle
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np
import time as t
from pathlib import Path
import base64


def get_background(file_name):

    main_bg_ext = "png"

    with open(file_name, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    return st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{encoded_image});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to display a success message while logging in
@st.cache_resource
def loading_msg():
    success_message = st.success("Logging in")
    t.sleep(0.2)
    success_message.empty()

# Function to customize the web layout by hiding certain elements

def web_customes():                                                                         
    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """

    return st.markdown(hide_st_style, unsafe_allow_html=True)

# CSS styling for login page
def set_custom_login_style():
    
    st.markdown(
        """
        <style>
            .stApp {
                color: white !important;
            }
            .stTextInput div div div {
                color: white !important;
                background-color: #FFFFFF !important;
            }
            .stTextInput div div input {
                color: white !important;
                background-color: #FFFFFF !important;
            }
            .stButton {
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def app() :
    
    st.set_page_config(page_title="Fraud Detection System", page_icon=":Info", layout="centered")

    get_background("robo.png") 
    set_custom_login_style()

    # Function to get data from a CSV file (cached for efficiency)
    @st.cache_data
    def get_data():
        df = pd.read_csv("fraud_dataset_20K.csv", index_col=None)

        return df

    df = get_data()             # Read the data
    
    web_customes()              # Customize the web layout

    #--------------------------------------------------
    st.markdown("<h2 style='text-align: center; color: #6c757d; font-size: 1px;'>Welcome Back!</h2>", unsafe_allow_html=True)
    #--------------------------------------------------

    # User-Authentication
    usernames = ['Admin', 'Shuhaib']
    names = ['name1', 'name2']

    # Read hashed passwords from a file
    file_path = Path(__file__).parent / "hashed_pw.pkl"
    with open(file_path, 'rb') as file:
        hashed_passwords = pickle.load(file)

    credentials = {'usernames' : {}}

    # Create a dictionary with usernames, names, and hashed passwords
    for uname, name, pwd in zip(usernames, names, hashed_passwords):
        user_dict = {'name' : name, 'password' : pwd}
        credentials['usernames'].update({uname: user_dict})

    # Set up user-authentication
    authenticator = stauth.Authenticate(credentials, "Fraud Detection", "random_key", cookie_expiry_days=0)
    name, authentication_status, username = authenticator.login('Login', 'main')

    # Handle authentication status
    if authentication_status == None:
        st.markdown("<h3 style='text-align: center; color: #6c757d; font-size: 16px;'>Please Enter the Username and Password</h3>", unsafe_allow_html=True)

    if authentication_status == False:
        st.error("Username / Password is incorrect")
    elif authentication_status == True:
        loading_msg()
    

    # If Authentication is successful
    if authentication_status:

        selected = option_menu(
            menu_title=None,
            options=['Model', 'Info', 'Contact'],
            icons=['house', 'book', 'envelope'],
            menu_icon='cast',
            default_index=0,
            orientation='horizontal',
        )

        # Handle different menu - Model selections
        if selected == 'Model':

            get_background("tech1.png")

            st.markdown("### Enter the details")

            with st.form(key='form', clear_on_submit=False):

                features_1 = st.text_input("Enter : Category | Amount | Population | Job")
                features_2 = st.text_input("Enter : Latitude | Longitude | Year | Month |Hour | Day")

                gender = st.selectbox("Gender", ['Male', 'Female'])
                age = st.slider("Age", 1, 100)

                if gender == 'Male':
                    gender = 1
                else:
                    gender = 0

                if features_1 and features_2 :

                    category, amt, pop, job = map(str.strip, features_1.split(','))
                    lat, lon, year, month, hour, day = map(str.strip, features_2.split(','))

                    to_convert = [category, amt, pop, job, lat, lon, year, month, hour, day]

                    for i in range(len(to_convert)):
                        to_convert[i] = float(to_convert[i]) if to_convert[i] else 0

                    lst = [category, amt, pop, job, lat, lon, year, month, hour, day, gender, age]

                submit = st.form_submit_button("Predict")

                if submit:                                                        
                    with open('Model.pkl', 'rb') as f:
                        model = pickle.load(f)                                    
                    
                    features = np.asarray(lst, dtype=np.float64)
                    prediction = model.predict(features.reshape(1,-1))

                    if prediction[0] == 0:                                        
                        st.success("Legitimate Transaction")
                        st.balloons()
                    else:       
                        st.error("Fradulant Transaction")       

            # Logout after using the Model section
            authenticator.logout("Logout", "main")


        # Handle different menu - Info selections
        if selected == "Info":

            get_background("tech_bg4.png")                      # Set background for Info section
            
            # Display information about the data
            if 'number_of_rows' not in st.session_state:
                st.session_state['number_of_rows'] = 5

            # Input for the number of rows
            increment = st.text_input('Specify the number of rows to be displayed')
            if increment:
                increment = int(increment)
                st.session_state['number_of_rows'] = increment

            # Input for target classes in the sidebar
            st.sidebar.markdown("<h2 style='text-align: center; color: #ff758f; font-size: 18px;'>Please Filter Here:</h2>", unsafe_allow_html=True)

            target_category = st.sidebar.multiselect(
                "Select the Category: ",
                options=df['category'].unique(),
                default=['entertainment', 'gas_transport', 'personal_care', 'health_fitness', 'home']
            )

            # Apply filters to the DataFrame
            filtered_df = df[df['category'].isin(target_category)].head(st.session_state['number_of_rows'])

            # Display the filtered DataFrame
            st.dataframe(filtered_df)


        if selected == "Contact":

            get_background("robo1.png")
            
            # Display a heading for the Contact section
            st.markdown("<h2 style='text-align: center; color: #ced4da; font-size: 22px;'><span style='margin-right: 10px;'>📬</span>Get In Touch with us!</h2>", unsafe_allow_html=True)

            # Contact form HTML code
            contact_form = """
            <form action="https://formsubmit.co/bursins77@gmail.com" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Name" required style="color: #001d3d;">
                <input type="email" name="email" placeholder="Email" pattern="[a-zA-Z]+[0-9]*@[a-zA-Z]+\.[a-zA-Z]{2,}" required style="color: #001d3d;">
                <textarea name="Message" placeholder="Your message here"></textarea>
                <button type="Submit">Send</button>
            </form>            
            """
            
            # Display the contact form
            st.markdown(contact_form, unsafe_allow_html=True)

            #Apply css style
            def Style_css(file_name):
                with open(file_name) as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

            Style_css("Style_css.css")

# Main function to run the app
def main():
    app()

# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
