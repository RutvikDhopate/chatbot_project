Hello, welcome to the ChatBot Integration Project

Please, read me before any installation, for a smooth onboarding.

My recommendation is for you to create a virtual environment, I personally recommend uv package manager as it is much faster than pip.

Command to install the requirements
pip install -r requirements.txt

The above command will install everything needed for the files to run.

Create a folder, export the GitHub files into a single folder. 
Feel free to make changes to the architecture and refactor accordingly.

* Please create and add respective API_KEYS in the .env file in the root of your folder.
Example of .env file is given below
GROQ_API_KEY=<YOUR_API_KEY>

temp_main.py - Streamlit Frontend, file handling and session state management
openai_llm.py - Groq's Open AI SDK implementation to configure images, csv, and pdfs.
utils.py - Custom functions for smooth performance
.streamlit/config.toml - Streamlit frontend custom configurations
requirements.txt - Necessary Python Packages
llm_prompt.txt - Custom System Prompt given by the user to the LLM
testing_llm.py - Backend testing scripts for enhanced performance

* Run the streamlit file
Once the architecture is setup, you need the below command to run the file
python3 -m streamlit run temp_main.py

Happy Chatting! :D