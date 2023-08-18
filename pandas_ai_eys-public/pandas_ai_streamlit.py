import pandas as pd
import streamlit as st
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pandasai.middlewares.streamlit import StreamlitMiddleware

df_salary = pd.read_csv("ds_salaries.csv")
df_cars = pd.read_csv("Automobile.csv")
df_pop = pd.read_csv("countries-table.csv")

llm = OpenAI(api_token="YOUR API KEY")
pandas_ai = PandasAI(llm, middlewares=[StreamlitMiddleware()], enable_cache=False)

def main():
    st.title("PandasAI")
    
    st.markdown('''
    Pandas, a powerhouse in data analytics, has been given a turbo-boost. We’re talking about Pandas AI, which interlaces the might of large language models (LLMs), like those from OpenAI, into the established Pandas environment.
    
    The shining star of Pandas AI is undoubtedly its querying capability. This unique feature enables one to directly ask the AI model any sort of question about the data at hand. This is an interactive web app that demonstrates the capabilities of PandasAI.
    
    Learn More: [https://github.com/gventuri/pandas-ai](https://github.com/gventuri/pandas-ai)
    ''')

    st.title("How to Use It?")

    st.markdown('''
    1. Select what dataset you want to use.
    2. Input a prompt on what you want to do with the dataset.
    3. Press Run
    ''')

    # DataFrame selection dropdown
    df_choice = st.selectbox("Datasets", ("Data Professionals Salaries", "Automobile Industry", "Population per Country"))

    # Choose DataFrame based on selection
    if df_choice == "Data Professionals Salaries":
        df = df_salary
        st.markdown("### Data Professionals Salaries")
    elif df_choice == "Automobile Industry":
        df = df_cars
        st.markdown("### Automobile Industry")
    elif df_choice == "Population per Country":
        df = df_pop
        st.markdown("### Population per Country")

    # Display DataFrame head
    st.write(df.head())

    # Example Prompts
    st.markdown("#### Example Prompts")

    if df_choice == "Data Professionals Salaries":
        st.markdown("_What are the top 10 highest paying job titles?_")
        st.markdown("_Create a pie chart that shows the distribution of different experience levels._")
        st.markdown("_Create a horizontal bar chart that shows the top 3 countries with the highest average salaries._")
    elif df_choice == "Automobile Industry":
        st.markdown("_What are the 10 heaviest cars?_")
        st.markdown("_Are American cars faster than cars from other countries?_")
        st.markdown("_Make a box plot for the horsepower of the cars._")
    elif df_choice == "Population per Country":
        st.markdown("_Does India have the fastest growth rate?_")
        st.markdown("_Through a bar chart, compare the population densities between different continents._")
        st.markdown("_Make a line chart for China's population growth using the pop1980 to pop2050 columns._")

    # User prompt input
    prompt = st.text_input("Enter a prompt:")
    if st.button("Run"):
        with st.spinner("Generating..."):
            st.write(pandas_ai(df, prompt=prompt))

if __name__ == "__main__":
    main()
