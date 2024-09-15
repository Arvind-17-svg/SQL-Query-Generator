import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_name = "Salesforce/codegen-350M-mono"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_sql_query(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=150,
        num_return_sequences=1,
        temperature=0.5,
        pad_token_id=tokenizer.eos_token_id
    )
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

def main():
    
    st.set_page_config(page_title="SQL Query Generator", page_icon="🤖")

    
    st.title("SQL Query Generator")
    
    
    user_input = st.text_area("Enter a description for the SQL query you need:")
    
    
    if st.button("Generate SQL Query"):
        if user_input:
            with st.spinner("Generating SQL Query..."):
                prompt = f"Generate an SQL query for the following request:\n{user_input}\n\nSQL Query:"
                sql_query = generate_sql_query(prompt)
                
                # Display the generated SQL query
                st.subheader("Generated SQL Query")
                st.code(sql_query, language="sql")
        else:
            st.error("Please enter a description for the SQL query.")
    
if __name__ == "__main__":
    main()
