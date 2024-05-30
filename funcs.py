from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer
import torch
import re
import numpy as np


model = T5ForConditionalGeneration.from_pretrained('./my_model24k/')



model_name = 'cointegrated/rut5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer_chat = T5Tokenizer.from_pretrained("cointegrated/rut5-small-chitchat")
model_chat = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-small-chitchat")




def generate_tox(input_text, model=model, tokenizer=tokenizer):
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate the output
    output_ids = model.to('cpu').generate(input_ids, max_length=50)

    # Decode the output
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    def remove_text_inside_brackets(text):
        return re.sub(r'<[^>]*>', '', text)

    return remove_text_inside_brackets(decoded_output)



def generate_text(prompt, model_chat = model_chat, tokenizer_chat = tokenizer_chat, max_length=50):
    
    inputs = tokenizer_chat(prompt, return_tensors='pt')
    with torch.no_grad():
        hypotheses = model_chat.generate(
            **inputs, 
            do_sample=True, top_p=0.5, num_return_sequences=3, 
            repetition_penalty=2.5,
            max_length=max_length,
        )

    
    choice_idx = np.random.randint(0, len(hypotheses))
    h = hypotheses[choice_idx]
    return tokenizer_chat.decode(h, skip_special_tokens=True)



def msg_pipeline(promt):
    answer = generate_text(promt)
    tox_answer = generate_tox(answer)
    return tox_answer


