from flask import Flask, render_template, request, session
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper,\
    ServiceContext
from langchain import OpenAI
import os


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="gpt-3.5-turbo-0301", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index


def ask_ai(query):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(query)
    response = str(response)
    return response


app = Flask(__name__)
app.secret_key = 'super_secret_key'

app.config['SESSION_TYPE'] = 'filesystem'


@app.route('/', methods=['GET', 'POST'])
def home():
    if 'messages' not in session:
        session['messages'] = []

    if request.method == 'POST':
        user_query = request.form['user_query']
        response = ask_ai(user_query)
        with open('log.txt', 'a') as f:
            f.writelines(user_query + '\n' + response + '\n' + '\n')
            f.close()
        session['messages'].append({'user_query': user_query, 'response': response})
        session.modified = True  # Garante que a sessão seja salva após a modificação
        return render_template('index.html', last_message=session['messages'][-1], messages=session['messages'])
    else:
        session['messages'] = []

    return render_template('index.html', last_message=None, messages=session['messages'])


os.environ["OPENAI_API_KEY"] = "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"
construct_index("data")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
