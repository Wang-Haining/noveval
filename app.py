import os
import torch
from functools import lru_cache
from model import GPTConfig, GPT
from flask import Flask, render_template, request, jsonify
from utils import encode, calculate_perplexity, colorize_text, decode_ids_for_visualization


@lru_cache(maxsize=1_000)
def load_model():
    ckpt_path = os.path.join('out_wikipedia_en', 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    return model


app = Flask(__name__)


@app.route('/get_token_count', methods=['POST'])
def get_token_count():
    text = request.form['text']
    tokens = encode(text)
    return jsonify(token_count=len(tokens))


@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ""
    results = None

    if request.method == 'POST':
        user_input = request.form['user_input']
        ppl, cross_entropy, ids, _ = calculate_perplexity(
            text=user_input,
            model=load_model(),
            computing_method='long_history',
            device='cpu',
            sequence_length=2048,
            block_size=1024,
            minimum_context_length=512,
            sampling=True,
            random_state=42,
            compile_model=True,
            verbosity=True
        )
        # Decode ids to get words
        words = decode_ids_for_visualization(ids)

        # Colorize the words based on their cross-entropy values
        colorized_text = colorize_text(words, cross_entropy)
        results = {
            'ppl': ppl,
            'colorized_text': colorized_text
        }

    return render_template('index.html', user_input=user_input, results=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
