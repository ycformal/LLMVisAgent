from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json
import os
import openai
from PIL import Image
from engine.step_interpreters import *
from engine.utils import ProgramInterpreter, ProgramGenerator
from prompts.gqa_baseline1_v1 import create_prompt as create_prompt_gqa
from prompts.nlvr import create_prompt as create_prompt_nlvr
from engine.validator import validate
from engine.judger import judge

import openai
key = "Your API key"
openai.api_key=key

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this for production

# Define global variables for the objects you want to be available across requests.
program_generator = ProgramGenerator(model_name_or_path="gpt-3.5-turbo-instruct")
module_list = [
    "LOC", "CROP", "CROP_ABOVE", "CROP_BELOW", "CROP_LEFTOF", "CROP_RIGHTOF", 
    "CROP_BEHIND", "CROP_INFRONT", "CROP_INFRONTOF", "CROP_AHEAD", "FROP_FRONTOF", 
    "COUNT", "VQA", "RESULT", "EVAL", "GET"
]
# We'll set program_interpreter later (per user selection)
program_interpreter = None

def solve(images, question):
    if len(images) == 1:
        script = program_generator.generate(dict(question=question), create_prompt_gqa)
    elif len(images) == 2:
        script = program_generator.generate(dict(statement=question), create_prompt_nlvr)
    return script

def ssparser(question, script):
    try:
        init_state = {'IMAGE': Image.new('RGB', (640, 640))}
        if 'IMAGE' not in script:
            init_state['LEFT'] = Image.new('RGB', (640, 640))
            init_state['RIGHT'] = Image.new('RGB', (640, 640))
        return validate(script, question, module_list, init_state)
    except Exception as e:
        if 'IMAGE' not in script:
            question = "Is '{question.replace('(','').replace(')','')}' true or false?"
        return f"""ANSWER0=VQA(image=IMAGE,question="{question}")\nFINAL_RESULT=RESULT(var=ANSWER0)"""

def execute(script, images, question):
    global program_interpreter
    html_list = None
    result = None
    print(images)
    if len(images) == 1:
        init_state = {'IMAGE': images[0]}
        try:
            result, _, html_list = program_interpreter.execute(script, init_state, inspect=True)
        except:
            result, _, html_list = program_interpreter.execute(
                f"X=VQA(image=IMAGE,question='{question}')", init_state, inspect=True)
    elif len(images) == 2:
        image = Image.new('RGB', (images[0].width + images[1].width, max(images[0].height, images[1].height)))
        image.paste(images[0], (0, 0))
        image.paste(images[1], (images[0].width, 0))
        init_state = {'LEFT': images[0], 'RIGHT': images[1], 'IMAGE': image}
        try:
            result, _, html_list = program_interpreter.execute(script, init_state, inspect=True)
        except:
            question = "Is '{question.replace('(','').replace(')','')}' true or false?"
            result, _, html_list = program_interpreter.execute(
                f"X=VQA(image=IMAGE,question='{question}')", init_state, inspect=True)
    return html_list, result

def caption(image):
    script = "X=CAP(image=IMAGE)"
    init_state = {'IMAGE': image}
    caption_text, _ = program_interpreter.execute(script, init_state)
    return caption_text

# Now judge expects a JSON with the format: {'agent': {'program': script, 'answer': result}}
def output_verifier(question, script, result, caption_text):
    results = {'agent': {'program': script, 'answer': result}, 'caption': caption_text}
    verified_result, analysis = judge(results, question)
    return analysis, verified_result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_models', methods=['POST'])
def select_models():
    _vqa_models = request.form.getlist('vqa_models')
    _loc_models = request.form.getlist('loc_models')
    vqa_models = {'blip': [], 'vilt': [], 'paligemma': []}
    loc_models = {'owlvit': [], 'owlv2': []}
    for model in _vqa_models:
        if 'blip' in model:
            vqa_models['blip'].append(model.split('|')[-1].strip())
        elif 'vilt' in model:
            vqa_models['vilt'].append(model.split('|')[-1].strip())
        elif 'paligemma' in model:
            vqa_models['paligemma'].append(model.split('|')[-1].strip())
    for model in _loc_models:
        if 'owlvit' in model:
            loc_models['owlvit'].append(model.split('|')[-1].strip())
        elif 'owlv2' in model:
            loc_models['owlv2'].append(model.split('|')[-1].strip())
    # We can store these choices in the session if you like; here they are used to initialize the global interpreter.
    session['vqa_models'] = vqa_models
    session['loc_models'] = loc_models
    print(vqa_models, loc_models)
    global program_interpreter
    program_interpreter = ProgramInterpreter(loc_models, vqa_models)
    return redirect(url_for('main'))

@app.route('/main')
def main():
    examples = []
    examples_path = os.path.join('examples', 'examples.json')
    if os.path.exists(examples_path):
        with open(examples_path, 'r') as f:
            examples = json.load(f)
    return render_template('main.html', examples=examples)

@app.route('/solve', methods=['POST'])
def solve_route():
    question = request.form.get('question')
    # First, try to get uploaded images.
    uploaded_images = request.files.getlist('images[]')
    uploaded_images = [img for img in request.files.getlist('images[]') if img.filename != ""]
    if uploaded_images and any(img.filename for img in uploaded_images):
        script = solve(uploaded_images, question)
    else:
        # Otherwise, use the example image URLs from hidden inputs.
        ex1 = request.form.get('example_image1')
        ex2 = request.form.get('example_image2')
        example_images = []
        if ex1:
            example_images.append(ex1)
        if ex2:
            example_images.append(ex2)
        script = solve(example_images, question)
    return jsonify({"script": script})

@app.route('/ssparser', methods=['POST'])
def ssparser_route():
    script = request.form.get('script')
    question = request.form.get('question')
    new_script = ssparser(question, script)
    return jsonify({"script": new_script})

@app.route('/execute', methods=['POST'])
def execute_route():
    script = request.form.get('script')
    uploaded_images = request.files.getlist('images[]')
    uploaded_images = [img for img in request.files.getlist('images[]') if img.filename != ""]
    processed_images = []
    if uploaded_images and any(img.filename for img in uploaded_images):
        for img_file in uploaded_images:
            if img_file.filename != "":
                img = Image.open(img_file)
                img.thumbnail((640, 640), Image.Resampling.LANCZOS)
                img = img.convert('RGB')
                processed_images.append(img)
    html_list, result = execute(script, processed_images, request.form.get('question'))
    return jsonify({"html": html_list, "result": result})

@app.route('/caption', methods=['POST'])
def caption_route():
    uploaded_images = request.files.getlist('images[]')
    uploaded_images = [img for img in request.files.getlist('images[]') if img.filename != ""]
    processed_images = []
    if uploaded_images and any(img.filename for img in uploaded_images):
        for img_file in uploaded_images:
            if img_file.filename != "":
                img = Image.open(img_file)
                img.thumbnail((640, 640), Image.Resampling.LANCZOS)
                img = img.convert('RGB')
                processed_images.append(img)
    cap_text = caption(processed_images[0])
    return jsonify({"caption": cap_text})

@app.route('/output_verifier', methods=['POST'])
def output_verifier_route():
    caption_text = request.form.get('caption')
    script = request.form.get('script')
    result = request.form.get('result')
    question = request.form.get('question')
    analysis, verified_result = output_verifier(question, script, result, caption_text)
    return jsonify({"analysis": analysis, "result": verified_result})

if __name__ == '__main__':
    app.run(debug=True)
