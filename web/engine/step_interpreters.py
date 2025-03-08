import torch
import io, tokenize
from PIL import Image,ImageDraw
from transformers import (ViltProcessor, ViltForQuestionAnswering, 
    OwlViTProcessor, OwlViTForObjectDetection, AutoProcessor, BlipForQuestionAnswering,
    AutoModelForCausalLM, Owlv2Processor, Owlv2ForObjectDetection,
    PaliGemmaForConditionalGeneration)
from nltk.stem import PorterStemmer
import math
import nltk

from .nms import nms
from vis_utils import html_embed_image, html_colored_span, vis_masks

nltk.download('punkt_tab') # If there's error message, just modify the package name according to what the error message says


def parse_step(step_str,partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2*i].string] = arg_tokens[2*i+1].string
    parsed_result['args'] = args
    return parsed_result


def html_step_name(content):
    step_name = html_colored_span(content, 'red')
    return f'<b>{step_name}</b>'


def html_output(content):
    output = html_colored_span(content, 'green')
    return f'<b>{output}</b>'


def html_var_name(content):
    var_name = html_colored_span(content, 'blue')
    return f'<b>{var_name}</b>'


def html_arg_name(content):
    arg_name = html_colored_span(content, 'darkorange')
    return f'<b>{arg_name}</b>'

    
class EvalInterpreter():
    step_name = 'EVAL'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        step_input = eval(parse_result['args']['expr'])
        assert(step_name==self.step_name)
        return step_input, output_var
    
    def html(self,eval_expression,step_input,step_output,output_var):
        eval_expression = eval_expression.replace('{','').replace('}','')
        step_name = html_step_name(self.step_name)
        var_name = html_var_name(output_var)
        output = html_output(step_output)
        expr = html_arg_name('expression')
        return f"""<div>{var_name}={step_name}({expr}="{eval_expression}")={step_name}({expr}="{step_input}")={output}</div>"""

    def execute(self,prog_step,inspect=False):
        step_input, output_var = self.parse(prog_step)
        prog_state = dict()
        for var_name,var_value in prog_step.state.items():
            if isinstance(var_value,str):
                if var_value in ['yes','no']:
                    prog_state[var_name] = var_value=='yes'
                elif var_value.isdecimal():
                    prog_state[var_name] = var_value
                else:
                    prog_state[var_name] = f"'{var_value}'"
            else:
                prog_state[var_name] = var_value
        
        eval_expression = step_input

        if 'xor' in step_input:
            step_input = step_input.replace('xor','!=')

        step_input = step_input.format(**prog_state)
        step_output = eval(step_input)
        prog_step.state[output_var] = step_output
        if inspect:
            html_str = self.html(eval_expression, step_input, step_output, output_var)
            return step_output, html_str

        return step_output
    
class GetInterpreter():
    step_name = 'GET'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        image_var = parse_result['args']['image']
        assert(step_name==self.step_name)
        return image_var, output_var
    
    def html(self,img,bounding_box,output_var):
        step_name = html_step_name(self.step_name)
        img_str = html_embed_image(img)
        bounding_box = html_output(bounding_box)
        output_var = html_var_name(output_var)
        image_arg = html_arg_name('image')
        return f"""<div>{output_var}={step_name}({image_arg}={img_str})={bounding_box}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var, output_var = self.parse(prog_step)
        prog_state = dict()
        img = prog_step.state[img_var]

        bounding_box = [[0, 0, img.size[0] - 1, img.size[1] - 1]]

        prog_step.state[output_var] = bounding_box
        if inspect:
            html_str = self.html(img, bounding_box, output_var)
            return bounding_box, html_str

        return bounding_box

class ResultInterpreter():
    step_name = 'RESULT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['args']['var']
        assert(step_name==self.step_name)
        return output_var

    def html(self,output,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        if isinstance(output, Image.Image):
            output = html_embed_image(output,300)
        else:
            output = html_output(output)
            
        return f"""<div>{step_name} -> {output_var} -> {output}</div>"""

    def execute(self,prog_step,inspect=False):
        output_var = self.parse(prog_step)
        if output_var in prog_step.state:
            output = prog_step.state[output_var]
        else:
            output = eval(output_var)
        if inspect:
            html_str = self.html(output,output_var)
            return output, html_str

        return output

class AssignInterpreter():
    step_name = 'ASSIGN'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['args']['var']
        new_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return output_var, new_var

    def html(self,output,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        if isinstance(output, Image.Image):
            output = html_embed_image(output,300)
        else:
            output = html_output(output)
            
        return f"""<div>{step_name} -> {output_var} -> {output}</div>"""

    def execute(self,prog_step,inspect=False):
        output_var, new_var = self.parse(prog_step)
        if output_var in prog_step.state:
            output = prog_step.state[output_var]
            prog_step.state[new_var] = output
        else:
            output = eval(output_var)
            prog_step.state[new_var] = output
        if inspect:
            html_str = self.html(output,output_var)
            return output, html_str

        return output


class VQAInterpreter():
    step_name = 'VQA'

    def __init__(self, models):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = [AutoProcessor.from_pretrained(model) for model in models['blip']] + [ViltProcessor.from_pretrained(model) for model in models['vilt']] + [AutoProcessor.from_pretrained(model) for model in models['paligemma']]
        self.model = [BlipForQuestionAnswering.from_pretrained(model).to(self.device) for model in models['blip']] + [ViltForQuestionAnswering.from_pretrained(model).to(self.device) for model in models['vilt']] + [PaliGemmaForConditionalGeneration.from_pretrained(model).to(self.device) for model in models['paligemma']]
        for model in self.model:
            model.eval()
        self.stemmer = PorterStemmer()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        args = parse_result['args']
        img_var = args['image']
        question = eval(args['question'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,question,output_var

    def predict(self,img,question):
        results = []
        for i in range(len(self.processor)):
            encoding = self.processor[i](img,question,return_tensors='pt')
            encoding = {k:v.to(self.device) for k,v in encoding.items()}
            with torch.no_grad():
                try:
                    outputs = self.model[i].generate(**encoding, max_new_tokens=50)
                    result = self.processor[i].decode(outputs[0], skip_special_tokens=True)
                    if question not in result:
                        results.append(result)
                    else:
                        result = result.replace(question,'').strip()
                        results.append(result)
                except:
                    try:
                        outputs = self.model[i](**encoding)
                        logits = outputs.logits
                        idx = logits.argmax(-1).item()
                        results.append(self.model[i].config.id2label[idx])
                    except:
                        print(question)
                        print('Model failed to generate answer. Update the code for VQA answer generation.')
                        results.append('Runtime error on this VQA model.')
        stemmed_results = [''] * len(results)
        for i in range(len(results)):
            words = nltk.word_tokenize(results[i].lower())
            stems = {self.stemmer.stem(word) for word in words if word.isalnum()}
            stemmed_results[i] = ' '.join(stems)

        votes = [0] * len(results)

        for i, stems_i in enumerate(stemmed_results):
            for j, stems_j in enumerate(stemmed_results):
                if i != j:
                    # Check if there is any overlap between the two sentences' stems
                    if set(stems_i.split()).intersection(set(stems_j.split())):
                        votes[i] += 1

        max_votes = max(votes)
        if max_votes == 0:
            return min(results, key=lambda x: len(x.split()))

        candidate_indices = [i for i, vote in enumerate(votes) if vote == max_votes]
        best_index = min(candidate_indices, key=lambda i: len(results[i].split()))
        return results[best_index], results

    def html(self,img,question,answer,output_var):
        step_name = html_step_name(self.step_name)
        img_str = html_embed_image(img)
        answer = html_output(answer)
        output_var = html_var_name(output_var)
        image_arg = html_arg_name('image')
        question_arg = html_arg_name('question')
        return f"""<div>{output_var}={step_name}({image_arg}={img_str},&nbsp;{question_arg}='{question}')={answer}</div>"""
    
    def html_ensemble(self,answers,answer):
        answers = [html_output(answer) for answer in answers]
        return f"""<div>{','.join(answers)}->{answer}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,question,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        parse_result = parse_step(prog_step.prog_str)
        if 'index' in parse_result['args']:
            index = eval(parse_result['args']['index'])
            img = img[index % len(img)]
        answer, answers = self.predict(img,question)
        prog_step.state[output_var] = answer
        if inspect:
            html_str = self.html(img, question, answer, output_var)
            if len(answers) > 1:
                ensemble_detials = self.html_ensemble(answers, answer)
            html_str = [html_str, ensemble_detials]
            return answer, html_str

        return answer


class LocInterpreter():
    step_name = 'LOC'

    def __init__(self,models,thresh=0.1,nms_thresh=0.5):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = [OwlViTProcessor.from_pretrained(
            model) for model in models['owlvit']] + [Owlv2Processor.from_pretrained(model) for model in models['owlv2']]
        self.model = [OwlViTForObjectDetection.from_pretrained(model).to(self.device) for model in models['owlvit']] + [Owlv2ForObjectDetection.from_pretrained(model).to(self.device) for model in models['owlv2']]
        for model in self.model:
            model.eval()
        self.thresh = thresh
        self.nms_thresh = nms_thresh
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_name = eval(parse_result['args']['object'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_name,output_var

    def normalize_coord(self,bbox,img_size):
        w,h = img_size
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2,w-1)
        y2 = min(y2,h-1)
        return [x1,y1,x2,y2]

    def predict(self,img,obj_name):
        boxes = []
        scores = []
        model_boxes = []
        for i in range(len(self.processor)):
            encoding = self.processor[i](
                text=[[f'a photo of {obj_name}']], 
                images=img,
                return_tensors='pt')
            encoding = {k:v.to(self.device) for k,v in encoding.items()}
            with torch.no_grad():
                outputs = self.model[i](**encoding)
                for k,v in outputs.items():
                    if v is not None:
                        outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
            
            target_sizes = torch.Tensor([img.size[::-1]])
            results = self.processor[i].post_process_object_detection(outputs=outputs,threshold=self.thresh,target_sizes=target_sizes)
            _boxes, _scores = results[0]["boxes"], results[0]["scores"]
            _boxes = _boxes.cpu().detach().numpy().tolist()
            _scores = _scores.cpu().detach().numpy().tolist()

            if len(_boxes) == 0:
                continue

            _boxes, _scores = zip(*sorted(zip(_boxes,_scores),key=lambda x: x[1],reverse=True))
            selected_boxes = []
            selected_scores = []
            for i in range(len(_scores)):
                if _scores[i] > self.thresh:
                    coord = self.normalize_coord(_boxes[i],img.size)
                    selected_boxes.append(coord)
                    selected_scores.append(_scores[i])

            selected_boxes, selected_scores = nms(
                selected_boxes,selected_scores,self.nms_thresh)
            boxes.extend(selected_boxes)
            scores.extend(selected_scores)
            model_boxes.append(selected_boxes)

        if len(boxes) == 0:
            return [], []
        
        if len(self.model) == 1:
            selected_boxes = boxes
            selected_scores = scores
            selected_boxes, selected_scores = zip(*sorted(zip(selected_boxes, selected_scores), key=lambda x: x[1], reverse=True))
            return selected_boxes, selected_boxes

        box_groups = []
        score_groups = []
        # boxes with IoU > 0.8 are in the same group
        for i in range(len(boxes)):
            found = False
            for j in range(len(box_groups)):
                for k in range(len(box_groups[j])):
                    x1 = max(boxes[i][0], box_groups[j][k][0])
                    y1 = max(boxes[i][1], box_groups[j][k][1])
                    x2 = min(boxes[i][2], box_groups[j][k][2])
                    y2 = min(boxes[i][3], box_groups[j][k][3])
                    if x1 < x2 and y1 < y2:
                        overlap_area = (x2 - x1) * (y2 - y1)
                        area1 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                        area2 = (box_groups[j][k][2] - box_groups[j][k][0]) * (box_groups[j][k][3] - box_groups[j][k][1])
                        if overlap_area / (area1 + area2 - overlap_area) > 0.8 and k == len(box_groups[j]) - 1:
                            box_groups[j].append(boxes[i])
                            score_groups[j].append(scores[i])
                            found = True
                            break
                        elif overlap_area / (area1 + area2 - overlap_area) <= 0.8:
                            break
                if found:
                    break
            if not found:
                box_groups.append([boxes[i]])
                score_groups.append([scores[i]])

        # only keep groups with total score > math.ceil(len(self.processor) / 2) * self.thresh
        selected_boxes = []
        selected_scores = []
        for i in range(len(box_groups)):
            total_score = sum(score_groups[i])
            if total_score > math.ceil(len(self.processor) / 2) * self.thresh:
                selected_boxes.append(box_groups[i])
                selected_scores.append(sum(score_groups[i]) / len(score_groups[i]))

        if len(selected_boxes) == 0:
            return [], []
        
        for i in range(len(selected_boxes)):
            min_x = min([j[0] for j in selected_boxes[i]])
            min_y = min([j[1] for j in selected_boxes[i]])
            max_x = max([j[2] for j in selected_boxes[i]])
            max_y = max([j[3] for j in selected_boxes[i]])
            selected_boxes[i] = [min_x, min_y, max_x, max_y]

        # sort by score
        selected_boxes, selected_scores = zip(*sorted(zip(selected_boxes, selected_scores), key=lambda x: x[1], reverse=True))
        return selected_boxes, model_boxes

    def top_box(self,img):
        w,h = img.size        
        return [0,0,w-1,int(h/2)]

    def bottom_box(self,img):
        w,h = img.size
        return [0,int(h/2),w-1,h-1]

    def left_box(self,img):
        w,h = img.size
        return [0,0,int(w/2),h-1]

    def right_box(self,img):
        w,h = img.size
        return [int(w/2),0,w-1,h-1]

    def box_image(self,img,boxes,highlight_best=True):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            if i==0 and highlight_best:
                color = 'red'
            else:
                color = 'blue'

            draw.rectangle(box,outline=color,width=5)

        return img1

    def html(self,img,box_img,output_var,obj_name):
        step_name=html_step_name(self.step_name)
        obj_arg=html_arg_name('object')
        img_arg=html_arg_name('image')
        output_var=html_var_name(output_var)
        img=html_embed_image(img)
        box_img=html_embed_image(box_img,300)
        return f"<div>{output_var}={step_name}({img_arg}={img}, {obj_arg}='{obj_name}')={box_img}</div>"
    
    def html_ensemble(self,box_imgs,box_img):
        box_imgs = [html_embed_image(box_img,300) for box_img in box_imgs]
        box_img = html_embed_image(box_img,300)
        return f"""<div>[{','.join(box_imgs)}]->{box_img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        if obj_name=='TOP':
            bboxes = [self.top_box(img)]
        elif obj_name=='BOTTOM':
            bboxes = [self.bottom_box(img)]
        elif obj_name=='LEFT':
            bboxes = [self.left_box(img)]
        elif obj_name=='RIGHT':
            bboxes = [self.right_box(img)]
        else:
            bboxes, model_boxes  = self.predict(img,obj_name)
        
        bboxes = list(bboxes)
        model_boxes = list(model_boxes)

        parse_result = parse_step(prog_step.prog_str)
        highlight_best = True
        if 'plural' in parse_result['args'] and len(bboxes) > 0 and eval(parse_result['args']['plural'])==True:
            bboxes = [[0, 0, img.size[0] - 1, img.size[1] - 1]] + bboxes
            highlight_best = False

        box_img = self.box_image(img, bboxes,highlight_best=highlight_best)
        box_imgs = None
        if len(self.model) > 1 and obj_name not in ['TOP', 'BOTTOM', 'LEFT', 'RIGHT']:
            box_imgs = [self.box_image(img, boxes, highlight_best=False) for boxes in model_boxes]
        prog_step.state[output_var] = bboxes
        prog_step.state[output_var+'_IMAGE'] = box_img
        if inspect:
            html_str = self.html(img, box_img, output_var, obj_name)
            if box_imgs:
                ensemble_details = self.html_ensemble(box_imgs, box_img)
                html_str = [html_str, ensemble_details]
            return bboxes, html_str

        return bboxes
    
class CapInterpreter():
    step_name = 'CAP'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.model.eval()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,output_var

    def predict(self,img):
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>", image_size=(img.width, img.height))
        return parsed_answer['<MORE_DETAILED_CAPTION>']

    def html(self,img,output,output_var):
        step_name = html_step_name(self.step_name)
        img_str = html_embed_image(img)
        output = html_output(output)
        output_var = html_var_name(output_var)
        image_arg = html_arg_name('image')
        return f"""<div>{output_var}={step_name}({image_arg}={img_str})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        caption = self.predict(img)
        prog_step.state[output_var] = caption
        if inspect:
            html_str = self.html(img, caption, output_var)
            return caption, html_str

        return caption


class CountInterpreter():
    step_name = 'COUNT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return box_var,output_var

    def html(self,box_img,output_var,count):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        box_arg = html_arg_name('bbox')
        box_img = html_embed_image(box_img)
        output = html_output(count)
        return f"""<div>{output_var}={step_name}({box_arg}={box_img})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        box_var,output_var = self.parse(prog_step)
        boxes = prog_step.state[box_var]
        count = len(boxes)
        prog_step.state[output_var] = count
        if inspect:
            box_img = None
            try:
                box_img = prog_step.state[box_var+'_IMAGE']
            except:
                box_img = Image.new('RGB',(100,100),'black')
            html_str = self.html(box_img, output_var, count)
            return count, html_str

        return count


class CropInterpreter():
    step_name = 'CROP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def expand_box(self,box,img_size,factor=1.5):
        W,H = img_size
        x1,y1,x2,y2 = box
        dw = int(factor*(x2-x1)/2)
        dh = int(factor*(y2-y1)/2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0,cx - dw)
        x2 = min(cx + dw,W)
        y1 = max(0,cy - dh)
        y2 = min(cy + dh,H)
        return [x1,y1,x2,y2]

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,box_var,output_var

    def html(self,img,out_img,output_var,box_img):
        img = html_embed_image(img)
        out_img = html_embed_image(out_img,300)
        box_img = html_embed_image(box_img)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        box_arg = html_arg_name('bbox')
        return f"""<div>{output_var}={step_name}({box_arg}={box_img})={out_img}</div>"""
    
    def html_array(self,img,out_imgs,output_var,box_img):
        img = html_embed_image(img)
        out_imgs = [html_embed_image(out_img,300) for out_img in out_imgs]
        box_img = html_embed_image(box_img)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        box_arg = html_arg_name('bbox')
        return f"""<div>{output_var}={step_name}({box_arg}={box_img})={out_imgs}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if 'ARRAY' not in output_var:
            if len(boxes) > 0:
                box = boxes[0]
                box = self.expand_box(box, img.size)
                out_img = img.crop(box)
            else:
                box = []
                out_img = img

            prog_step.state[output_var] = out_img
            if inspect:
                box_img = None
                try:
                    box_img = prog_step.state[box_var+'_IMAGE']
                except:
                    box_img = img
                html_str = self.html(img, out_img, output_var, box_img)
                return out_img, html_str

            return out_img
        else:
            out_imgs = []
            for i,box in enumerate(boxes):
                box = self.expand_box(box, img.size)
                out_img = img.crop(box)
                out_imgs.append(out_img)
            prog_step.state[output_var] = out_imgs
            if inspect:
                box_img = None
                try:
                    box_img = prog_step.state[box_var+'_IMAGE']
                except:
                    box_img = img
                html_str = self.html_array(img, out_imgs, output_var, box_img)
                return out_imgs, html_str

            return out_imgs


class CropRightOfInterpreter(CropInterpreter):
    step_name = 'CROP_RIGHTOF'

    def right_of(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        return [cx,0,w-1,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,index,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[index]
            right_box = self.right_of(box, img.size)
        else:
            w,h = img.size
            box = []
            right_box = [int(w/2),0,w-1,h-1]
        
        out_img = img.crop(right_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = None
            try:
                box_img = prog_step.state[box_var+'_IMAGE']
            except:
                box_img = img
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img


class CropLeftOfInterpreter(CropInterpreter):
    step_name = 'CROP_LEFTOF'

    def left_of(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        return [0,0,cx,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,index,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[index]
            left_box = self.left_of(box, img.size)
        else:
            w,h = img.size
            box = []
            left_box = [0,0,int(w/2),h-1]
        
        out_img = img.crop(left_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = None
            try:
                box_img = prog_step.state[box_var+'_IMAGE']
            except:
                box_img = img
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img


class CropAboveInterpreter(CropInterpreter):
    step_name = 'CROP_ABOVE'

    def above(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        return [0,0,w-1,cy]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,index,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[index]
            above_box = self.above(box, img.size)
        else:
            w,h = img.size
            box = []
            above_box = [0,0,int(w/2),h-1]
        
        out_img = img.crop(above_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = None
            try:
                box_img = prog_step.state[box_var+'_IMAGE']
            except:
                box_img = img
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img

class CropBelowInterpreter(CropInterpreter):
    step_name = 'CROP_BELOW'

    def below(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        return [0,cy,w-1,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,index,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[index]
            below_box = self.below(box, img.size)
        else:
            w,h = img.size
            box = []
            below_box = [0,0,int(w/2),h-1]
        
        out_img = img.crop(below_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = None
            try:
                box_img = prog_step.state[box_var+'_IMAGE']
            except:
                box_img = img
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img

class CropFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_FRONTOF'

class CropInFrontInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONT'

class CropInFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONTOF'

class CropBehindInterpreter(CropInterpreter):
    step_name = 'CROP_BEHIND'


class CropAheadInterpreter(CropInterpreter):
    step_name = 'CROP_AHEAD'

def register_step_interpreters(models_loc, models_vqa):
    return dict(
        LOC=LocInterpreter(models_loc),
        COUNT=CountInterpreter(),
        CROP=CropInterpreter(),
        CROP_RIGHTOF=CropRightOfInterpreter(),
        CROP_LEFTOF=CropLeftOfInterpreter(),
        CROP_FRONTOF=CropFrontOfInterpreter(),
        CROP_INFRONTOF=CropInFrontOfInterpreter(),
        CROP_INFRONT=CropInFrontInterpreter(),
        CROP_BEHIND=CropBehindInterpreter(),
        CROP_AHEAD=CropAheadInterpreter(),
        CROP_BELOW=CropBelowInterpreter(),
        CROP_ABOVE=CropAboveInterpreter(),
        VQA=VQAInterpreter(models_vqa),
        EVAL=EvalInterpreter(),
        RESULT=ResultInterpreter(),
        CAP=CapInterpreter(),
        GET=GetInterpreter()
    )