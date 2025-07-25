from email.mime import audio
import os
import tempfile
import traceback
from pathlib import Path

from tokenization_mio import MIOTokenizer
from transformers import AutoModelForCausalLM
import torch
import argparse
import json
from tqdm import tqdm
# 添加导入utils模块
from utils import extract_frames

tempfile.tempdir = "/share/nlp/tuwenming/projects/HAVIB/tmp"


# 本文件用于保存所有的定义
maic_cls_list = ['bus', 'hair-dryer', 'pipa', 'man', 'ambulance', 'razor', 'harp', 'tabla', 'bass', 'handpan', 
        'girl', 'sitar', 'car', 'lion', 'guitar', 'vacuum-cleaner', 'cat', 'mower', 'helicopter', 'boy', 'drum', 
        'keyboard', 'tuba', 'saw', 'flute', 'cello', 'woman', 'gun', 'accordion', 'violin', 'clarinet', 'erhu', 
        'saxophone', 'guzheng', 'dog', 'baby', 'horse', 'male', 'wolf', 'bird', 'ukulele', 'piano', 'female', 
        'marimba', 'not sure', 'no available option']

mvic_cls_list = ['sushi', 'banana', 'cake', 'butterfly', 'bird', 'microphone', 'hamburger', 'pineapple', 
        'man', 'book', 'sunglasses', 'goat', 'tie', 'cabinetry', 'motorcycle', 'drawer', 'strawberry', 
        'sheep', 'pasta', 'parrot', 'bull', 'table', 'penguin', 'watch', 'pillow', 'shellfish', 'kangaroo', 
        'flower', 'paddle', 'rocket', 'helicopter', 'bus', 'mushroom', 'bee', 'tree', 'boat', 'saxophone', 
        'football', 'lizard', 'violin', 'dog', 'cucumber', 'cello', 'airplane', 'horse', 'drum', 'box', 
        'rabbit', 'car', 'door', 'orange', 'shelf', 'camera', 'poster', 'lemon', 'cat', 'fish', 'bread', 
        'piano', 'apple', 'glasses', 'bicycle', 'truck', 'deer', 'woman', 'wheelchair', 'cheese', 'chair', 
        'plate', 'tomato', 'bed', 'starfish', 'balloon', 'bottle', 'crab', 'beer', 'frog', 'shrimp', 'tower', 
        'guitar', 'pig', 'peach', 'train', 'pumpkin', 'elephant', 'jellyfish', 'parachute', 'monkey', 'flag',
        'not sure', 'no available option']


pmp_avl_ans_format = "answer={'category1_id1': '[x_min, y_min, x_max, y_max]', 'category2_id2': '[x_min, y_min, x_max, y_max]'}"
avl_cls_list = ['dog', 'clarinet', 'banjo', 'cat', 'guzheng', 'tree', 'lion', 'tuba', 
        'ukulele', 'flute', 'piano', 'person', 'violin', 'airplane', 'bass', 'pipa', 
        'trumpet', 'accordion', 'saxophone', 'car', 'lawn-mower', 'cello', 'bassoon', 
        'horse', 'guitar', 'erhu', 'not sure', 'no available option']
prompt_avl = f"""
        ctaegories list: {avl_cls_list}
        (1) There may be multiple sounding instances, you can choose instance categories from the given categories list.
        (2) The naming format for instances is: category_id. Instance IDs start from 1, e.g., male_1, dog_2, dog_3, cat_4. 
        (3) The bbox format is: [x_min, y_min, x_max, y_max], where x_min, y_min represent the coordinates of the top-left corner. 
        (4) The bbox values should be normalized into the range of 0 and 1, e.g., [0.1, 0.12, 0.26, 0.14].
        Do not explain, you must strictly follow the format: {pmp_avl_ans_format}
    """

prompt_avlg = """
        Please output the answer in a format that strictly matches the following example, do not explain:
        answer={'frame_0': [x0_min, y0_min, x0_max, y0_max], 'frame_1': None, ..., 'frame_9': [x9, y9, w9, h9]}.
        Note, 
        (1) x_min, y_min represent the coordinates of the top-left corner, while x_max, y_max for the bottom_right corner.
        (2) The bbox values should be normalized into the range of 0 and 1, e.g., [0.1, 0.12, 0.26, 0.14]. 
        (3) Frames should be ranged from frame_0 to frame_9.
    """

avqa_cls_list = ['ukulele', 'cello', 'clarinet', 'violin', 'bassoon', 'accordion', 'banjo', 'tuba', 'flute', 'electric_bass', 'bagpipe', 
        'drum', 'congas', 'suona', 'xylophone', 'saxophone', 'guzheng', 'trumpet', 'erhu', 'piano', 'acoustic_guitar', 'pipa', 'not sure', 'no available option']

havib_constants = {
    'L1_LAQA': {
        'options_sound_clarity': ['first', 'last', 'same', 'not sure'],
        'options_sound_order': ['sound', 'noise', 'not sure'],
        'options_sound_volume': ['first', 'last', 'same', 'not sure'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LIQA': {
        'get_from_background_binary': ['yes', 'no', 'not sure'],
        'get_from_image_binary': ['yes', 'no', 'not sure'],
        'get_from_foreground_binary': ['yes', 'no', 'not sure'],
        'get_from_image_triple': ['blurred', 'normal', 'clear', 'not sure'],
        'get_from_3d-task1': ['center', 'left', 'right', 'not sure'],
        'get_from_3d-task2': ['cone', 'cube', 'cylinder', 'cuboid', 'no available option', 'not sure'],
        # 'get_from_3d-task3': [0, 1, 2, 3, 4, 5, 6],
        'get_from_space_hard': ['center', 'top left', 'top center', 'top right', 'bottom left', 'bottom center', 'bottom right', 'no available option', 'not sure'],
        'get_from_color': ['blue', 'green', 'red', 'puprle', 'yellow', 'no available option', 'not sure'],
        'get_yes_no': ['yes', 'no', 'not sure'],
        # 'get_lines_count': [0, 1, 2, 3, 4],
        'get_lines_direction': ['horizontal', 'vertical', 'inclined', 'not sure'],
        'get_from_space_easy_area': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'get_from_space_easy_bbrightness': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LVQA': {
        'which_object': ['square', 'circle', 'triangle', 'not sure', 'no available option', 'not sure'],
        'what_shape': ['Triangular pyramid', 'Cone', 'Cube', 'Sphere', 'None', 'not sure'],
        # 'how_many': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'what_movement_2d': ['horizontal', 'inclined', 'vertical', 'no movenment', 'None', 'not sure'],
        'what_movement_3d': ['Rotation', 'Shrinking', 'Translation', 'Enlarging', 'None', 'not sure'],
        'what_surface': ['Rough', 'Moderate', 'Smooth', 'None', 'not sure'],
        'spacial_change': ['Bottom-left to top-right', 'Bottom-right to top-left', 'Top-left to bottom-right', 'Top-right to bottom-left', 'None', 'not sure', 'No movement',],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L2_MAIC': {
        'maic_cls_list': maic_cls_list,
        'prompt_maic': "There may be one or more sound-emitting objects in the provided audio. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n"
    },

    'L2_MVIC': {
        'mvic_cls_list': mvic_cls_list,
        'prompt_mvic': "There may be one or more visible objects in the provided image. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n Possible categoris are in the list: mvic_cls_list"
    },

    'L3_AVH': {
        'prompt_avh': "Please answer the question based on the given audio and video.",
        'avh_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_VAH': {
        'prompt_vah': "Please answer the question based on the given audio and video.",
        'vah_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVL': {
        'prompt_avl': prompt_avl,
        'avl_cls_list': avl_cls_list,
    },

    'L3_AVM': {
        'prompt_avm': 'Please answer the question based on the given audio and video.',
        'avm_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVR': {
        'prompt_avr': "Please output the indices of the images list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
    },

    'L3_VAR': {
        'prompt_var': "Please output the indices of the wavs list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
    },

    'L4_AVC': {

    },

    'L4_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L4_AVQA': {
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },

    'L5_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L5_AVQA': {
        'avqa_cls_list': avqa_cls_list,
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },

    'L6_AVSQA': {
        'avsqa_options_list_object': ['cube', 'pyramid', 'cone', 'sphere', 'no availabel option', 'not sure'],
        'avsqa_options_list_color': ['red', 'blue', 'white', 'black', 'gray', 'green', 'no availabel option', 'not sure'],
        'prompt_avsqa': "You may choose multiple options; separated by semicolons. Your answer should be enclosed within ##, for example: #your_ans#.",
        'options_yes_no': ['yes', 'no', 'not sure'],
    }
}


def get_real_path(task_path: str, src_path: str) -> str:
    """传入taskpath和一些文件的path，构造文件的真实path

    Args:
        task_path (str): task path
        src_path (str): 每个文件的path

    Returns:
        str: 文件的真实path
    """
    temp_path = os.path.join(task_path, src_path)
    return os.path.normpath(temp_path)

def get_real_options_or_classes(d: dict) -> str:
    """Replace pseudo-options with real options text."""
    opts = d['input']['question'].get('options')
    if opts in havib_constants.get(d['task'], {}):
        opts = havib_constants[d['task']][opts]
    if opts:
        label = 'semantic categories' if 'cls' in opts else 'options'
        return f"Available {label} are: {opts}"
    return ''

def get_real_prompt(d: dict) -> str:
    """Replace pseudo-prompt with real prompt text."""
    prm = d['input']['question'].get('prompt')
    if prm in havib_constants.get(d['task'], {}):
        prm = havib_constants[d['task']][prm]
    return prm or ''

def get_real_input(d: dict) -> str:
    """Concatenate prompt, options, and question text into one input string."""
    prompt = get_real_prompt(d)
    options = get_real_options_or_classes(d)
    question = d['input']['question']['text'] or ''
    # 去掉多余的句点
    parts = [p for p in (prompt, options, question) if p]
    return " ".join(parts)


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="HumanOmni Inference Script")
    parser.add_argument(
        "--model_path", type=str, default='/share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct', help="Path to the model"
    )

    parser.add_argument(
        "--task_path",
        type=str,
        required=True,
        help="Path to the task folder containing data.json and media files",
    )

    args = parser.parse_args()
    
    model_name_or_path = args.model_path # TODO: change your model path here.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mio_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    ).half().eval()
    mio_tokenizer = MIOTokenizer(model_name_or_path, device)
    
    # 设置生成配置
    generation_config = {
        "num_beams": 5,
        "do_sample": False,  # False if num_beams > 1 else True
        "temperature": 1.0,
        "top_p": 0.7,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
        "length_penalty": 1.0,
        "top_k": 0,
        "pad_token_id": mio_tokenizer.tokenizer.pad_token_id,
        "eos_token_id": 7 if "Instruct" in model_name_or_path else mio_tokenizer.tokenizer.eos_token_id,
        "num_return_sequences": 1,
        "guidance_scale": None,
    }
    
    task_path = args.task_path
    task_name = f"L{task_path.rsplit('/', 1)[0][-1]}_{task_path.rsplit('/', 1)[-1]}"
    model_name = args.model_path.split('/')[-1]
    save_prediction_json = f'/share/nlp/tuwenming/projects/HAVIB/eval/user_outputs/{model_name}/tasks/{task_name}.json'
    os.makedirs(os.path.dirname(save_prediction_json), exist_ok=True)
    print('>>> save res to:', save_prediction_json)
    
    
    data_json_path = os.path.join(task_path, "data.json")
    with open(data_json_path, "r", encoding='utf-8') as f:
        raw_data = json.load(f)
    print(">>>Finished load raw data...")
    parsed_data = []
    for item in raw_data:
        inp = item.get('input', {})
        question = inp.get('question', {})
        entry = {
            'id': item.get('id'),
            'task': item.get('task'),
            'subtask': item.get('subtask', None),
            'text': get_real_input(item),
            'audio_list': inp.get('audio_list', None),
            'image_list': inp.get('image_list', None),
            'video': inp.get('video', None)
        }
        parsed_data.append(entry)

    print(">>>Finished parse raw data...")    
    
    predictions = []
    
    for data in tqdm(parsed_data):
        _id = data['id']
        _task = data['task']
        _subtask = data['subtask']
        text = data['text']
        audio_list = (
            [get_real_path(task_path, p) for p in data["audio_list"]]
            if data["audio_list"] else None
        )
        image_list = (
            [get_real_path(task_path, p) for p in data["image_list"]]
            if data["image_list"] else None
        )
        video = (
            get_real_path(task_path, data['video'])
            if data['video'] else None
        )
        print(f">>> text input=:{text}")
        
        output = None
        
        try:
            # 根据输入内容判断是哪种任务，然后调用不同的函数进行推理
            batch_image_paths = None
            batch_speech_paths = None
            conversations = None
            
            # 处理视频输入 - 如果有视频，提取帧
            if video:
                video_frames_dir = os.path.join(tempfile.tempdir, f"frames_{_id}")
                video_frames = extract_frames(video, video_frames_dir, force_uniform=True)
                # 将视频帧添加到图像列表中
                if image_list:
                    image_list.extend(video_frames)
                else:
                    image_list = video_frames
            
            # case1: audio + text
            if audio_list and not image_list:
                print(">>> Processing case1: audio + text")
                batch_speech_paths = [audio_list]
                conversations = [
                    [{"role": "user", "content": f"<speech_placeholder_0>\n{text}"}]
                ]
                
            # case2: image + text
            elif image_list and not audio_list:
                print(">>> Processing case2: image + text")
                batch_image_paths = [image_list]
                # 构造图像占位符
                image_placeholders = "".join([f"<image_placeholder_{i}>" for i in range(len(image_list))])
                conversations = [
                    [{"role": "user", "content": f"{image_placeholders}\n{text}"}]
                ]
                
            # case3: video + text (已在上面处理，视频转换为图像列表)
            # 这种情况会被case2捕获
            
            # case4: video + audio + text
            # case5: image list + audio + text  
            # case6: image + audio list + text
            elif image_list and audio_list:
                print(">>> Processing case4/5/6: multimodal with image and audio")
                batch_image_paths = [image_list]
                batch_speech_paths = [audio_list]
                
                # 构造占位符
                image_placeholders = "".join([f"<image_placeholder_{i}>" for i in range(len(image_list))])
                speech_placeholders = "".join([f"<speech_placeholder_{i}>" for i in range(len(audio_list))])
                
                conversations = [
                    [{"role": "user", "content": f"{image_placeholders}{speech_placeholders}\n{text}"}]
                ]
                
            # 纯文本情况
            else:
                print(">>> Processing pure text")
                conversations = [
                    [{"role": "user", "content": text}]
                ]
            
            # 执行推理
            if "Instruct" in model_name_or_path:
                # 使用Instruct模型的chat模板
                inputs = mio_tokenizer.apply_chat_template(
                    conversations, 
                    batch_image_paths=batch_image_paths, 
                    batch_speech_paths=batch_speech_paths,
                    mode='std',  # 默认使用标准模式
                    padding=True, 
                    truncation=True, 
                    max_length=2048, 
                    return_tensors='pt'
                )
            else:
                # 使用Base模型
                # 将对话转换为单个prompt
                prompt = conversations[0][0]["content"]
                inputs = mio_tokenizer(
                    [prompt], 
                    batch_image_paths=batch_image_paths, 
                    batch_speech_paths=batch_speech_paths, 
                    padding=True, 
                    truncation=True, 
                    max_length=3000, 
                    return_tensors='pt'
                )
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # 生成输出
            with torch.no_grad():
                model_output = mio_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
            
            # 解码输出
            generated_sequences, decoded_image_paths, decoded_speech_paths = (
                mio_tokenizer.detokenize(
                    model_output,
                    output_image_dir=f"generated_images/{_id}",
                    output_speech_dir=f"generated_speeches/{_id}",
                    extract_assistant=True if "Instruct" in model_name_or_path else False,
                    save_images=False,
                    save_speeches=False
                )
            )
            
            # 获取生成的文本
            if generated_sequences:
                output = generated_sequences[0].strip()
            else:
                output = "Failed to generate response"
                
        except Exception as e:
            print(f">>> Error processing sample {_id}: {str(e)}")
            traceback.print_exc()
            output = f"Error: {str(e)}"
        
        pred_record = {
            "task": _task,
            "subtask": _subtask,
            "id": _id,
            "predict": output,
        }
        predictions.append(pred_record)
        print('>>> ans=:', pred_record)
        
    with open(save_prediction_json, 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)