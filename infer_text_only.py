import torch
print(torch.__version__)

import torchaudio

print(torchaudio.__version__)

from tokenization_mio import MIOTokenizer
from transformers import AutoModelForCausalLM



model_name_or_path = '/share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct' # TODO: change your model path here.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mio_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
).half().eval()
mio_tokenizer = MIOTokenizer(model_name_or_path, device)

generation_config = {
    "num_beams": 5,
    "do_sample": False,  # False if num_beams > 1 else True
    "temperature": 1.0,
    "top_p": 0.7,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512, # TODO: please reduce this value to speed up the inference when YOU ARE NOT GENERATING SPEECH.
    "length_penalty": 1.0,
    "top_k": 0,
    "pad_token_id": mio_tokenizer.tokenizer.pad_token_id,
    "eos_token_id": 7 if "Instruct" in model_name_or_path else mio_tokenizer.tokenizer.eos_token_id, # 7 is the <|im_end|> token.
    "num_return_sequences": 1,
    "guidance_scale": None,
}

# text-only prompts for base model.
text_prompts_base = [
    "Question: What's the most beautiful thing you've ever seen? Answer:",
    "Question: 1+1=? Answer:",
    "Question: What's the capital of France? Answer:",
]


# text-only prompts for Instruct model.
text_conversations_inst = [
    [
        {"role": "user", "content": "What's the most beautiful thing you've ever seen?"},
    ],
    [
        {"role": "user", "content": "What's the capital of France?"},
    ],
    [
        {"role": "user", "content": "1+1=?"},
    ],
]  # There are three conversations.



# TODO: learn the following code to generate responses for your own needs.
if "Base" in model_name_or_path:
    inputs = mio_tokenizer(text_prompts_base, batch_image_paths=None, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    # inputs = mio_tokenizer(img_cap_prompts_base, batch_image_paths=batch_image_paths_img_cap_base, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    # inputs = mio_tokenizer(imagen_prompts_base, batch_image_paths=None, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    # inputs = mio_tokenizer(asr_prompts_base, batch_image_paths=None, batch_speech_paths=batch_speech_paths_asr_base, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    # inputs = mio_tokenizer(tts_prompts_base, batch_image_paths=None, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    # inputs = mio_tokenizer(vid_cap_prompts_base, batch_image_paths=batch_image_paths_vid_cap_base, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    # inputs = mio_tokenizer(vid_gen_prompts_base, batch_image_paths=None, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
elif "Instruct" in model_name_or_path:
    # for standard modes (text generation, image generation, video generation)
    inputs = mio_tokenizer.apply_chat_template(
        text_conversations_inst, batch_image_paths=None, batch_speech_paths=None,
        mode='std', padding=True, truncation=True, max_length=2048, return_tensors='pt'
    )
    # # for voice modes (speech generation, including tts)
    # inputs = mio_tokenizer.apply_chat_template(
    #     tts_conversations_inst, batch_image_paths=None, batch_speech_paths=None, mode='voice', padding=True,
    #     truncation=True, max_length=2048, return_tensors='pt'
    # )

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

output = mio_model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    **generation_config
)

print("======output======")
generated_sequences, decoded_image_paths, decoded_speech_paths = (
    mio_tokenizer.detokenize(output,
                             output_image_dir="generated_images/temp", # TODO: change the output_image_dir here.
                             output_speech_dir="generated_speeches/temp", # TODO: change the output_speech_dir here.
                             extract_assistant=True,  # TODO: set as True only if you are using the Instruct model.
                             save_images=False, # TODO
                             save_speeches=False))  # TODO: make sure to set save_speeches=False when YOU ARE NOT GENERATING SPEECH.
for i, string in enumerate(generated_sequences):
    print(f"{i}-th response:\n{string}")
