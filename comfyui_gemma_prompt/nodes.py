import gc
import os
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = None
TOKENIZER = None
CURRENT_MODEL_PATH = None


def load_model(model_path):
    global MODEL, TOKENIZER, CURRENT_MODEL_PATH

    if MODEL is not None and CURRENT_MODEL_PATH == model_path:
        return TOKENIZER, MODEL

    if MODEL is not None:
        del MODEL
        MODEL = None
    if TOKENIZER is not None:
        del TOKENIZER
        TOKENIZER = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    print(f"[GemmaPrompt] 正在加载模型: {model_path}")
    TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    CURRENT_MODEL_PATH = model_path
    print("[GemmaPrompt] 模型加载完成!")
    return TOKENIZER, MODEL


class GemmaPromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "/opt/aigc_apps/models/zh/Qwen2.5-7B-Instruct",
                    "multiline": False,
                }),
                "template_prompt": ("STRING", {
                    "default": "请根据以下用户需求，生成一个高质量的AI绘画提示词。\n要求：\n1. 提示词要详细描述画面内容、风格、光线、构图\n2. 使用英文输出（Stable Diffusion格式）\n3. 包含质量标签如 masterpiece, best quality 等\n\n用户需求：{user_input}",
                    "multiline": True,
                }),
                "user_prompt": ("STRING", {
                    "default": "一只在月光下奔跑的白色狐狸",
                    "multiline": True,
                }),
                "language": (["中文", "English"], {
                    "default": "中文",
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "label_on": "开启思考",
                    "label_off": "关闭思考",
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_prompt",)
    FUNCTION = "generate"
    CATEGORY = "GemmaPrompt"

    def generate(self, model_path, template_prompt, user_prompt, language,
                 enable_thinking, max_new_tokens, temperature):
        if not template_prompt.strip():
            return ("错误: 模板提示词不能为空",)
        if not user_prompt.strip():
            return ("错误: 用户提示词不能为空",)

        try:
            tokenizer, model = load_model(model_path)
        except Exception as exc:
            traceback.print_exc()
            return (f"模型加载失败: {type(exc).__name__}: {str(exc)}",)

        if language == "中文":
            lang_instruction = "请用中文输出。"
        else:
            lang_instruction = "Please output in English."

        system_content = (
            f"你是一个专业的提示词生成助手。\n"
            f"请严格按照以下模板格式来生成提示词：\n\n"
            f"---模板开始---\n{template_prompt}\n---模板结束---\n\n"
            f"将模板中的 {{user_input}} 替换为用户的具体需求来生成最终提示词。\n"
            f"{lang_instruction}\n"
            f"只输出最终生成的提示词，不要输出任何解释或其他内容。"
        )

        final_request = template_prompt.replace(
            "{user_input}",
            user_prompt,
        )

        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": final_request,
            },
        ]

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
            ).to(model.device)
            input_length = inputs["input_ids"].shape[-1]

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = 0.9

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            return (response.strip(),)

        except Exception as exc:
            traceback.print_exc()
            return (f"生成失败: {type(exc).__name__}: {str(exc)}",)


NODE_CLASS_MAPPINGS = {
    "GemmaPromptGenerator": GemmaPromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GemmaPromptGenerator": "Qwen2.5 Prompt Generator",
}