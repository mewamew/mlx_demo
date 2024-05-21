from mlx_lm import load, generate
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_chat_template(messages, tokenizer):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def load_model():
    model_list = [
        "mlx-community/Qwen1.5-7B-Chat-4bit",
        "mlx-community/Qwen1.5-72B-4bit",
    ]

    print("可用的模型列表:")
    for i, model_name in enumerate(model_list, start=1):
        print(f"{i}. {model_name}")

    selected_model = input("请输入要使用的模型编号: ")
    selected_model_index = int(selected_model) - 1

    if 0 <= selected_model_index < len(model_list):
        selected_model_name = model_list[selected_model_index]
        print(f"正在加载模型: {selected_model_name}")
        model, _ = load(selected_model_name)
    else:
        print("无效的模型编号，请重新运行程序并选择有效的模型编号。")
        exit(1)
    return model

model = load_model()
auto_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-72B-Chat")


#支持多轮对话哦
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]
while True:
    user_input = input("请输入您的问题（输入'退出'结束对话）: ")
    if user_input.lower() == '退出':
        break
    
    messages.append({"role": "user", "content": user_input})
    prompt = apply_chat_template(messages, auto_tokenizer)

    response = generate(
                    model, 
                    auto_tokenizer, 
                    prompt=prompt, 
                    max_tokens=1024,
                    verbose=False)
    print(f"\033[97m{response}\033[0m")
    messages.append({"role": "assistant", "content": response})
