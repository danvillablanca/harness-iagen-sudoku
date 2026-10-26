from qwen import load_model


if __name__ == "__main__":
    model = load_model()
    prompt = """
You got four digits: "1", "2", "3", "4".
Please fill the missing digit in the list [1, 2, _, 4].
Return only the complete list.
"""
    print(model.generate(prompt))