import os
import time
import random
from model import build_model, NGramModel


def generate_paragraph(model, start_context, length=20, delay=0.25):
    generated = list(start_context)
    context = tuple(start_context)

    for _ in range(length):
        predictions = model.predict(context)
        if not predictions:
            break

        predictions.sort(key=lambda x: list(x.values())[0], reverse=True)
        next_word = list(predictions[0].keys())[0]
        generated.append(next_word)
        # Update context by shifting and adding the new word
        context = (context + (next_word,))[1:]

        os.system("cls")
        print(" ".join(generated))
        time.sleep(delay)

    return generated


if __name__ == "__main__":
    model, tokens = build_model(n=3, smoothing=False)
    start_context  = ("the", "pride")
    if start_context not  in model.contexts:
        start_context = random.choice(list(model.contexts))

    generated_words = generate_paragraph(model, start_context, length=10, delay=0.25)
    os.system("cls")
    print("Final paragraph:")
    print(" ".join(generated_words))
