from __future__ import annotations

import argparse

from src.emotion_model import GoEmotionsClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default=(
            "I keep replaying what happened and blaming myself. "
            "I wish I had handled it differently."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()

    classifier = GoEmotionsClassifier(device=args.device)
    result = classifier.predict(args.text, top_k=10)

    print("Text:")
    print(args.text)
    print()
    print("Top predictions:")
    for item in result.top_k:
        print(f"{item['label']:15s} {item['score']:.4f}")
    print()
    print("Selected at threshold:")
    for item in result.selected:
        print(f"{item['label']:15s} {item['score']:.4f}")


if __name__ == "__main__":
    main()
