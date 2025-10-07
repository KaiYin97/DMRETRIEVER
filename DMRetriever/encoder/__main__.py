# DMRetriever/encoder/__main__.py
import os
import json
from dataclasses import asdict
from transformers import HfArgumentParser
from DMRetriever.core import ModelArguments, DataArguments, TrainingArguments
from DMRetriever.encoder import EncoderRunner


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "config_args.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"model_args": asdict(model_args), "data_args": asdict(data_args), "training_args": asdict(training_args)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    runner = EncoderRunner(model_args=model_args, data_args=data_args, training_args=training_args)
    runner.run()


if __name__ == "__main__":
    main()
