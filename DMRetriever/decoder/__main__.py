# DMRetriever/decoder/__main__.py
from transformers import HfArgumentParser
from DMRetriever.core import DataArguments, TrainingArguments
from DMRetriever.decoder import DecoderModelArguments, DecoderRunner


def main():
    parser = HfArgumentParser((DecoderModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    runner = DecoderRunner(model_args=model_args, data_args=data_args, training_args=training_args)
    runner.run()


if __name__ == "__main__":
    main()
