from argparse import Namespace
from tornado import process
from shlex import quote
import websocket
from websocket import WebSocket
from content_processor import ContentProcessor
from typing import List
import argparse
from tqdm import tqdm
import multiprocessing as mp
import time


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def count_lines(input_path: str) -> int:
    with open(input_path, "r") as f:
        return sum(bl.count("\n") for bl in blocks(f))


def run_marian_server(decoder_path: str, port: int = 10001) -> None:
    out_file = open("marian-server.out", "w")
    err_file = open("marian-server.err", "w")
    process.Subprocess(
        [
            "marian-server",
            "-c",
            quote(decoder_path),
            "-p",
            quote(str(port)),
            "--allow-unk",
            "-b",
            "6",
            "--mini-batch",
            "64",
            "--normalize",
            "0.6",
            "--maxi-batch-sort",
            "src",
            "--maxi-batch",
            "100",
        ],
        stdout=out_file,
        stderr=err_file,
    )


def do_lowercase_all(sentence: str) -> str:
    return sentence.lower()


def do_lowercase_capitals(sentence: str) -> str:
    return " ".join(word[:1] + word[1:].lower() for word in sentence.split(" "))


def translate_batch(
    port: int,
    lowercase_all: bool,
    lowercase_capitals: bool,
    contentprocessor: ContentProcessor,
    lines: List[str],
) -> str:

    translated_lines: List[str] = []
    for line in lines:
        if lowercase_all:
            line = do_lowercase_all(line)
        elif lowercase_capitals:
            line = do_lowercase_capitals(line)
        sentences: List[str] = contentprocessor.preprocess(line)
        ws: WebSocket = websocket.create_connection(f"ws://localhost:{port}/translate")
        ws.send("\n".join(sentences))
        translated_sentences: List[str] = ws.recv().split("\n")
        ws.close()
        translation: List[str] = contentprocessor.postprocess(translated_sentences)
        translated_lines.append(" ".join(translation))

    return "\n".join(translated_lines)


def translate_dataset(
    dataset_path: str,
    output_path: str,
    port: int,
    decoder_path: str,
    source_lang: str,
    target_lang: str,
    sourcebpe: str = None,
    targetbpe: str = None,
    sourcespm: str = None,
    targetspm: str = None,
    block_size: int = 5120,  # bytes
    lowercase_all: bool = False,
    lowercase_capitals: bool = False,
) -> None:
    print("Starting marian server...")
    marian_server = mp.Process(target=run_marian_server, args=(decoder_path, port))
    marian_server.start()
    time.sleep(10.0)

    print("Dataset translation...")
    contentprocessor: ContentProcessor = ContentProcessor(
        source_lang,
        target_lang,
        sourcebpe=sourcebpe,
        targetbpe=targetbpe,
        sourcespm=sourcespm,
        targetspm=targetspm,
    )
    num_lines: int = count_lines(dataset_path)

    with tqdm(total=num_lines) as pbar:
        with open(dataset_path, "r", encoding="utf-8") as input_file:
            with open(output_path, "w+", encoding="utf-8") as output_file:
                lines: List[str] = input_file.readlines(block_size)
                while lines:
                    pbar.update(len(lines))

                    result: str = translate_batch(
                        port,
                        lowercase_all,
                        lowercase_capitals,
                        contentprocessor,
                        lines,
                    )

                    print(result, file=output_file)

                    lines = input_file.readlines(block_size)

    print("Terminating marian server...")
    marian_server.terminate()
    marian_server.join(timeout=1.0)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate a dataset using OpusMT")

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset in the source language. Txt format, one sentence per line",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the datasets in the target language",
    )

    parser.add_argument(
        "--decoder_path",
        type=str,
        required=True,
        help="Path to the decoder (decoder.yml) to use",
    )

    parser.add_argument(
        "--sourcebpe",
        type=str,
        default=None,
        help="Path to the source lang bpe codes to use",
    )

    parser.add_argument(
        "--targetbpe",
        type=str,
        default=None,
        help="Path to the target lang bpe codes to use",
    )

    parser.add_argument(
        "--sourcespm",
        type=str,
        default=None,
        help="Path to the source lang sentence piece model to use",
    )

    parser.add_argument(
        "--targetspm",
        type=str,
        default=None,
        help="Path to the target lang sentence piece model to use",
    )

    parser.add_argument(
        "--source_lang", type=str, required=True, help="Source lang ID",
    )

    parser.add_argument(
        "--target_lang", type=str, required=True, help="Source lang ID",
    )

    parser.add_argument(
        "--port", type=int, default=10001, help="Port where the server is listening",
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=5120,
        help="Number of bytes to read from the dataset each iteration."
        " Larger block_size = Lager batch size",
    )

    lowercase_group = parser.add_mutually_exclusive_group()

    lowercase_group.add_argument(
        "--lowercase_all", action="store_true", help="Lowercase all the words",
    )

    lowercase_group.add_argument(
        "--lowercase_capitals",
        action="store_true",
        help="For every word, lowercase all the letters except the first one",
    )

    args: Namespace = parser.parse_args()

    translate_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        port=args.port,
        decoder_path=args.decoder_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        sourcebpe=args.sourcebpe,
        targetbpe=args.targetbpe,
        sourcespm=args.sourcespm,
        targetspm=args.targetspm,
        block_size=args.block_size,
        lowercase_all=args.lowercase_all,
        lowercase_capitals=args.lowercase_capitals,
    )
