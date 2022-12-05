import torch
import faiss
import japanese_clip as ja_clip

def load_models():
    clip, preprocess = ja_clip.load(
        "rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip")
    tokenizer = ja_clip.load_tokenizer()
    return {
        'clip': clip,
        'preprocess': preprocess,
        'tokenizer': tokenizer,
    }


def text2vectors(texts, models):
    encodings = ja_clip.tokenize(
        texts=texts,
        tokenizer=models['tokenizer'],
    )
    with torch.no_grad():
        vectors = models['clip'].get_text_features(**encodings)
    return vectors.detach().numpy()


def load_image_list(image_list_path):
    with open(image_list_path) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def load_index(index_path):
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    return index


def search(query, index, k=3):
    _, searched_index = index.search(query, k)
    return searched_index


def main():
    models = load_models()
    image_list = load_image_list('output/image_list.txt')
    index = load_index('output/index.faiss')
    texts = ['黒い犬']
    query = text2vectors(texts, models)
    result = search(query, index)
    for img_idx in result[0]:
        print(image_list[img_idx])


if __name__ == '__main__':
    main()
