import time
import streamlit as st
from PIL import Image
import glob
import itertools
import os
from PIL import Image
import torch
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


def create_dataset(dataset_dir, models, batchsize=50):

    image_path_list = glob.glob(os.path.join(dataset_dir, '*.jpg'))
    vector_list = []
    idx = 0
    while True:
        image_path_batch = list(itertools.islice(image_path_list, idx, idx + batchsize))
        if len(image_path_batch) == 0:
            break
        print('Get vectors from image {} to {}...'.format(idx, idx + batchsize))
        idx += batchsize
        images = [Image.open(image_path) for image_path in image_path_batch]
        processed = torch.cat([models['preprocess'](img).unsqueeze(0) for img in images], dim=0)
        with torch.no_grad():
            vector_list.append(models['clip'].get_image_features(processed))
    image_path_list = [f'{pl}\n' for pl in image_path_list]
    vectors = torch.cat(vector_list, dim=0)
    return {
        'path_list': image_path_list,
        'vectors': vectors.detach().numpy(),
    }


def main():
    st.set_page_config(layout="wide")
    with st.spinner('Loading...'):
        models = load_models()
        image_list = load_image_list('output/image_list.txt')
        index = load_index('output/index.faiss')

    st.title('Image search by Japanese-CLIP')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.form('text_form'):
            search_text = st.text_input('Search Text', '黒い犬')
            button = st.form_submit_button('Search Image')

    if not button or search_text == '':
        st.stop()

    t2v_start = time.time()
    query = text2vectors([search_text], models)
    search_start = time.time()
    searched_index = search(query, index)[0]
    search_end = time.time()
    results = [image_list[idx] for idx in searched_index]
    st.write('Text to Vector: {:.4f}[s]'.format(search_start - t2v_start))
    st.write('Search        : {:.4f}[s]'.format(search_end - search_start))

    cols = [col2, col3, col4]
    for i, img_path in enumerate(results):
        with cols[i]:
            img = Image.open(img_path)
            st.image(img, caption=img_path, use_column_width='always')


if __name__ == '__main__':
    main()