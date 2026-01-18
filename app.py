import time
import pathlib
from pathlib import Path
import streamlit as st
from streamlit_image_comparison import image_comparison
from streamlit_image_coordinates import streamlit_image_coordinates

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
from custom_unet import SelfMadeUNet

if "best_time_models" not in st.session_state:
    st.session_state.best_time_models = {}
if "best_miou_models" not in st.session_state:
    st.session_state.best_miou_models = {}
if "best_per_iou_models" not in st.session_state:
    st.session_state.best_per_iou_models = {}

colors_map = {
  0:  '#FF69B4',
  1:  '#FF7F00',
  2:  '#804080',
  3:  '#DC143C',
  4:  '#FFFFFF',
  5:  '#6B8E23',
  6:  '#1E90FF',
  7:  '#FF4500',
  8:  '#4682B4',
  9:  '#464646',
  10: '#999999',
  11: '#FFD700',
  12: '#ADFF2F',
  13: '#00008E',
  14: '#000000',
}

class_names = {
  0:  'Животные',
  1:  'Преграды',
  2:  'Плоскости (дорога)',
  3:  'Люди',
  4:  'Разметка',
  5:  'Природа',
  6:  'Объекты',
  7:  'Баннеры',
  8:  'Небо',
  9:  'Здания',
  10: 'Поддержки',
  11: 'Светофоры',
  12: 'Дорожный знак',
  13: 'Транспортные средства',
  14: 'Пустота',
}

best_time_models = st.session_state.best_time_models
best_miou_models = st.session_state.best_miou_models
best_per_iou_models = st.session_state.best_per_iou_models

class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        # Определите слои модели

    def forward(self, x):
        return self.model(x)
    

       
        

# @st.cache_data
def prediction(image, orig_mask=None):
    global selected_model, arg_example

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    start_time = time.perf_counter()

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to('cpu')
    with torch.no_grad():
        pred = model(image_tensor).squeeze(0)
    
    inf_time = time.perf_counter() - start_time

    best_time_models.setdefault(arg_example, {})
    best_time_models[arg_example].setdefault('time', 100)
    best_time_models[arg_example].setdefault('name', '')
    diff_time = best_time_models[arg_example]['time'] - inf_time
    if best_time_models[arg_example]['time'] > inf_time:
        best_time_models[arg_example]['time'] = inf_time
        best_time_models[arg_example]['name'] = selected_model

    col1, col2, col3 = st.columns(3)
    if diff_time < 90:
        model_name = best_time_models[arg_example]['name']
        col1.metric('Время предсказания', f'{inf_time:.2f} сек.',  f'{diff_time:.2f} | ...{model_name[-10:]}')
    else:
        col1.metric('Время предсказания', f'{inf_time:.2f} сек.')

    num_classes = pred.shape[0]
    mask = pred.argmax(dim=0).numpy().astype(np.uint8)

    if orig_mask != None:
        compare_mask = np.zeros_like(mask)
        compare_mask[mask == orig_mask] = 1
        compare_mask[orig_mask == 14] = 2 

        tp, fp, fn, tn = smp.metrics.get_stats(torch.from_numpy(mask), orig_mask,
                                           mode="multiclass",
                                           num_classes=14,
                                           ignore_index=14)

        iou = smp.metrics.iou_score(tp.sum(0), fp.sum(0), fn.sum(0), tn.sum(0))
        animal_iou = iou[0]
        miou = iou.mean()

        mask = compare_mask
        palette = np.array([
            [200, 0, 0],
            [0, 200, 0],
            [40, 40, 40],
        ], dtype=np.uint8)
        
        best_miou_models.setdefault(arg_example, {})
        best_per_iou_models.setdefault(arg_example, {})
        best_miou_models[arg_example].setdefault('miou', 0)
        best_per_iou_models[arg_example].setdefault('per_iou', 0)
        best_miou_models[arg_example].setdefault('name', '')
        best_per_iou_models[arg_example].setdefault('name', '')

        diff_miou = miou - best_miou_models[arg_example]['miou']
        diff_per_iou = animal_iou - best_per_iou_models[arg_example]['per_iou']

        if best_miou_models[arg_example]['miou'] < miou:
            best_miou_models[arg_example]['miou'] = miou
            best_miou_models[arg_example]['name'] = selected_model
        if best_per_iou_models[arg_example]['per_iou'] < animal_iou:
            best_per_iou_models[arg_example]['per_iou'] = animal_iou
            best_per_iou_models[arg_example]['name'] = selected_model


        if diff_miou != miou:
            model_name = best_miou_models[arg_example]['name']
            model_per_iou_name = best_miou_models[arg_example]['name']
            col2.metric('Cредний mIoU', f'{miou:.3f}', f'{diff_miou:.3f} | ...{model_name[-14:]}')
            col3.metric('IoU по животным', f'{animal_iou:.3f}', f'{diff_per_iou:.3f} | ...{model_per_iou_name[-14:]}')
        else:            
            col2.metric('Cредний mIoU', f'{miou:.3f}')
            col3.metric('IoU по животным', f'{animal_iou:.3f}')
        
    else:
        
        palette = np.array([
            hex_to_rgb(colors_map.get(i, "#000000"))
            for i in range(num_classes)
        ], dtype=np.uint8)

    mask_rgb = palette[mask]
    mask_rgb_pil = Image.fromarray(mask_rgb).convert("RGBA").resize(image.size, Image.NEAREST)       

    return mask_rgb_pil



def visualisation():          
    global image, orig_mask
    compare_mask = False    

    if orig_mask != None:
        compare_mask = st.toggle('Сравнить маски')    
    
    
    if compare_mask:    
        mask_rgb_pil = prediction(image, orig_mask)  
        alpha = st.slider("Альфа", 0.0, 1.0, 0.5, 0.05)
    else:
        mask_rgb_pil = prediction(image)
        alpha = st.slider("Альфа", 0.0, 1.0, 0.7, 0.05)  
    

    
    overlay = Image.blend(image.convert("RGBA"), mask_rgb_pil, alpha)
  
    image_comparison(
        img1=image,
        img2=overlay,
        label1="Оригинал",
        label2="Сегментация",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=False,
    )

    legend(compare_mask)


def legend(compare_mask):
    if compare_mask:
        palette = {
            'Неверно предсказанные' : [200, 0, 0],
            'Верно предсказанные': [0, 200, 0],
            'Не участвует в оценке': [40, 40, 40],
        }

        with st.sidebar:
            st.markdown("### 🗂 Легенда")
            for key, value in palette.items():
                r, g, b = value
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;margin-bottom:2px;">
                        <div style="width:16px;height:16px;background:rgb({r},{g},{b});
                                    border:1px solid #555;margin-right:8px;"></div>
                        {key}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        with st.sidebar:
            st.markdown("### 🗂 Легенда")
            for i in range(len(class_names)):
                c = colors_map.get(i, "#000000")
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;margin-bottom:2px;">
                        <div style="width:16px;height:16px;background:{c};
                                    border:1px solid #555;margin-right:8px;"></div>
                        {class_names[i]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def select_model(model_name):
    if model_name.startswith('UNET_EFFNET'):
        model = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=14,
        )
    elif model_name.startswith('UNET_VGG-19'):
        backbone = torchvision.models.vgg19_bn(weights=None, progress=False).to('cpu')
        model = SelfMadeUNet(14, backbone).to('cpu')
    elif model_name.startswith('DEEPLAB'):
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=14,
        )
    else:
        model = smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=14,
        )
    return model

@st.cache_resource(show_spinner=True)
def get_model(selected_model):
    checkpoint_path = parent_path + model_names[selected_model]
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model_state_dict = checkpoint['state_dict']

    model = select_model(selected_model)
    model = Model(model)
    model.load_state_dict(model_state_dict)
    model.eval()

    return model


def hex_to_rgb(hex_color):
    hex_color = hex_color.partition('#')[2]
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)







parent_path = 'models/'
model_names = {    
    'UNET_EFFNET-B3': 'unet_effnet_ignore-epoch=39.ckpt',    
    'UNET_EFFNET-B3 + SOFT_AUGM': 'unet_effnet_soft_augm-epoch=42.ckpt',
    'UNET_EFFNET-B3 + LOSS_COFF': 'unet_effnet_coff-epoch=44.ckpt',
    'UNET_EFFNET-B3 + CLAMP_WEIGHT': 'unet_effnet_clamp_weight-epoch=28.ckpt',
    'UNET_VGG-19': 'unet_vgg19_ignore-epoch=25.ckpt',
    'SEGFORMER_MIT-B2': 'segformer_ignore-epoch=46.ckpt',
    'DEEPLAB_V3+_RESNET50': 'deeplab_v3_plus_ignore-epoch=42.ckpt',
    # '': '.ckpt',
    # '': '.ckpt',
    # '': '.ckpt',
}

with st.sidebar:
    selected_model = st.selectbox('Сегментационные модели',
                                  [name for name in model_names])
model = get_model(selected_model)

with st.sidebar:
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        arg_example = 0
        orig_mask = None
    else:
        examples = [f for f in Path('examples/').iterdir() if f.suffix != '.pt']
        arg_example = st.pills('Примеры', [i+1 for i in range(len(examples))], default=1)
        image = Image.open(examples[arg_example-1]).convert("RGB")
        orig_mask = torch.load(examples[arg_example-1].with_suffix('.jpg.pt'))



with st.spinner('Модель обрабатывает изображение'):
    visualisation()