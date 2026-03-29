
https://github.com/user-attachments/assets/34d6b9f8-a35f-4aa9-9cd7-8a6e9e0669cd


## Quick Start

### 1. Installation
pip install -r requirements.txt

2.Download model and put on models\RealRestorer
https://huggingface.co/RealRestorer/RealRestorer

3.workflow
fp8_24G_Example.json 
<img width="831" height="680" alt="f8123ad41fe29a169bc0abfe40b6f7a9" src="https://github.com/user-attachments/assets/4aa5490e-f0df-451a-bab2-ad09aeab0237" />


fp8_2G_Example.json
<img width="1378" height="624" alt="c6997fbe8fcefedef119936e676545d3" src="https://github.com/user-attachments/assets/aeb6e6df-1436-4f6e-9d8b-f4588722cb7d" />

Task Prompts
Task	English Prompt	中文 Prompt
Blur Removal	Please deblur the image and make it sharper	请将图像去模糊，变得更清晰。
Compression Artifact Removal	Please restore the image clarity and artifacts.	请修复图像清晰度和伪影。
Lens Flare Removal	Please remove the lens flare and glare from the image.	请去除图像中的光晕和炫光。
Moire Removal	Please remove the moiré patterns from the image	请将图像中的摩尔条纹去除
Dehazing	Please dehaze the image	请将图像去雾。
Low-light Enhancement	Please restore this low-quality image, recovering its normal brightness and clarity.	请修复这张低质量图像，恢复其正常的亮度和清晰度。
Denoising	Please remove noise from the image.	请去除图像中的噪声。
Rain Removal	Please remove the rain from the image and restore its clarity.	请去除图像中的雨水并恢复图像清晰度
Reflection Removal	Please remove the reflection from the image.	请移除图像中的反光
