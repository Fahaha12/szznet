import argparse
import torch
from model.SwinTransformerForRegression import SwinTransformerForRegression
from torchvision import transforms
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Validate the trained model')
    parser.add_argument('--image-path', type=str, required=True, help='path to the input image')
    parser.add_argument('--model-path', type=str, required=True, help='path to the trained model checkpoint')
    args = parser.parse_args()

    # 加载训练好的模型
    model = SwinTransformerForRegression()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 定义图像预处理转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为模型输入尺寸
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 加载并预处理输入图像
    image = Image.open(args.image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批次维度

    # 设置表型数据
    phenotype = torch.tensor([[68.17, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)

    # 将图像和表型数据传递给模型进行预测
    with torch.no_grad():
        output = model(image, phenotype)
        predicted_value = output.item()

    print(f'Predicted value: {predicted_value:.4f}')

if __name__ == '__main__':
    main()