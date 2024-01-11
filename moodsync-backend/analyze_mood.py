import torch
from torchvision import transforms
from PIL import Image

classes = ('happy', 'sad', 'focus', 'calm', 'angry')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('main_model.pth', map_location=device)

model.eval()

model.to(device)

def predict_emotion(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  

    with torch.no_grad():
        image = image.to(device)
        output = model(image)

    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()
    predicted_label = classes[predicted_class]

    return predicted_label
