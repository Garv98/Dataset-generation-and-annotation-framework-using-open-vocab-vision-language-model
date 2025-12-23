"""Quick test to verify Qwen2-VL-2B is actually working"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from io import BytesIO

print("Loading Qwen2-VL-2B-Instruct (lighter, CPU-friendly)...")
print(f"CUDA available: {torch.cuda.is_available()}")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Model loaded on {device}")

# Test with a sample image
print("\nTesting with a sample car image...")
url = "https://www.thoughtco.com/thmb/P3xlCIkCRVRzOZ5kj_mQuWRLqfk=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-90049312-5c51be0346e0fb0001066c98.jpg"
try:
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    
    question = "Does this image clearly show 'car license plate'? Answer YES or NO, then explain."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    
    print("Generating answer...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    
    print(f"\n🤖 Qwen2-VL Answer: {answer}")
    print("\n✅ Qwen2-VL is working correctly!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Using local test instead...")
