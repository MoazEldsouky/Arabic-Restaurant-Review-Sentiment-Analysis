import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model from the Hugging Face Model Hub
model_name = "moazx/AraBERT-Restaurant-Sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define the device to run inference on (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model.to(device)



# Define function for sentiment analysis
def predict_sentiment(review):
    # Step 1: Tokenization
    encoded_text = tokenizer(
        review, padding=True, truncation=True, max_length=256, return_tensors="pt"
    )

    # Move input tensors to the appropriate device
    input_ids = encoded_text["input_ids"].to(device)
    attention_mask = encoded_text["attention_mask"].to(device)

    # Step 2: Inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Step 3: Prediction with probabilities
    probs = torch.softmax(outputs.logits, dim=-1)
    probs = (
        probs.squeeze().cpu().numpy()
    )  # Convert to numpy array and remove the batch dimension

    # Map predicted class index to label
    label_map = {0: 'سلبي', 1: 'إيجابي'}

    output_dict = {label_map[i]: float(probs[i]) for i in range(len(probs))}
    return output_dict


# Create Gradio interface
examples = [
    ["كانت تجربتي في هذا المطعم رائعة والطعام كان لذيذاً للغاية"],
    ["المطعم يجنن والاكل تحفة"],
    ["المطعم ما عجبني، الطعم مو حلو والخدمة كانت سيئة جداً، والموظفين ما كانوا محترمين. الأسعار غالية مقارنة بالجودة. ما بنصح فيه."],
    ["الطعام لا يستحق الثمن المدفوع، جودة سيئة للغاية."],
    ["الاكل كان جميل جدا والناس هناك محترمه جدا,"],
    ["الأكل وحش والخدمة سيئة، مش هرجع تاني للمطعم ده."],
    ["المطعم وايد حلو وأكلهم طيب بشكل مو طبيعي! الخدمة عندهم ممتازة والأجواء جداً مريحة. بالتأكيد راح أزورهم مرة ثانية وأنصح الكل يجربهم!"],
    ["تجربتي في هذا المطعم كانت مخيبة للآمال. الأطعمة كانت جافة ومُضيعة للوقت، وخدمة العملاء كانت بطيئة ولا تلبي التوقعات. بالإضافة إلى ذلك، الأسعار مبالغ فيها مقارنة بجودة الطعام المقدمة. لن أعيد زيارة هذا المكان مرة أخرى."],
    
]

description_html = """
<p>This model was trained by Moaz Eldsouky. You can find more about me here:</p>
<p>GitHub: <a href="https://github.com/MoazEldsouky">GitHub Profile</a></p>
<p>LinkedIn: <a href="https://www.linkedin.com/in/moaz-eldesouky-762288251/">LinkedIn Profile</a></p>
<p>Kaggle: <a href="https://www.kaggle.com/moazeldsokyx">Kaggle Profile</a></p>
<p>Email: <a href="mailto:moazeldsoky8@gmail.com">moazeldsoky8@gmail.com</a></p>
"""


iface = gr.Interface(
    fn=predict_sentiment,
    inputs = gr.Textbox(placeholder='أدخل تقييماً لمطعم باللغة العربية'),
    outputs = gr.Label(num_top_classes=2, min_width=360),
    title = "تحليل المشاعر لتقيمات المطاعم باللغة العربية",
    article = description_html,
    allow_flagging = "auto",
    examples = examples,
)
iface.launch()



