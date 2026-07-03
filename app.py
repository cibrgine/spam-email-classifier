import gradio as gr
from transformers import pipeline
import os

MODEL_PATH = os.path.join('models', 'spam_classifier_distilbert')

try:
    # Load the fine-tuned model using Hugging Face pipeline
    # The pipeline handles tokenization and inference automatically
    classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, device=0)
    print("Transformer model loaded successfully on GPU.")
except Exception as e:
    print(f"Failed to load on GPU or model missing. Falling back to CPU. Error: {e}")
    try:
        classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, device=-1)
        print("Transformer model loaded successfully on CPU.")
    except Exception as e:
        print(f"Error loading model: {e}")
        classifier = None

def classify_email(email_text):
    if classifier is None:
        return "Error: Model not loaded.", "0.0%"
    
    if not email_text.strip():
        return "Please enter some text.", "0.0%"
        
    # Predict
    # Result format: [{'label': 'LABEL_1', 'score': 0.99}]
    # LABEL_0 is Safe, LABEL_1 is Spam
    try:
        result = classifier(email_text, truncation=True, max_length=512)[0]
        label_id = result['label']
        confidence = result['score'] * 100
        
        # Map LABEL_1 to Spam, LABEL_0 to Safe
        if label_id == 'LABEL_1' or label_id == '1':
            label = "🚨 SPAM"
        else:
            label = "✅ SAFE"
        
        return label, f"{confidence:.2f}%"
    except Exception as e:
        return f"Error: {e}", "0.0%"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📧 Deep Learning Spam Classifier (DistilBERT)")
    gr.Markdown("Enter the content of an email below to check if it's spam or safe using our fine-tuned Transformer model.")
    
    with gr.Row():
        with gr.Column(scale=2):
            email_input = gr.Textbox(
                lines=10, 
                placeholder="Paste the email content here...", 
                label="Email Content"
            )
            submit_btn = gr.Button("Classify Email", variant="primary")
            
        with gr.Column(scale=1):
            label_output = gr.Label(label="Classification")
            conf_output = gr.Textbox(label="Confidence Score")
            
    gr.Examples(
        examples=[
            ["Hi Team, Just a reminder that our weekly sync is tomorrow at 10 AM. Please review the attached document beforehand. Thanks!"],
            ["CONGRATULATIONS! You have been selected to win a FREE $1000 Walmart Gift Card. Click the link below to claim your prize now!! Hurry, offer expires in 24 hours!"],
            ["Dear customer, your PayPal account has been restricted due to suspicious activity. Please verify your identity by clicking here: http://paypal-update-security.com"],
            ["Hey John, it was great catching up yesterday. Let me know if you want to grab lunch next week."]
        ],
        inputs=email_input
    )

    submit_btn.click(
        fn=classify_email, 
        inputs=email_input, 
        outputs=[label_output, conf_output]
    )

if __name__ == "__main__":
    # Use 127.0.0.1 for local access
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
