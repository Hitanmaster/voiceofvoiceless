import os
import json
import cv2
import numpy as np
import torch
import gradio as gr
from unified_detector import RealTimeSignDetector

# Initialize the detector
detector = RealTimeSignDetector()

def predict_video(video_path):
    if not video_path:
        return None, "No video provided", []
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    detector.sequence_buffer.clear()
    detector.sentence.clear()
    
    detected_signs = []
    
    output_path = "output_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated = detector.process_frame(frame)
        
        if out_writer is None:
            h, w, _ = frame.shape
            out_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
            
        out_writer.write(annotated)
        
    cap.release()
    if out_writer:
        out_writer.release()
        
    sentence_text = " ".join(detector.sentence) if detector.sentence else "No clear sign sequence recognized."
    return output_path, sentence_text, detector.sentence

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown(
        """
        # 🤟 SignLanguageAI — Voice of the Voiceless
        ### Real-Time Indian Sign Language (ISL) to Speech System
        *BiLSTM + Self-Attention Neural Network (Normalized Holistic Features, 60-Frame Rolling Window)*
        """
    )
    
    with gr.Tab("📹 Video File Translation"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload ISL Video (.mp4)")
                btn_process = gr.Button("Translate Sign Video", variant="primary")
            with gr.Column():
                video_output = gr.Video(label="Annotated Video with Landmark Tracking")
                sentence_output = gr.Textbox(label="Recognized Sentence / Sign", interactive=False)
                
        btn_process.click(predict_video, inputs=[video_input], outputs=[video_output, sentence_output])
        
    with gr.Tab("📚 Supported Vocabulary"):
        gr.Markdown(f"### Currently Trained on {len(detector.classes)} Classes:")
        with gr.Row():
            classes_display = ", ".join([f"`{c}`" for c in detector.classes if c.lower() != "idle"])
            gr.Markdown(classes_display)

    with gr.Tab("🧠 Architecture & Specs"):
        gr.Markdown(
            """
            - **Input Vector:** 150-dimensional normalized keypoint vector (v2: wrist-relative handshapes + 8 pose landmarks, scale/position invariant; legacy v1: 138 raw).
            - **Temporal Length:** 60 frames (~2.0 seconds at 30 FPS).
            - **Architecture:** 2-Layer Bidirectional LSTM + Self-Attention + Dense Classifier.
            - **Inference Latency:** < 15ms per frame on CPU / < 4ms on CUDA GPU.
            """
        )

if __name__ == "__main__":
    demo.launch(share=False)
