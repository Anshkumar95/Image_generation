import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import concatenate_videoclips, CompositeVideoClip, ImageClip
from TTS.api import TTS
from scipy.io.wavfile import write
import gradio as gr
import random
from pydub import AudioSegment
import numpy as np
import requests


# Directories
if not os.path.exists('generated_videos'):
    os.makedirs('generated_videos')


# Check if MPS (Apple Silicon GPU) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Load Text-to-Video Model
print("Loading Video Generation Model...This may take a while.")
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16)
pipe.to(device)

# Ensure all components are properly set to use MPS
pipe.text_encoder.to(device)
pipe.vae.to(device)
pipe.unet.to(device)


# Load Text-to-Speech Model (Coqui.ai)
tts_model = TTS('tts_models/en/ljspeech/tacotron2-DDC', progress_bar=False)


# Function to Generate Video
def generate_video(prompt):
    print("Generating Video...")
    video_frames = pipe(prompt, num_frames=16).frames  # Generates numpy arrays (frames)
    clips = []

    # Adjusting for the unexpected shape
    if len(video_frames.shape) == 5 and video_frames.shape[0] == 1:
        video_frames = video_frames[0]  # Remove the first batch dimension (1)
    
    # Now, video_frames should have shape (16, 256, 256, 3)
    if len(video_frames.shape) == 4 and video_frames.shape[0] == 16:
        for i in range(16):  # Iterate over each frame
            frame = video_frames[i]  # Extract a single frame of shape (256, 256, 3)

            # Convert frame to uint8 format (0-255)
            frame = (frame * 255).astype(np.uint8)

            # If the frame has an alpha channel, remove it
            if frame.shape[-1] == 4:
                frame = frame[..., :3]

            # Ensure the frame is of shape (256, 256, 3)
            if frame.shape == (256, 256, 3):
                clip = ImageClip(frame).set_duration(0.1)  # Display each frame for 0.1 seconds
                clips.append(clip)
            else:
                print(f"Skipping frame with invalid shape: {frame.shape}")
    else:
        raise ValueError(f"Unexpected shape of video_frames: {video_frames.shape}")

    # Concatenate all clips to form a video
    if len(clips) > 0:
        video_clip = concatenate_videoclips(clips, method="compose")
        video_clip_path = os.path.join('generated_videos', 'generated_video.mp4')
        video_clip.write_videofile(video_clip_path, fps=8, codec="libx264")
        return video_clip_path
    else:
        raise ValueError("No valid frames were generated.")

# Function to Generate Audio
def generate_audio(prompt):
    try:
        print("Generating Voice Over...")
        audio_path = os.path.join('generated_videos', 'generated_audio.wav')
        
        # Generate audio from the model
        audio = tts_model.tts(prompt)
        
        # Check if the audio is returned as a list
        if isinstance(audio, list):
            try:
                # Convert list to numpy array
                audio = np.array(audio, dtype=np.float32)
                print(f"Audio successfully converted to numpy array. Shape: {audio.shape}")
            except Exception as e:
                raise ValueError(f"Failed to convert audio list to numpy array: {str(e)}")
        
        # Check if audio is a valid numpy array
        if isinstance(audio, np.ndarray):
            try:
                # Save the audio as a .wav file
                sample_rate = 22050  # Common sample rate used by TTS systems
                write(audio_path, sample_rate, audio)
                print(f"Audio saved successfully at: {audio_path}")
                return audio_path
            except Exception as e:
                raise IOError(f"Failed to save audio file: {str(e)}")
        
        # If the audio is not a list or numpy array, log the type
        raise TypeError(f"Unexpected audio type returned: {type(audio)}")
    
    except Exception as e:
        # Log any error that occurs
        print(f"Error in generate_audio(): {str(e)}")
        return None


# Function to Add AI-Generated Background Music

def add_background_music(audio_path):
    print("Adding AI-Generated Background Music...")
    audio = AudioSegment.from_wav(audio_path)

    # Fetch music sample from a royalty-free source (Using Riffusion)
    response = requests.get('https://github.com/riffusion/riffusion-app/releases/download/v1.0/ambient-loop.wav')
    if response.status_code == 200:
        with open('generated_videos/background_music.wav', 'wb') as f:
            f.write(response.content)

    music = AudioSegment.from_wav('generated_videos/generated_audio.wav')
    music = music - 20  # Reduce volume of background music

    # Loop music to match the duration of the voice-over
    while len(music) < len(audio):
        music += music

    music = music[:len(audio)]  # Trim music to match audio length
    combined = audio.overlay(music)

    final_audio_path = os.path.join('generated_videos', 'final_audio_with_music.wav')
    combined.export(final_audio_path, format="wav")

    return final_audio_path


# Function to Combine Video & Audio

def combine_video_audio(video_path, audio_path):
    print("Combining Video and Audio...")
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)

    output_path = os.path.join('generated_videos', 'final_video.mp4')
    final_video.write_videofile(output_path, codec='libx264', fps=24)

    return output_path


# Gradio Interface

def main(prompt):
    video_path = generate_video(prompt)
    audio_path = generate_audio(prompt)
    enhanced_audio_path = add_background_music(audio_path)
    final_video_path = combine_video_audio(video_path, enhanced_audio_path)

    return final_video_path


demo = gr.Interface(fn=main, inputs="text", outputs="video", title="Enhanced Video Generation Tool with AI Music")

demo.launch()