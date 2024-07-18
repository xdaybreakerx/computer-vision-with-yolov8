import yt_dlp
import os
import subprocess
import ffmpeg

# Function to ensure the directory exists
def ensure_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
# Function to check if ffmpeg is installed
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("ffmpeg is installed but there is an error with the installation.")
        raise e
    except FileNotFoundError:
        print("ffmpeg is not installed. Please install ffmpeg and ensure it is in your system's PATH.")
        exit(1)
        
# Download video from YouTube with specified filename
def download_video(url, output_path, filename):
    ydl_opts = {
        'outtmpl': os.path.join(output_path, filename),
        'format': 'best'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Trim video to 30 seconds
def trim_video(input_path, output_path):
    (
        ffmpeg
        .input(input_path, ss=0, t=30)
        .output(output_path)
        .run()
    )

def download_and_trim_video():
    # Ensure the ./video directory exists
    video_directory = "./video"
    ensure_directory(video_directory)

    # Example usage
    youtube_url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
    filename = "source.mp4"
    trimmed_filename = "source_trimmed.mp4"

    # Download the video
    download_video(youtube_url, video_directory, filename)
    print("Video downloaded from YouTube successfully")
    # Full path to the saved video
    video_path = os.path.join(video_directory, filename)
    trimmed_video_path = os.path.join(video_directory, trimmed_filename)

    # Check if ffmpeg is installed
    check_ffmpeg()
    # Trim the video to 30 seconds
    trim_video(video_path, trimmed_video_path)
    print("Video trimmed to 30 seconds successfully")
    
    
download_and_trim_video()