from pytube import YouTube
import os

# Function to ensure the directory exists
def ensure_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# Download video from YouTube with specified filename
def download_video(url, output_path, filename):
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    stream.download(output_path, filename=filename)

# Ensure the ./video directory exists
video_directory = "./video"
ensure_directory(video_directory)

# Example usage
youtube_url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
filename = "source.mp4"

download_video(youtube_url, video_directory, filename)

# Full path to the saved video
video_path = os.path.join(video_directory, filename)