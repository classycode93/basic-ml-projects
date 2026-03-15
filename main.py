
from src.video_processing import extract_frames
from src.compare import compare_sample_frames

def main():

    real_video = "data/real_video.avi"
    fake_video = "data/fake_video.avi"

    real_frames = "outputs/real_frames"
    fake_frames = "outputs/fake_frames"

    extract_frames(real_video, real_frames)
    extract_frames(fake_video, fake_frames)

    compare_sample_frames(real_frames, fake_frames)

if __name__ == "__main__":
    main()
