from utils.config_loader import cfg
from core.pipeline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline(
        input_video_path=cfg['paths']['input_video'],
        output_video_path=cfg['paths']['output_video']
    )
    pipeline.run()
