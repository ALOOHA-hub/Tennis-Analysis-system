import argparse
from core.pipeline import TennisAnalysisPipeline
from core.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Tennis Analysis AI System")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the source tennis video")
    parser.add_argument("--output_video", type=str, required=True, help="Path to save the processed video")
    
    args = parser.parse_args()

    logger.info("--- Starting Tennis Analysis Application ---")
    logger.info(f"Input path: {args.input_video}")
    logger.info(f"Output path: {args.output_video}")

    pipeline = TennisAnalysisPipeline(
        input_video_path=args.input_video,
        output_video_path=args.output_video
    )
    
    try:
        pipeline.run()
    except Exception as e:
        logger.exception(f"An unexpected error occurred during execution: {e}")

if __name__ == "__main__":
    main()