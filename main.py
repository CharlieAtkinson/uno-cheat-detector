"""Real-time card detection and rule enforcement CLI."""
import argparse
import logging
from pathlib import Path
import cv2
from ultralytics import YOLO

from src.state_machine import UnoReferee

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CLASS_MAP = {
    0: "red_0", 1: "red_1", 2: "red_2", 3: "red_reverse", 4: "red_skip",
    5: "wildcard", 6: "green_0", 7: "green_1", 8: "green_2",
    9: "green_reverse", 10: "green_skip",
}


def run_pipeline(video_path: Path, weights_path: Path, output_path: Path, debug: bool):
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at: {weights_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    model = YOLO(str(weights_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to read video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    referee = UnoReferee(stability_threshold=3, grace_period=5)
    display_name = f"'{video_path.name}'" if " " in video_path.name else video_path.name

    logging.info("Processing %s...", video_path.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ts = round(frame_idx / fps, 2)
        results = model(frame, conf=0.5, imgsz=640, verbose=False)

        det_id, det_label = None, None
        if results[0].boxes and len(results[0].boxes) > 0:
            det_id = int(results[0].boxes[0].cls[0])
            det_label = CLASS_MAP.get(det_id)

        play = referee.process_frame(det_id, det_label, frame_idx, ts)
        if play:
            verdict = "VALID" if play.valid else "CHEAT DETECTED"
            logging.info("[%ss] Card: %s -> %s", ts, play.name, verdict)

        if debug:
            annotated = results[0].plot()
            cv2.imshow("Referee Debug Feed", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    with open(output_path, "w", encoding="utf-8") as f:
        for p in referee.history:
            f.write(
                f"{display_name},{p.class_id},{p.name},{p.frame_idx},{p.timestamp},{1 if p.valid else 0}\n"
            )
    logging.info("Results successfully written to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous CV Referee for UNO")
    parser.add_argument("video", type=Path, help="Path to input video file")
    parser.add_argument("--weights", type=Path, default=Path("weights/weights.pt"), help="Path to YOLO weights")
    parser.add_argument("--output", type=Path, default=Path("output.txt"), help="Path to export log")
    parser.add_argument("--debug", action="store_true", help="Display annotated OpenCV window")
    args = parser.parse_args()

    run_pipeline(args.video, args.weights, args.output, args.debug)