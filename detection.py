import json

import cv2
import numpy as np
import torch
from ultralytics import YOLO

BALL_CLASS = 0
GOALKEEPER_CLASS = 1
PLAYER_CLASS = 2
REFEREE_CLASS = 3
LIST_CLIPS = ['clip_1']


def team_color(clip_name):
    """
    Function to return the home and away team colors for a given clip.
    :param clip_name: Name of the clip out of the three clips.
    :return: Color of the home and away teams.
    """
    if clip_name == 'clip_1':
        return (250, 229, 140), (17, 255, 252)


def remove_background(player_roi):
    """
    Function to remove the background from the player ROI and return the mean color.

    :param player_roi: The player's ROI.

    :return:The mean color of the player's ROI.
    """
    lab = cv2.cvtColor(player_roi, cv2.COLOR_BGR2LAB)
    # Threshold the LAB image to get the mask
    mask = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    non_black_mask = (mask == 255).astype(np.uint8)
    # Get the mean color of the player's ROI
    mean_color = cv2.mean(player_roi, mask=non_black_mask)[:3]
    return mean_color


def get_center_color(player_roi):
    """
    Function to get the color of the center of the player ROI.
    :param player_roi: The player's ROI.
    :return:The color of the center of the player's ROI.
    """
    center_x, center_y = player_roi.shape[1] // 2, player_roi.shape[0] // 2
    center_region = player_roi[center_y - 2:center_y + 3, center_x - 2:center_x + 3]
    center_color = cv2.mean(center_region)[:3]
    return center_color


def count_players_by_team(player_rois, home_team_color, away_team_color, color_function, classes):
    """
    Function to count the number of players in each team based on color similarity.
    :param player_rois: List of player regions of interest (ROIs) in the image.
    :param home_team_color: The BGR color representation of the home team's uniform.
    :param away_team_color: The BGR color representation of the away team's uniform.
    :param color_function: Function to calculate the color of a player's ROI.
    :param classes: List of classes for each ROI in the image.

    :return: The number of players in the home and away teams, and the colors of the bounding boxes.
    """
    home_team_count = 0
    away_team_count = 0
    colors_boxes = []
    for player_roi, class_box in zip(player_rois, classes):
        if class_box != PLAYER_CLASS:
            if class_box == BALL_CLASS:
                colors_boxes.append([255, 255, 255])
            else:
                colors_boxes.append([0, 0, 0])
            continue
        player_color = color_function(player_roi)

        # Compare the center color with the home and away team colors
        home_color_diff = sum(abs(c - player_color[i]) for i, c in enumerate(home_team_color))
        away_color_diff = sum(abs(c - player_color[i]) for i, c in enumerate(away_team_color))

        # Increment player count for the corresponding team
        if home_color_diff < away_color_diff:
            home_team_count += 1
            colors_boxes.append(home_team_color)
        else:
            away_team_count += 1
            colors_boxes.append(away_team_color)

    return home_team_count, away_team_count, colors_boxes


# Function to get the ROIs of the top half of the bounding boxes
def get_half_rois(image, boxes):
    """
    Function to get the top half of the bounding boxes as regions of interest (ROIs).
    :param image: Frame of the video.
    :param boxes: Bounding boxes of the objects in the image.
    :return: List of top half ROIs.
    """
    rois = []
    for i, (xyxy, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
        bbox = [int(coord) for coord in xyxy]
        rois.append(image[bbox[1]:round((bbox[3] + bbox[1]) / 2), bbox[0]:bbox[2]])

    return rois


def write_jsonl(frame, home_team, away_team, ball_loc, refs):
    """
    Function to write the frame data to a JSONL file.
    :param frame: Frame number.
    :param home_team: Number of players in the home team.
    :param away_team: Number of players in the away team.
    :param ball_loc: Last known location of the ball.
    :param refs: Number of referees in the frame.
    :return: JSONL string of the frame.
    """
    json_data = {
        "frame": frame,
        "home_team": home_team,
        "away_team": away_team,
        "refs": refs,
        "ball_loc": ball_loc
    }
    return json.dumps(json_data) + '\n'


# Function to draw bounding boxes and labels on the image
def draw_boxes(image, boxes, colors, ball_id):
    """
    Function to draw bounding boxes and labels on the image.
    :param image: Frame of the video.
    :param boxes: Bounding boxes of the objects in the image.
    :param colors: Colors of the bounding boxes.
    :param ball_id: Ball ID.
    :return: Void.
    """
    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[i], 2)
        if boxes.cls[i] == BALL_CLASS:
            cv2.putText(image, f'id:{int(ball_id)} Ball', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 2)
        else:
            cv2.putText(image, f'id:{str(int(boxes.id[i]))}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i],
                        2)


def get_data(clip_name, model):
    """
    Function to get the data from the video clip.
    :param clip_name: Name of the clip out of the three clips.
    :param model: Model to use for object detection.
    :return: Void.
    """

    # Get the home and away team colors
    home_team_color, away_team_color = team_color(clip_name)
    # Open the video file
    cap = cv2.VideoCapture(f'Data/{clip_name}.mp4')
    assert cap.isOpened(), "Error reading video file"
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Get the frame rate of the video to avoid changes in the speed of the video
    framerate = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(f'output_{clip_name}.mp4', fourcc, framerate, (640, 640))
    ball_coords = [0, 0, 0, 0]
    jsonl = []
    # Ball id is initialized as an empty tensor to avoid errors when writing the JSONL file
    ball_id = torch.tensor([])
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        # Resize the frame to 640x640 to fit the model's input size
        im0 = cv2.resize(im0, (640, 640))

        # Perform object detection with tracking and persistence in the ids
        results = model.track(im0, verbose=True, show=False, show_boxes=False, persist=True)

        boxes = results[0].boxes
        # Get the top half of the bounding boxes as ROIs
        rois = get_half_rois(im0, boxes)
        # Count the number of players in each team based on color similarity
        home_number, away_number, colors_boxes = count_players_by_team(rois, home_team_color, away_team_color,
                                                                       remove_background, boxes.cls)
        frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # If the ball has not been detected yet, get the id of the ball
        if ball_id.numel() == 0:
            ball_id = boxes.id[boxes.cls == BALL_CLASS]
        # Draw the bounding boxes and labels on the image
        draw_boxes(im0, boxes, colors_boxes, ball_id)
        # Write the JSONL file in each fifth frame
        if frame % 5 == 0:
            aux_ball_coords = boxes.xywh[boxes.cls == BALL_CLASS]
            if len(aux_ball_coords) != 0:
                ball_coords = [int(coord) for coord in aux_ball_coords[0]]
            jsonl += write_jsonl(frame, home_number, away_number,
                                 ball_coords, int(torch.sum(boxes.cls == REFEREE_CLASS)))
        # Show the built video
        cv2.imshow('Video', im0)
        # Write the frame to the output video
        out.write(im0)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Write the JSONL file
    with open(f'output_{clip_name}.jsonl', 'w') as file:
        for json_line in jsonl:
            file.write(json_line)
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    """
    Main function to get the data from the three video clips.
    """
    for clip in LIST_CLIPS:
        model_yolo = YOLO('models/yolov8n-football.onnx', 'detect')
        get_data(clip, model_yolo)
