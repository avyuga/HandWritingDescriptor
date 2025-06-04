import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import tritonclient.http as httpclient
from data_models import (PredictionResponse, SimpleResponse,
                         TranscribationRequest, UpdateRequest)
from database import db
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from utils import detection, misc, recognition

client = httpclient.InferenceServerClient(url="triton-server:8000")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.connect()
    yield
    # Shutdown
    await db.close()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(user_id: str = Form(), request_id: str = Form(), file: UploadFile = File()):
    
    img_bytes: BytesIO = await file.read()
    image = np.array(Image.open(BytesIO(img_bytes)))

    t1 = time.time()
    img_resized, target_ratio, _ = misc.resize_aspect_ratio(
        image, 640, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
    )
    ratio_h = ratio_w = 1 / target_ratio

    detector_input = misc.normalizeMeanVariance(img_resized)
    detector_input = np.transpose(detector_input, (2, 0, 1))[None, ...]

    # http request to triton for detection model
    infer_input = httpclient.InferInput("input", detector_input.shape, datatype="FP32")
    infer_input.set_data_from_numpy(detector_input, binary_data=True)

    responce = client.infer(model_name="detection", inputs=[infer_input])

    maps = responce.as_numpy('output')
    text_map = maps[0, :, :, 0]
    link_map = maps[0, :, :, 1]


    bboxes, _ = detection.getDetBoxes(
        text_map, link_map, 
        text_threshold=0.7, link_threshold=0.4, 
        low_text=0.4, estimate_num_chars=None
    )

    bboxes = detection.adjustResultCoordinates(bboxes, ratio_w, ratio_h)
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    result = []
    for bbox in bboxes:
        image_list = misc.get_image_list([bbox], image_gray, model_height=64)

        coord = [item[0] for item in image_list]
        img_list = [item[1][None, None, ...].astype(np.float32) / 255. for item in image_list]

        # http request to triton to recognition model
        infer_input = httpclient.InferInput("input1", img_list[0].shape, datatype="FP32")
        infer_input.set_data_from_numpy(img_list[0], binary_data=True)
        
        responce = client.infer(model_name="recognition", inputs=[infer_input])
        preds = responce.as_numpy('output')

        result1 = recognition.postprocess(preds)
        low_confident_idx = [i for i,item in enumerate(result1) if (item[1] < 0.1)]

        if len(low_confident_idx) > 0:
            img_list2 = [img_list[i] for i in low_confident_idx]

            infer_input = httpclient.InferInput("input1", img_list2[0].shape, datatype="FP32")
            infer_input.set_data_from_numpy(img_list2[0], binary_data=True)
            
            responce = client.infer(model_name="recognition", inputs=[infer_input])
            preds2 = responce.as_numpy('output')

            result2 = recognition.postprocess(preds2)

        for i, zipped in enumerate(zip(coord, result1)):
            box, pred1 = zipped
            if i in low_confident_idx:
                pred2 = result2[low_confident_idx.index(i)]
                if pred1[1]>pred2[1]:
                    result.append((box, pred1[0], pred1[1]))
                else:
                    result.append((box, pred2[0], pred2[1]))
            else:
                result.append((box, pred1[0], pred1[1]))

    result = [r for r in result if r[2] >= 0.6] #  remove unconfident detections
    t2 = time.time()
    # Store prediction in database
    prediction_data = {
        "user_id": user_id,
        "request_id": request_id,
        "image_id": uuid.uuid4(),
        "created_at": datetime.now(),
        "processing_time": t2 - t1,  # You can (probably) calculate this
        "detections": [],  # Add your processed detections here
        "user_rating": None,
        "user_transcription": None
    }

    await db.insert_prediction(prediction_data)

    phrase = ' '.join(r[1] for r in result) if len(result) > 0 else ''
    score = float(np.mean([r[2] for r in result])) if len(result) > 0 else 0.0
    return PredictionResponse(
        request_id=request_id,
        prediction=phrase,
        confidence=score,
        status="ok"
    )

@app.post("/rate", response_model=SimpleResponse)
async def update_rating(request: UpdateRequest):
    """Update the rating for a prediction.
    
    Args:
        request: RatingUpdateRequest containing request_id and rating
        
    Returns:
        RatingUpdateResponse indicating success or failure
    """
    try:
        # Check if prediction exists
        prediction = await db.get_prediction(request.request_id)
        if not prediction:
            raise HTTPException(
                status_code=404,
                detail=f"Prediction with request_id {request.request_id} not found"
            )
        
        # Update the rating
        success = await db.update_rating(request.request_id, request.rating)
        
        if success:
            return SimpleResponse(
                success=True,
                message="Rating updated successfully"
            )
        else:
            return SimpleResponse(
                success=False,
                message="Failed to update rating"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/transcribe", response_model=SimpleResponse)
async def update_transcription(request: TranscribationRequest):
    """Update the user transcription for a prediction.
    
    Args:
        request: TranscriptionUpdateRequest containing request_id and transcription
        
    Returns:
        TranscriptionUpdateResponse indicating success or failure
    """
    try:
        # Check if prediction exists
        prediction = await db.get_prediction(request.request_id)
        if not prediction:
            raise HTTPException(
                status_code=404,
                detail=f"Prediction with request_id {request.request_id} not found"
            )
        
        # Update the transcription
        success = await db.update_transcription(request.request_id, request.transcription)
        
        if success:
            return SimpleResponse(
                success=True,
                message="Transcription updated successfully"
            )
        else:
            return SimpleResponse(
                success=False,
                message="Failed to update transcription"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
