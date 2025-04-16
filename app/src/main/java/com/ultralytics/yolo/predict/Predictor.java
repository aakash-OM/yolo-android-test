package com.ultralytics.yolo.predict;

// Existing imports...
import org.json.JSONArray;
import org.json.JSONException;
import com.ultralytics.yolo.predict.detect.DetectedObject;

public abstract class Predictor {
    public static int INPUT_SIZE = 736; // Force 736x736 (as per metadata.yaml)

    // Existing code...

    // Modify method to return JSONArray
    public abstract JSONArray predict(Bitmap bitmap) throws JSONException;

    // Example implementation (to be done in subclass like TfliteDetector.java)
    protected JSONArray convertDetectionsToJSON(List<DetectedObject> detections) throws JSONException {
        JSONArray jsonArray = new JSONArray();
        for (DetectedObject obj : detections) {
            jsonArray.put(obj.toJSON());
        }
        return jsonArray;
    }
}


/*
changes:
Update INPUT_SIZE handling to ensure 736x736 (from metadata).
Modify predict() to return a JSONArray.
*/
