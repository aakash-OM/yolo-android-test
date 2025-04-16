package com.ultralytics.yolo.predict.detect;

import android.graphics.RectF;
import androidx.annotation.Keep;
import org.json.JSONException;
import org.json.JSONObject;

public class DetectedObject {
    // Existing code...

    // NEW METHOD: Convert to JSON
    public JSONObject toJSON() throws JSONException {
        JSONObject obj = new JSONObject();
        obj.put("x", boundingBox.centerX());
        obj.put("y", boundingBox.centerY());
        obj.put("w", boundingBox.width());
        obj.put("h", boundingBox.height());
        obj.put("class", label);
        obj.put("confidence", confidence);
        return obj;
    }
}


// Changes: Add a toJSON() method to serialize detections.