package com.ultralytics.yolo.predict.detect;

import android.content.Context;
import android.graphics.Bitmap;
import org.tensorflow.lite.Interpreter;
import com.ultralytics.yolo.ImageProcessing;
import com.ultralytics.yolo.predict.Predictor;
import org.json.JSONArray;
import org.json.JSONException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class TfliteDetector extends Predictor {
    private Interpreter tflite;
    private List<DetectedObject> detections;

    public TfliteDetector(Context context) {
        super(context);
    }

    @Override
    public void loadModel(YoloModel yoloModel, boolean useGpu) throws Exception {
        // Load TFLite model from assets
        tflite = new Interpreter(loadModelFile(context, yoloModel.modelPath));
        loadLabels(context.getAssets(), yoloModel.metadataPath); // Load class labels
    }

    @Override
    public JSONArray predict(Bitmap bitmap) throws JSONException {
        // Preprocess image (resize + normalize)
        ByteBuffer inputBuffer = ImageProcessing.bitmapToInputBuffer(bitmap);
        // Run inference
        float[][][] output = new float[1][8400][7]; // Shape depends on your model
        tflite.run(inputBuffer, output);
        // Postprocess to JSON
        detections = parseOutput(output[0]);
        return convertDetectionsToJSON(detections);
    }

    private List<DetectedObject> parseOutput(float[][] output) {
        List<DetectedObject> results = new ArrayList<>();
        for (float[] det : output) {
            if (det[4] > 0.5) { // Confidence threshold
                RectF box = new RectF(det[0], det[1], det[0] + det[2], det[1] + det[3]);
                results.add(new DetectedObject(det[4], box, (int) det[5], labels.get((int) det[5])));
            }
        }
        return results;
    }
}